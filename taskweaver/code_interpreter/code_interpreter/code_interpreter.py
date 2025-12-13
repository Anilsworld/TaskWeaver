import os
from typing import Dict, Literal, Optional

from injector import inject

from taskweaver.code_interpreter.code_executor import CodeExecutor
from taskweaver.code_interpreter.code_interpreter import CodeGenerator
from taskweaver.code_interpreter.code_verification import code_snippet_verification, format_code_correction_message
from taskweaver.code_interpreter.interpreter import Interpreter
from taskweaver.logging import TelemetryLogger
from taskweaver.memory import Memory, Post
from taskweaver.memory.attachment import AttachmentType
from taskweaver.module.event_emitter import PostEventProxy, SessionEventEmitter
from taskweaver.module.tracing import Tracing, get_tracer, tracing_decorator
from taskweaver.role import Role
from taskweaver.role.role import RoleConfig, RoleEntry


class CodeInterpreterConfig(RoleConfig):
    def _configure(self):
        self.use_local_uri = self._get_bool(
            "use_local_uri",
            self.src.get_bool(
                "use_local_uri",
                True,
            ),
        )
        self.max_retry_count = self._get_int("max_retry_count", 3)

        # for verification
        self.code_verification_on = self._get_bool("code_verification_on", False)
        self.allowed_modules = self._get_list(
            "allowed_modules",
            [
                "pandas",
                "matplotlib",
                "numpy",
                "sklearn",
                "scipy",
                "seaborn",
                "datetime",
                "typing",
            ],
        )
        self.blocked_functions = self._get_list(
            "blocked_functions",
            [
                "eval",
                "exec",
                "execfile",
                "compile",
                "open",
                "input",
                "raw_input",
                "reload",
                "__import__",
            ],
        )

        self.code_prefix = self._get_str("code_prefix", "")


def update_verification(
    response: PostEventProxy,
    status: Literal["NONE", "INCORRECT", "CORRECT"] = "NONE",
    error: str = "No verification is done.",
):
    response.update_attachment(status, AttachmentType.verification)
    response.update_attachment(
        error,
        AttachmentType.code_error,
    )


def update_execution(
    response: PostEventProxy,
    status: Literal["NONE", "SUCCESS", "FAILURE"] = "NONE",
    result: str = "No code is executed.",
):
    response.update_attachment(status, AttachmentType.execution_status)
    response.update_attachment(result, AttachmentType.execution_result)


class CodeInterpreter(Role, Interpreter):
    @inject
    def __init__(
        self,
        generator: CodeGenerator,
        executor: CodeExecutor,
        logger: TelemetryLogger,
        tracing: Tracing,
        event_emitter: SessionEventEmitter,
        config: CodeInterpreterConfig,
        role_entry: RoleEntry,
    ):
        super().__init__(config, logger, tracing, event_emitter, role_entry)

        self.generator = generator
        self.generator.set_alias(self.alias)
        self.generator.configure_verification(
            code_verification_on=self.config.code_verification_on,
            allowed_modules=self.config.allowed_modules,
            blocked_functions=self.config.blocked_functions,
        )

        self.executor = executor
        self.logger = logger
        self.tracing = tracing
        self.event_emitter = event_emitter
        self.retry_count = 0

        self.plugin_description = "    " + "\n    ".join(
            [f"{plugin.spec.plugin_description()}" for plugin in generator.plugin_pool],
        )

        self.logger.info(f"{self.alias} initialized successfully.")

    def get_intro(self) -> str:
        return self.intro.format(plugin_description=self.plugin_description)

    def update_session_variables(self, session_variables: Dict[str, str]):
        self.logger.info(f"Updating session variables: {session_variables}")
        self.executor.update_session_var(session_variables)

    @tracing_decorator
    def reply(
        self,
        memory: Memory,
        prompt_log_path: Optional[str] = None,
        **kwargs: ...,
    ) -> Post:
        post_proxy = self.event_emitter.create_post_proxy(self.alias)
        post_proxy.update_status("generating code")
        self.executor.start()
        self.generator.reply(
            memory,
            post_proxy,
            prompt_log_path,
        )

        if post_proxy.post.message is not None and post_proxy.post.message != "":  # type: ignore
            update_verification(
                post_proxy,
                "NONE",
                "No code verification is performed.",
            )
            update_execution(post_proxy, "NONE", "No code is executed.")

            return post_proxy.end()

        code = next(
            (a for a in post_proxy.post.attachment_list if a.type == AttachmentType.reply_content),
            None,
        )

        if code is None:
            # no code is generated is usually due to the failure of parsing the llm output
            self.tracing.set_span_status("ERROR", "Failed to generate code.")

            update_verification(
                post_proxy,
                "NONE",
                "No code verification is performed.",
            )
            update_execution(
                post_proxy,
                "NONE",
                "No code is executed due to code generation failure.",
            )
            post_proxy.update_message("Failed to generate code.")
            if self.retry_count < self.config.max_retry_count:
                error_message = self.generator.format_output_revision_message()
                post_proxy.update_attachment(
                    error_message,
                    AttachmentType.revise_message,
                )
                post_proxy.update_send_to("CodeInterpreter")
                self.retry_count += 1
            else:
                self.retry_count = 0
                post_proxy.update_send_to("User")
                post_proxy.update_message(
                    "❌ Failed to generate code after 3 attempts. "
                    "The workflow may be too complex. Please try:\n"
                    "1. Simplifying your request (fewer apps/steps)\n"
                    "2. Breaking it into smaller workflows\n"
                    "3. Contacting support if the issue persists"
                )

            return post_proxy.end()

        self.tracing.set_span_attribute("code", code.content)
        post_proxy.update_status("verifying code")

        self.tracing.set_span_attribute("code_verification_on", self.config.code_verification_on)
        self.logger.info(f"Code to be verified: {code.content}")
        with get_tracer().start_as_current_span("CodeInterpreter.verify_code") as span:
            span.set_attribute("code", code.content)
            # ✅ ARCHITECTURAL FIX: Pass session variables for constraint enforcement
            # Following battle-tested plugin pattern (composio_action, form_collect)
            code_verify_errors = code_snippet_verification(
                code.content,
                self.config.code_verification_on,
                allowed_modules=self.config.allowed_modules,
                blocked_functions=self.config.blocked_functions,
                session_variables=self.executor.session_variables,  # ✅ NEW: Enable hard constraints
            )

        if code_verify_errors is None:
            update_verification(
                post_proxy,
                "NONE",
                "No code verification is performed.",
            )
        elif len(code_verify_errors) > 0:
            self.logger.info(
                f"Code verification finished with {len(code_verify_errors)} errors.",
            )

            code_error = "\n".join(code_verify_errors)
            update_verification(post_proxy, "INCORRECT", code_error)

            self.tracing.set_span_status("ERROR", "Code verification failed.")
            self.tracing.set_span_attribute("verification_error", code_error)

            if self.retry_count < self.config.max_retry_count:
                # Track error history to detect repeated mistakes
                error_summary = code_error[:150]  # First 150 chars as fingerprint
                
                # Check if LLM is repeating the same error
                if hasattr(self, '_last_error') and self._last_error == error_summary:
                    # LLM is stuck! Add explicit guidance
                    repetition_warning = (
                        f"\n\n**ALERT: RETRY {self.retry_count + 1}/{self.config.max_retry_count}** - "
                        "You're generating the SAME error again!\n\n"
                        "**What's happening:** Your last code had the exact same syntax error.\n"
                        "**What to try:** Change your approach completely:\n"
                        "- Format dict fields on separate lines (better readability)\n"
                        "- Double-check ALL brackets before 'depends_on' field\n"
                        "- Count opening { and closing } braces carefully\n\n"
                        "**Original error:**\n"
                    )
                    code_error = repetition_warning + code_error
                
                # Store error fingerprint for next iteration
                self._last_error = error_summary
                
                # ✅ Parso/Black already provides detailed error context
                post_proxy.update_message(code_error)
                post_proxy.update_attachment(
                    format_code_correction_message(code.content, code_error),
                    AttachmentType.revise_message,
                )
                post_proxy.update_send_to("CodeInterpreter")
                self.retry_count += 1
                self.logger.warning(f"⚠️ [RETRY] Code verification failed, retry attempt {self.retry_count}/{self.config.max_retry_count}")
            else:
                # 🚨 MAX RETRIES REACHED - Send to User
                self.logger.error(f"🚨 [MAX_RETRY] Reached max retry limit ({self.config.max_retry_count})")
                self.retry_count = 0
                self._last_error = None  # Clear error history
                post_proxy.update_send_to("User")
                post_proxy.update_message(
                    f"❌ Code verification failed after {self.config.max_retry_count} attempts. "
                    "The generated code contains errors. Please try:\n"
                    "1. Simplifying your request\n"
                    "2. Breaking it into smaller workflows\n"
                    f"Error: {code_error[:200]}"
                )

            # add execution status and result
            update_execution(
                post_proxy,
                "NONE",
                "No code is executed due to code verification failure.",
            )
            return post_proxy.end()
        elif len(code_verify_errors) == 0:
            update_verification(post_proxy, "CORRECT", "No error is found.")
            self._last_error = None  # Clear error history on success

        executable_code = f"{code.content}"
        full_code_prefix = None
        if self.config.code_prefix:
            full_code_prefix = f"{self.config.code_prefix}\n" "## CODE START ##\n"
            executable_code = f"{full_code_prefix}{executable_code}"

        # 🔍 EARLY TRUNCATION DETECTION: Check if WORKFLOW dict is incomplete
        if "WORKFLOW" in executable_code and "result = WORKFLOW" in executable_code:
            # Check for common truncation patterns
            if executable_code.rstrip().endswith(("${from_step:", '"${from_step:', "'${from_step:")):
                self.logger.error("🚨 [TRUNCATION] Detected incomplete WORKFLOW dict - code ends mid-parameter")
                truncation_error = (
                    "❌ Code generation was truncated (incomplete WORKFLOW dict). "
                    "The workflow is too complex for the current token limit. "
                    "Please simplify your request by:\n"
                    "1. Using fewer apps/platforms (try 2-3 instead of 4+)\n"
                    "2. Breaking it into smaller workflows\n"
                    "3. Reducing the number of steps"
                )
                update_verification(post_proxy, "INCORRECT", truncation_error)
                update_execution(post_proxy, "NONE", "Code not executed due to truncation.")
                post_proxy.update_message(truncation_error)
                post_proxy.update_send_to("User")
                self.retry_count = 0
                return post_proxy.end()

        post_proxy.update_status("executing code")
        self.logger.info(f"Code to be executed: {executable_code}")

        exec_result = self.executor.execute_code(
            exec_id=post_proxy.post.id,
            code=executable_code,
        )

        code_output = self.executor.format_code_output(
            exec_result,
            with_code=False,
            use_local_uri=self.config.use_local_uri,
            code_mask=full_code_prefix,
        )

        update_execution(
            post_proxy,
            status="SUCCESS" if exec_result.is_success else "FAILURE",
            result=code_output,
        )

        # add artifact paths
        post_proxy.update_attachment(
            [
                (
                    a.file_name
                    if os.path.isabs(a.file_name) or not self.config.use_local_uri
                    else os.path.join(self.executor.execution_cwd, a.file_name)
                )
                for a in exec_result.artifact
            ],  # type: ignore
            AttachmentType.artifact_paths,
        )

        post_proxy.update_message(
            self.executor.format_code_output(
                exec_result,
                with_code=True,  # the message to be sent to the user should contain the code
                use_local_uri=self.config.use_local_uri,
                code_mask=full_code_prefix,
            ),
            is_end=True,
        )

        if exec_result.is_success or self.retry_count >= self.config.max_retry_count:
            if not exec_result.is_success and self.retry_count >= self.config.max_retry_count:
                # 🚨 MAX RETRIES REACHED - Send to User, not Planner
                self.logger.error(f"🚨 [MAX_RETRY] Reached max retry limit ({self.config.max_retry_count})")
                post_proxy.update_send_to("User")
                post_proxy.update_message(
                    f"❌ Failed to execute code after {self.config.max_retry_count} attempts. "
                    "The workflow may be too complex or contains errors. Please try:\n"
                    "1. Simplifying your request\n"
                    "2. Breaking it into smaller workflows\n"
                    "3. Using fewer apps/platforms"
                )
            self.retry_count = 0
        else:
            post_proxy.update_send_to("CodeInterpreter")
            post_proxy.update_attachment(
                self.generator.format_code_revision_message(),
                AttachmentType.revise_message,
            )
            self.retry_count += 1
            self.logger.warning(f"⚠️ [RETRY] Code execution failed, retry attempt {self.retry_count}/{self.config.max_retry_count}")

        if not exec_result.is_success:
            self.tracing.set_span_status("ERROR", "Code execution failed.")

        reply_post = post_proxy.end()

        self.tracing.set_span_attribute("out.from", reply_post.send_from)
        self.tracing.set_span_attribute("out.to", reply_post.send_to)
        self.tracing.set_span_attribute("out.message", reply_post.message)
        self.tracing.set_span_attribute("out.attachments", str(reply_post.attachment_list))

        return reply_post

    def close(self) -> None:
        self.generator.close()
        self.executor.stop()
        super().close()
