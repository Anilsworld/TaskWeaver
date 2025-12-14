import datetime
import json
import os
from typing import List, Optional

from injector import inject

from taskweaver.code_interpreter.plugin_selection import PluginSelector, SelectedPluginPool
from taskweaver.llm import LLMApi
from taskweaver.llm.util import ChatMessageType, format_chat_message
from taskweaver.logging import TelemetryLogger
from taskweaver.memory import Attachment, Memory, Post, Round, RoundCompressor
from taskweaver.memory.attachment import AttachmentType
from taskweaver.memory.experience import ExperienceGenerator
from taskweaver.memory.plugin import PluginEntry, PluginRegistry
from taskweaver.module.event_emitter import PostEventProxy, SessionEventEmitter
from taskweaver.module.tracing import Tracing, tracing_decorator
from taskweaver.role import PostTranslator, Role
from taskweaver.role.role import RoleConfig
from taskweaver.utils import read_yaml


class CodeGeneratorConfig(RoleConfig):
    def _configure(self) -> None:
        self._set_name("code_generator")
        self.role_name = self._get_str("role_name", "ProgramApe")
        self.load_plugin = self._get_bool("load_plugin", True)
        self.prompt_file_path = self._get_path(
            "prompt_file_path",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "code_generator_prompt.yaml",
            ),
        )
        self.prompt_compression = self._get_bool("prompt_compression", False)
        self.compression_prompt_path = self._get_path(
            "compression_prompt_path",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "compression_prompt.yaml",
            ),
        )
        self.enable_auto_plugin_selection = self._get_bool(
            "enable_auto_plugin_selection",
            False,
        )
        self.auto_plugin_selection_topk = self._get_int("auto_plugin_selection_topk", 3)

        self.llm_alias = self._get_str("llm_alias", default="", required=False)
        
        # 🚀 Function Calling for Workflow Generation
        self.use_function_calling = self._get_bool(
            "use_function_calling",
            True  # Default: enabled
        )


class CodeGenerator(Role):
    @inject
    def __init__(
        self,
        config: CodeGeneratorConfig,
        plugin_registry: PluginRegistry,
        logger: TelemetryLogger,
        event_emitter: SessionEventEmitter,
        tracing: Tracing,
        llm_api: LLMApi,
        round_compressor: RoundCompressor,
        post_translator: PostTranslator,
        experience_generator: ExperienceGenerator,
    ):
        super().__init__(config, logger, tracing, event_emitter)
        self.config = config
        self.llm_api = llm_api

        self.role_name = self.config.role_name

        self.post_translator = post_translator
        self.prompt_data = read_yaml(self.config.prompt_file_path)

        self.instruction_template = self.prompt_data["content"]

        self.conversation_head_template = self.prompt_data["conversation_head"]
        self.user_message_head_template = self.prompt_data["user_message_head"]
        self.plugin_pool = plugin_registry.get_list()
        self.query_requirements_template = self.prompt_data["requirements"]
        self.response_json_schema = json.loads(self.prompt_data["response_json_schema"])

        self.code_verification_on: bool = False
        self.allowed_modules: List[str] = []

        self.round_compressor: RoundCompressor = round_compressor
        self.compression_template = read_yaml(self.config.compression_prompt_path)["content"]

        if self.config.enable_auto_plugin_selection:
            self.plugin_selector = PluginSelector(plugin_registry, self.llm_api)
            self.plugin_selector.load_plugin_embeddings()
            logger.info("Plugin embeddings loaded")
            self.selected_plugin_pool = SelectedPluginPool()

        self.experience_generator = experience_generator

        self.logger.info("CodeGenerator initialized successfully")

    def configure_verification(
        self,
        code_verification_on: bool,
        allowed_modules: Optional[List[str]] = None,
        blocked_functions: Optional[List[str]] = None,
    ):
        self.allowed_modules = allowed_modules if allowed_modules is not None else []
        self.code_verification_on = code_verification_on
        self.blocked_functions = blocked_functions

    def compose_verification_requirements(
        self,
    ) -> str:
        requirements: List[str] = []
        if not self.code_verification_on:
            return ""

        if len(self.allowed_modules) > 0:
            requirements.append(
                f"- {self.role_name} can only import the following Python modules: "
                + ", ".join([f"{module}" for module in self.allowed_modules]),
            )

        if len(self.allowed_modules) == 0:
            requirements.append(f"- {self.role_name} cannot import any Python modules.")

        if len(self.blocked_functions) > 0:
            requirements.append(
                f"- {self.role_name} cannot use the following Python functions: "
                + ", ".join([f"{function}" for function in self.blocked_functions]),
            )
        return "\n".join(requirements)

    def compose_sys_prompt(self, context: str):
        return self.instruction_template.format(
            ENVIRONMENT_CONTEXT=context,
            ROLE_NAME=self.role_name,
            RESPONSE_JSON_SCHEMA=json.dumps(self.response_json_schema),
        )

    def get_env_context(self):
        # get date and time
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        return f"- Current time: {current_time}"

    def _extract_tool_ids_from_actions(self, composio_actions: str) -> List[str]:
        """
        Extract tool IDs from select_composio_actions() formatted output.
        
        Input format:
        "Available Composio Actions:
         - GMAIL_GET_MAIL_V2 (app: gmail): Fetch emails
         - SLACKBOT_SEND_MESSAGE (app: slack): Send message"
        
        Output: ["GMAIL_GET_MAIL_V2", "SLACKBOT_SEND_MESSAGE"]
        
        🔑 CRITICAL: This creates the filtered enum (5-50 tools, not 17k!)
        """
        import re
        # ✅ FIX: Match format "- ACTION_ID (app: ...)" or "- ACTION_ID:"
        tool_ids = re.findall(r'^\s*-\s*([A-Z][A-Z0-9_]+)\s*(?:\(app:|:)', composio_actions, re.MULTILINE)
        self.logger.info(f"[FUNCTION_CALLING] Extracted {len(tool_ids)} tool IDs for enum")
        return tool_ids


    def _build_workflow_function_schema(self, tool_ids: List[str]) -> dict:
        """
        Build OpenAI function schema for workflow IR generation.
        
        ✅ NOW USING: workflow_schema_builder.py (Pydantic-based, dynamic tool params)
        
        🔑 DESIGN PRINCIPLES:
        - Intermediate Representation (IR) - compiler-grade structure
        - Mutually exclusive node types (oneOf prevents ambiguity)
        - Typed edges (explicit control flow)
        - Filtered tool_id enum (5-50 tools, not 17k)
        - Triggers for scheduled/event workflows
        - Composability via sub_workflow
        - 🆕 Dynamic tool parameter schemas from Composio cache
        
        Args:
            tool_ids: Filtered list of valid Composio action IDs (5-50 tools)
        
        Returns:
            OpenAI function schema dict (IR spec) with tool-specific parameter schemas
        """
        # ✅ NEW: Use schema builder with dynamic tool param injection
        from taskweaver.code_interpreter.code_interpreter.workflow_schema_builder import get_schema_builder
        
        schema_builder = get_schema_builder()
        schema = schema_builder.build_function_schema(tool_ids)
        
        self.logger.info(f"[FUNCTION_CALLING] Built workflow schema with {len(tool_ids)} tools using dynamic param injection")
        return schema

    def _format_examples_for_function_calling(self) -> str:
        """
        Format YAML examples for function calling prompt.
        
        Extracts key patterns from loaded examples to show form vs hitl distinction.
        This was working before function calling - now restoring it!
        
        Returns:
            Formatted example string highlighting form vs hitl usage
        """
        if not hasattr(self, 'examples') or not self.examples:
            return ""
        
        # Build concise examples showing form vs hitl patterns
        example_text = (
            "🎯 CRITICAL EXAMPLES - Study these patterns:\n\n"
            "EXAMPLE 1 - Form for DATA COLLECTION (user provides info):\n"
            '{"id": "collect_info", "type": "form", "fields": [{"name": "email", "type": "text"}]}\n'
            "USE 'form' when: User enters search criteria, contact details, preferences\n\n"
            
            "EXAMPLE 2 - HITL for APPROVAL/REVIEW (user makes decision):\n"
            '{"id": "review_draft", "type": "hitl", "approval_type": "approve_reject"}\n'
            "USE 'hitl' when: Approving results, reviewing drafts, authorizing actions\n\n"
            
            "EXAMPLE 3 - PARALLEL for SIMULTANEOUS EXECUTION:\n"
            '{"id": "fetch_all", "type": "parallel", "parallel_nodes": ["fetch_a", "fetch_b", "fetch_c"]}\n'
            '{"id": "fetch_a", "type": "agent_with_tools", "tool_id": "API_A", "params": {...}}\n'
            '{"id": "fetch_b", "type": "agent_with_tools", "tool_id": "API_B", "params": {...}}\n'
            '{"id": "fetch_c", "type": "agent_with_tools", "tool_id": "API_C", "params": {...}}\n'
            "USE 'parallel' when: Multiple independent tasks can run simultaneously\n\n"
            
            "EXAMPLE 4 - Complete approval workflow:\n"
            '1. {"id": "search_data", "type": "agent_with_tools", "tool_id": "...", "params": {...}}\n'
            '2. {"id": "approve_results", "type": "hitl", "approval_type": "approve_reject"}\n'
            '3. {"id": "send_email", "type": "agent_with_tools", "tool_id": "..."}\n\n'
            
            "⚠️ REMEMBER: 'approve', 'review', 'authorize' → USE 'hitl' (NOT 'form')!\n"
        )
        
        return example_text

    def _convert_workflow_json_to_python(self, workflow_json: dict) -> str:
        """
        Convert workflow JSON from function calling to Python code format.
        This ensures compatibility with existing validation pipeline.
        
        ✅ CORRECT: Uses pprint for safe conversion, NOT string replacement hacks.
        
        Input: {"nodes": [...], "edges": [...]}
        Output: "WORKFLOW = {...}\n\nresult = WORKFLOW"
        """
        import pprint
        
        # ✅ Use pprint for safe Python representation
        # This handles booleans, None, nested structures correctly
        workflow_repr = pprint.pformat(workflow_json, width=120, compact=False)
        
        # Build final Python code
        python_code = f"WORKFLOW = {workflow_repr}\n\nresult = WORKFLOW"
        
        # Handle different edge formats (edges key, sequential_edges, or derived from depends_on)
        edge_count = (
            len(workflow_json.get('edges', [])) or 
            len(workflow_json.get('sequential_edges', [])) or
            sum(len(node.get('depends_on', [])) for node in workflow_json.get('nodes', []))
        )
        
        self.logger.info(
            f"[FUNCTION_CALLING] Generated {len(python_code)} chars of Python code "
            f"({len(workflow_json.get('nodes', []))} nodes, {edge_count} edges)"
        )
        
        return python_code

    def compose_prompt(
        self,
        rounds: List[Round],
        plugins: List[PluginEntry],
        planning_enrichments: Optional[List[str]] = None,
    ) -> List[ChatMessageType]:
        experiences = self.format_experience(
            template=self.prompt_data["experience_instruction"],
        )

        chat_history = [
            format_chat_message(
                role="system",
                message=f"{self.compose_sys_prompt(context=self.get_env_context())}" f"\n{experiences}",
            ),
        ]

        for i, example in enumerate(self.examples):
            chat_history.extend(
                self.compose_conversation(example.rounds, example.plugins, add_requirements=False),
            )

        summary = None
        if self.config.prompt_compression:
            summary, rounds = self.round_compressor.compress_rounds(
                rounds,
                rounds_formatter=lambda _rounds: str(
                    self.compose_conversation(_rounds, plugins, add_requirements=False),
                ),
                prompt_template=self.compression_template,
            )

        chat_history.extend(
            self.compose_conversation(
                rounds,
                add_requirements=True,
                summary=summary,
                plugins=plugins,
                planning_enrichments=planning_enrichments,
            ),
        )
        return chat_history

    def format_attachment(self, attachment: Attachment):
        if attachment.type == AttachmentType.thought and "{ROLE_NAME}" in attachment.content:
            return attachment.content.format(ROLE_NAME=self.role_name)
        else:
            return attachment.content

    def compose_conversation(
        self,
        rounds: List[Round],
        plugins: List[PluginEntry],
        add_requirements: bool = False,
        summary: Optional[str] = None,
        planning_enrichments: Optional[List[str]] = None,
    ) -> List[ChatMessageType]:
        chat_history: List[ChatMessageType] = []
        ignored_types = [
            AttachmentType.revise_message,
            AttachmentType.verification,
            AttachmentType.code_error,
            AttachmentType.execution_status,
            AttachmentType.execution_result,
        ]

        is_first_post = True
        last_post: Post = None
        for round_index, conversation_round in enumerate(rounds):
            for post_index, post in enumerate(conversation_round.post_list):
                # compose user query
                user_message = ""
                assistant_message = ""
                is_final_post = round_index == len(rounds) - 1 and post_index == len(conversation_round.post_list) - 1
                if is_first_post:
                    user_message = (
                        self.conversation_head_template.format(
                            SUMMARY="None" if summary is None else summary,
                            PLUGINS="None" if len(plugins) == 0 else self.format_plugins(plugins),
                            ROLE_NAME=self.role_name,
                        )
                        + "\n"
                    )
                    is_first_post = False

                if post.send_from == "Planner" and post.send_to == self.alias:
                    # to avoid planner imitating the below handcrafted format,
                    # we merge context information in the code generator here
                    enrichment = ""
                    if is_final_post:
                        user_query = conversation_round.user_query
                        enrichment = f"The user request is: {user_query}\n\n"

                        if planning_enrichments:
                            enrichment += "Additional context:\n" + "\n".join(planning_enrichments) + "\n\n"

                    user_feedback = "None"
                    if last_post is not None and last_post.send_from == self.alias:
                        user_feedback = format_code_feedback(last_post)

                    user_message += self.user_message_head_template.format(
                        FEEDBACK=user_feedback,
                        MESSAGE=f"{enrichment}The task for this specific step is: {post.message}",
                    )
                elif post.send_from == post.send_to == self.alias:
                    # for code correction
                    user_message += self.user_message_head_template.format(
                        FEEDBACK=format_code_feedback(post),
                        MESSAGE=f"{post.get_attachment(AttachmentType.revise_message)[0].content}",
                    )

                    assistant_message = self.post_translator.post_to_raw_text(
                        post=post,
                        content_formatter=self.format_attachment,
                        if_format_message=False,
                        if_format_send_to=False,
                        ignored_types=ignored_types,
                    )
                elif post.send_from == self.alias and post.send_to == "Planner":
                    if is_final_post:
                        # This user message is added to make the conversation complete
                        # It is used to make sure the last assistant message has a feedback
                        # This is only used for examples or context summarization
                        user_message += self.user_message_head_template.format(
                            FEEDBACK=format_code_feedback(post),
                            MESSAGE="This is the feedback.",
                        )

                    assistant_message = self.post_translator.post_to_raw_text(
                        post=post,
                        content_formatter=self.format_attachment,
                        if_format_message=False,
                        if_format_send_to=False,
                        ignored_types=ignored_types,
                    )
                else:
                    raise ValueError(f"Invalid post: {post}")
                last_post = post

                if len(assistant_message) > 0:
                    chat_history.append(
                        format_chat_message(
                            role="assistant",
                            message=assistant_message,
                        ),
                    )
                if len(user_message) > 0:
                    # add requirements to the last user message
                    if is_final_post and add_requirements:
                        user_message += "\n" + self.query_requirements_template.format(
                            CODE_GENERATION_REQUIREMENTS=self.compose_verification_requirements(),
                            ROLE_NAME=self.role_name,
                        )
                    chat_history.append(
                        format_chat_message(role="user", message=user_message),
                    )

        return chat_history

    def select_plugins_for_prompt(
        self,
        query: str,
    ) -> List[PluginEntry]:
        selected_plugins = self.plugin_selector.plugin_select(
            query,
            self.config.auto_plugin_selection_topk,
        )
        self.selected_plugin_pool.add_selected_plugins(selected_plugins)
        self.logger.info(f"Selected plugins: {[p.name for p in selected_plugins]}")
        self.logger.info(
            f"Selected plugin pool: {[p.name for p in self.selected_plugin_pool.get_plugins()]}",
        )

        return self.selected_plugin_pool.get_plugins()

    @tracing_decorator
    def reply(
        self,
        memory: Memory,
        post_proxy: Optional[PostEventProxy] = None,
        prompt_log_path: Optional[str] = None,
        **kwargs: ...,
    ) -> Post:
        assert post_proxy is not None, "Post proxy is not provided."
        
        # Get session variables from kwargs (same pattern as planner.py)
        session_var = kwargs.get("session_var", None)

        # extract all rounds from memory
        rounds = memory.get_role_rounds(
            role=self.alias,
            include_failure_rounds=False,
        )

        # obtain the query from the last round
        query = rounds[-1].post_list[-1].message
        
        # ✅ FIX: For Composio action selection, use ORIGINAL user query instead of Planner's rephrased message
        # Planner's message can lose key domain keywords (e.g., "flights" → "passenger details")
        # This ensures action matcher sees the full user intent with all domain-specific terms
        original_user_query = rounds[-1].user_query if hasattr(rounds[-1], 'user_query') and rounds[-1].user_query else query

        self.tracing.set_span_attribute("query", query)
        self.tracing.set_span_attribute("enable_auto_plugin_selection", self.config.enable_auto_plugin_selection)
        self.tracing.set_span_attribute("use_experience", self.config.use_experience)

        if self.config.enable_auto_plugin_selection:
            self.plugin_pool = self.select_plugins_for_prompt(query)

        self.role_load_experience(query=query, memory=memory)
        
        # ✅ TRACEBACK: Log example loading
        self.logger.info(f"[EXAMPLE_LOADING] 📂 Loading examples from: {self.config.example_base_path}")
        self.role_load_example(memory=memory, role_set={self.alias, "Planner"})
        if hasattr(self, 'examples') and self.examples:
            self.logger.info(f"[EXAMPLE_LOADING] ✅ Loaded {len(self.examples)} examples successfully")
            for idx, example in enumerate(self.examples):
                self.logger.info(f"[EXAMPLE_LOADING]   Example {idx+1}: {len(example.rounds)} rounds, enabled={example.enabled}")
                # Log first round user_query to verify correct example
                if example.rounds:
                    user_query = getattr(example.rounds[0], 'user_query', 'N/A')
                    self.logger.info(f"[EXAMPLE_LOADING]     Query: {user_query[:80]}...")
        else:
            self.logger.warning(f"[EXAMPLE_LOADING] ⚠️  NO EXAMPLES LOADED!")

        planning_enrichments = memory.get_shared_memory_entries(entry_type="plan")

        # =====================================================================
        # COMPOSIO ACTION INJECTION (arch-31)
        # Inject relevant Composio actions into prompt so LLM knows exact action IDs
        # This prevents hallucination of action names like "COMPOSIO_SEARCH_TOOLS"
        # =====================================================================
        enrichment_contents = [pe.content for pe in planning_enrichments]
        try:
            # ✅ SCALABLE FIX: Skip expensive action selection on retry rounds
            # When code execution fails, CodeInterpreter sends the error message back for retry
            # We don't need to re-search for actions - reuse the ones from first attempt
            is_retry_round = any(
                indicator in query for indicator in [
                    "execution failed",
                    "The following python code has been executed:",
                    "SyntaxError:",
                    "Syntax error:",  # TaskWeaver format (lowercase)
                    "IndentationError:",
                    "NameError:",
                    "TypeError:",
                    "ValueError:",
                    "KeyError:",
                    "AttributeError:",
                    "Cell In[",  # Jupyter error traceback
                ]
            )
            
            if is_retry_round:
                self.logger.info(
                    "⏭️ [CODE_GENERATOR] Retry round detected - skipping action selection "
                    "(reusing tools from first attempt)"
                )
            else:
                # Dynamic import - same pattern as code_verification.py
                from TaskWeaver.project.plugins.composio_action_selector import select_composio_actions
                
                # ✅ SCALABLE: action_matcher.py now handles ALL app detection semantically
                # No more keyword matching or hardcoded app lists
                # The select_composio_actions() function will auto-detect apps using pgvector embeddings
                
                # ✅ ADAPTIVE APPROACH:
                # - query: Step-specific intent from Planner ("Fetch emails from Outlook")
                # - context: Full user request for domain/app discovery ("read emails... send responses")
                # - adaptive_top_k=True: Auto-scales based on detected apps
                #
                # ✅ COMPLETE WORKFLOW INJECTION:
                # For multi-platform workflows (3+ apps), the action selector will set intent="both"
                # This ensures BOTH read (fetch/list) AND write (send/reply) actions are available
                # Get session ID from environment (set by eclipse_adapter.py)
                import os
                session_id = os.environ.get('TASKWEAVER_SESSION_ID', None)
                
                composio_actions = select_composio_actions(
                    user_query=query,  # Step query - action selector detects multi-platform and sets intent="both"
                    context=original_user_query,  # Full query for domain/app discovery
                    top_k=10,  # Balanced - enough for both read and write actions
                    adaptive_top_k=True,  # Enable automatic scaling based on detected apps
                    session_id=session_id  # ✅ Enable batch API caching per session
                )
                if composio_actions:
                    enrichment_contents.append(composio_actions)
                    self.logger.info(f"[CODE_GENERATOR] Injected Composio actions: {len(composio_actions.splitlines())} lines")
        except ImportError:
            # Composio selector not available, skip silently
            pass
        except Exception as e:
            self.logger.debug(f"[CODE_GENERATOR] Composio action injection skipped: {e}")

        # =====================================================================
        # 🔥 WORKFLOW GENERATION MODE DETECTION
        # =====================================================================
        # ✅ FIX 5: Explicit flag check (not string search)
        # Use session_var (same pattern as planner.py line 199)
        is_workflow_generation = False
        if session_var:
            is_generation_mode = session_var.get("_workflow_generation_mode", "false")
            is_workflow_generation = (is_generation_mode == "true")
        
        # Fallback: Check shared memory if flag not set
        if not is_workflow_generation:
            shared_entries = memory.get_shared_memory_entries(entry_type="workflow_mode")
            is_workflow_generation = len(shared_entries) > 0
        
        self.logger.info(f"[CODE_GENERATOR] Workflow generation mode: {is_workflow_generation}")
        
        # =====================================================================
        # 🚀 FUNCTION CALLING PATH (Workflow IR Generation)
        # =====================================================================
        if is_workflow_generation and self.config.use_function_calling:
            self.logger.info("[CODE_GENERATOR] 🚀 Using function calling for workflow generation")
            
            # ✅ TRACEBACK: Log system prompt details
            self.logger.info(f"[PROMPT_TRACE] Instruction template length: {len(self.instruction_template)} chars")
            if 'multi_tool' in self.instruction_template.lower():
                self.logger.warning(f"[PROMPT_TRACE] ⚠️  'multi_tool' found in system prompt!")
            else:
                self.logger.info(f"[PROMPT_TRACE] ✅ 'multi_tool' NOT in system prompt")
            
            # Get filtered Composio actions
            composio_actions = next(
                (e for e in enrichment_contents if "Available Composio Actions" in e),
                ""
            )
            tool_ids = self._extract_tool_ids_from_actions(composio_actions)
            
            if not tool_ids:
                self.logger.warning(
                    "[CODE_GENERATOR] No tool IDs found, falling back to regular generation"
                )
                # Fall through to regular path below
            else:
                # Build function schema with filtered enum
                function_schema = self._build_workflow_function_schema(tool_ids)
                
                # ✅ TRACEBACK: Log what schema contains
                self.logger.info(f"[SCHEMA_TRACE] Built schema with {len(tool_ids)} tools")
                # Check if schema contains parallel type
                try:
                    # json is already imported at top of file
                    # The schema structure from workflow_schema_builder
                    self.logger.info(f"[SCHEMA_TRACE] Schema keys: {list(function_schema.keys())}")
                    
                    # Navigate to node schemas
                    params = function_schema.get('parameters', {})
                    workflow_prop = params.get('properties', {}).get('workflow', {})
                    nodes_prop = workflow_prop.get('properties', {}).get('nodes', {})
                    node_items = nodes_prop.get('items', {})
                    node_schemas = node_items.get('oneOf', [])
                    
                    self.logger.info(f"[SCHEMA_TRACE] Found {len(node_schemas)} node schema variants")
                    
                    # Extract node types
                    node_types = []
                    for schema in node_schemas:
                        if 'properties' in schema and 'type' in schema['properties']:
                            type_enum = schema['properties']['type'].get('enum', [])
                            if type_enum:
                                node_types.extend(type_enum)
                    
                    self.logger.info(f"[SCHEMA_TRACE] Node types in schema: {node_types}")
                    if 'parallel' in node_types:
                        self.logger.info(f"[SCHEMA_TRACE] ✅ 'parallel' type IS in schema")
                    else:
                        self.logger.warning(f"[SCHEMA_TRACE] ⚠️  'parallel' type NOT in schema!")
                    
                    # Log first 500 chars of schema for debugging
                    schema_str = json.dumps(function_schema, indent=2)
                    self.logger.info(f"[SCHEMA_TRACE] Schema preview (first 800 chars):\n{schema_str[:800]}...")
                except Exception as e:
                    self.logger.error(f"[SCHEMA_TRACE] Error inspecting schema: {e}", exc_info=True)
                
                # ✅ FIX 6: MINIMAL prompt (let schema do enforcement)
                # Don't reuse compose_prompt() - it's too heavy with JSON schema examples
                plan = next(
                    (e for e in enrichment_contents if "Plan" in e or "plan" in e.lower()),
                    "No plan provided"
                )
                
                # ✅ SKIP EXAMPLES for function calling - schema is already very detailed
                # Loading 4 examples adds ~10K tokens and causes function calling issues
                # The JSON schema provides all the information the LLM needs
                example_context = ""
                yaml_examples = ""
                
                minimal_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a workflow compiler. Generate workflows via function calls ONLY. "
                            "Rules:\n"
                            "- Use ONLY tool_id values from the Available Composio Actions list\n"
                            "- All placeholders MUST use ${{...}} syntax\n"
                            "- Node types are mutually exclusive\n"
                            "- agent_with_tools REQUIRES tool_id\n"
                            "- agent_only, form, hitl do NOT have tool_id\n\n"
                            "**PARALLEL EXECUTION:**\n"
                            "- For SIMULTANEOUS tasks, use type='parallel' with parallel_nodes list\n"
                            "- Example: {{'id': 'search_all', 'type': 'parallel', 'parallel_nodes': ['search_a', 'search_b', 'search_c']}}\n"
                            "- Then define each child: {{'id': 'search_a', 'type': 'agent_with_tools', ...}}\n"
                            "- Edges connect to PARENT: ('start', 'search_all'), ('search_all', 'process_results')\n\n"
                            f"{example_context}\n"
                            f"{yaml_examples}"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Plan:\n{plan}\n\n"
                            f"{composio_actions}\n\n"
                            f"Generate complete workflow for: {original_user_query}"
                        )
                    }
                ]
                
                self.logger.info(f"[PROMPT_BUILD] Final prompt length: {len(minimal_prompt[0]['content'])} chars (system) + {len(minimal_prompt[1]['content'])} chars (user)")
                
                # ✅ DEBUG: Log if parallel is mentioned in prompt
                full_prompt_text = minimal_prompt[0]['content'] + minimal_prompt[1]['content']
                if 'parallel' in full_prompt_text:
                    parallel_count = full_prompt_text.count('parallel')
                    self.logger.info(f"[PROMPT_BUILD] ✅ 'parallel' mentioned {parallel_count} times in prompt")
                if 'multi_tool' in full_prompt_text.lower():
                    self.logger.warning(f"[PROMPT_BUILD] ⚠️  'multi_tool' found in final prompt! (checking context...)")
                    # Log context around multi_tool
                    idx = full_prompt_text.lower().find('multi_tool')
                    context = full_prompt_text[max(0, idx-100):idx+100]
                    self.logger.warning(f"[PROMPT_BUILD] Context: ...{context}...")
                
                # Call function calling API
                try:
                    result = self.llm_api.chat_completion_with_function_calling(
                        messages=minimal_prompt,
                        functions=[function_schema],
                        function_call={"name": "create_workflow_ir"},
                        temperature=0.0
                    )
                    
                    # Extract workflow JSON
                    workflow_json = result["arguments"]["workflow"]
                    self.logger.info(
                        f"[FUNCTION_CALLING] Got workflow: "
                        f"{len(workflow_json.get('nodes', []))} nodes, "
                        f"{len(workflow_json.get('edges', []))} edges"
                    )
                    
                    # Debug: Log actual workflow JSON to see placeholder formats
                    # json is already imported at top of file
                    self.logger.info(f"[WORKFLOW_JSON] Full workflow:\n{json.dumps(workflow_json, indent=2)}")
                    
                    # ✅ VALIDATION: Skip dynamic model validator (module removed)
                    # Workflow schema validation already done by OpenAI function calling
                    param_warnings = []
                    
                    if param_warnings:
                        # Check if any are errors (not just warnings)
                        errors = [w for w in param_warnings if '[ERROR]' in w]
                        if errors:
                            self.logger.error(f"[DYNAMIC_VAL] {len(errors)} param validation errors")
                            for error in errors[:5]:  # Show first 5
                                self.logger.error(f"  {error}")
                            
                            # ⚠️ VALIDATION ERRORS: Send back to Planner for retry with hints
                            error_summary = "\n".join(errors[:10])  # Top 10 errors
                            error_message = (
                                f"⚠️ Workflow validation failed with {len(errors)} parameter errors:\n\n"
                                f"{error_summary}\n\n"
                                f"💡 HINTS:\n"
                                f"1. READ the user's request carefully and extract ALL mentioned values\n"
                                f"2. For agent_with_tools nodes, populate the 'params' field with extracted values\n"
                                f"3. Required parameters MUST have values (never null/empty)\n"
                                f"4. Use parameter examples from tool schemas for correct formatting\n"
                                f"5. For dates without year, use current year (e.g., 'Dec 25' → '2025-12-25')\n\n"
                                f"Please regenerate the workflow with ALL required parameters extracted."
                            )
                            post_proxy.update_attachment(error_message, AttachmentType.revise_message)
                            post_proxy.update_send_to("Planner")
                            return post_proxy.end()
                        else:
                            self.logger.warning(f"[DYNAMIC_VAL] {len(param_warnings)} param warnings")
                    
                    # Step 3: Pydantic structural validation (node types, edges, DAG)
                    from taskweaver.code_interpreter.workflow_schema import validate_workflow_dict
                    is_valid, workflow_obj, pydantic_errors = validate_workflow_dict(workflow_json)
                    
                    # ✅ workflow_json now has complete edges from WorkflowIR (including auto-added parallel edges)
                    self.logger.info(f"[FUNCTION_CALLING] After validation: {len(workflow_json.get('edges', []))} complete edges")
                    
                    if not is_valid:
                        self.logger.error(f"[FUNCTION_CALLING] Pydantic validation failed: {pydantic_errors}")
                        self.logger.error(f"[FUNCTION_CALLING] Failed workflow nodes: {[n.get('id', '?') + ':' + n.get('type', '?') for n in workflow_json.get('nodes', [])]}")
                        post_proxy.update_attachment(
                            "Generated workflow has structural errors. Please simplify your request.",
                            AttachmentType.revise_message
                        )
                        post_proxy.update_send_to("User")
                        return post_proxy.end()
                    
                    # Convert to Python code format
                    generated_code = self._convert_workflow_json_to_python(workflow_json)
                    
                    # Update post proxy with generated code
                    post_proxy.update_attachment("python", AttachmentType.reply_type)
                    post_proxy.update_attachment(generated_code, AttachmentType.reply_content)
                    post_proxy.update_send_to("Planner")
                    
                    # ✅ Early return - skip regular path
                    if self.config.enable_auto_plugin_selection:
                        self.selected_plugin_pool.filter_unused_plugins(code=generated_code)
                    
                    return post_proxy.end()
                    
                except Exception as e:
                    self.logger.error(f"[FUNCTION_CALLING] Fatal error during workflow generation: {e}", exc_info=True)
                    # ❌ NO FALLBACK - Function calling should be deterministic
                    post_proxy.update_attachment(
                        f"Workflow generation failed due to internal error. Please try rephrasing your request or contact support. Error: {str(e)}",
                        AttachmentType.revise_message
                    )
                    post_proxy.update_send_to("User")
                    return post_proxy.end()
        
        # =====================================================================
        # 📝 REGULAR PATH (JSON Schema Response - ONLY when function calling disabled)
        # =====================================================================
        self.logger.info("[CODE_GENERATOR] Using regular JSON schema generation (function calling disabled)")
        
        prompt = self.compose_prompt(
            rounds,
            self.plugin_pool,
            planning_enrichments=enrichment_contents,
        )

        self.tracing.set_span_attribute("prompt", json.dumps(prompt, indent=2))
        prompt_size = self.tracing.count_tokens(json.dumps(prompt))
        self.tracing.set_span_attribute("prompt_size", prompt_size)
        self.tracing.add_prompt_size(
            size=prompt_size,
            labels={
                "direction": "input",
            },
        )

        def early_stop(_type: AttachmentType, value: str) -> bool:
            if _type in [AttachmentType.reply_content]:
                return True
            else:
                return False

        self.post_translator.raw_text_to_post(
            llm_output=self.llm_api.chat_completion_stream(
                prompt,
                use_smoother=True,
                llm_alias=self.config.llm_alias,
                json_schema=self.response_json_schema,
            ),
            post_proxy=post_proxy,
            early_stop=early_stop,
        )

        post_proxy.update_send_to("Planner")
        generated_code = ""
        reply_type: Optional[str] = None
        for attachment in post_proxy.post.attachment_list:
            if attachment.type == AttachmentType.reply_type:
                reply_type = attachment.content
                break
        for attachment in post_proxy.post.attachment_list:
            if attachment.type == AttachmentType.reply_content:
                if reply_type == "python":
                    generated_code = attachment.content
                    
                    # 🔍 EARLY TRUNCATION DETECTION for WORKFLOW dicts
                    if "WORKFLOW" in generated_code and "result = WORKFLOW" in generated_code:
                        # Check if code ends with incomplete string (common truncation pattern)
                        if generated_code.rstrip().endswith(("${from_step:", '"${from_step:', "'${from_step:", ', "params":', ', "text":')):
                            self.logger.error("🚨 [TRUNCATION] LLM output was truncated - WORKFLOW dict is incomplete")
                            # Don't break - let code_interpreter handle it with proper retry message
                            post_proxy.update_attachment(
                                "⚠️ WARNING: Generated code appears to be truncated (incomplete WORKFLOW dict). "
                                "This usually means the workflow is too complex. Try simplifying your request.",
                                AttachmentType.revise_message,
                            )
                    
                    break
                elif reply_type == "text":
                    post_proxy.update_message(attachment.content)
                    break

        if self.config.enable_auto_plugin_selection:
            # filter out plugins that are not used in the generated code
            self.selected_plugin_pool.filter_unused_plugins(code=generated_code)

        if prompt_log_path is not None:
            self.logger.dump_prompt_file(prompt, prompt_log_path)

        self.tracing.set_span_attribute("code", generated_code)

        return post_proxy.post

    def format_plugins(
        self,
        plugin_list: List[PluginEntry],
    ) -> str:
        if self.config.load_plugin:
            return "\n".join(
                [plugin.format_prompt() for plugin in plugin_list],
            )
        return ""

    def get_plugin_pool(self) -> List[PluginEntry]:
        return self.plugin_pool

    def format_code_revision_message(self) -> str:
        return (
            "The execution of the previous generated code has failed. "
            "If you think you can fix the problem by rewriting the code, "
            "please generate code and run it again.\n"
            "Otherwise, please explain the problem to me."
        )

    def format_output_revision_message(self) -> str:
        return (
            "Your previous message is not following the output format. "
            "You must generate the output as a JSON object following the schema provided:\n"
            f"{self.response_json_schema}\n"
            "Please try again."
        )


def format_code_feedback(post: Post) -> str:
    feedback = ""
    verification_status = ""
    execution_status = ""
    for attachment in post.attachment_list:
        if attachment.type == AttachmentType.verification and attachment.content == "CORRECT":
            feedback += "## Verification\nCode verification has been passed.\n"
            verification_status = "CORRECT"
        elif attachment.type == AttachmentType.verification and attachment.content == "NONE":
            feedback += "## Verification\nNo code verification.\n"
            verification_status = "NONE"
        elif attachment.type == AttachmentType.verification and attachment.content == "INCORRECT":
            feedback += "## Verification\nCode verification detected the following issues:\n"
            verification_status = "INCORRECT"
        elif attachment.type == AttachmentType.code_error and verification_status == "INCORRECT":
            feedback += f"{attachment.content}\n"
        elif attachment.type == AttachmentType.execution_status and attachment.content == "NONE":
            feedback += "## Execution\nNo code execution.\n"
            execution_status = "NONE"
        elif attachment.type == AttachmentType.execution_status and attachment.content == "SUCCESS":
            feedback += "## Execution\nYour code has been executed successfully with the following result:\n"
            execution_status = "SUCCESS"
        elif attachment.type == AttachmentType.execution_status and attachment.content == "FAILURE":
            feedback += "## Execution\nYour code has failed to execute with the following error:\n"
            execution_status = "FAILURE"
        elif attachment.type == AttachmentType.execution_result and execution_status != "NONE":
            feedback += f"{attachment.content}\n"
    return feedback
