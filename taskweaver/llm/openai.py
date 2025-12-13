from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Optional

from injector import inject

from taskweaver.llm.util import ChatMessageType, format_chat_message

from .base import CompletionService, EmbeddingService, LLMServiceConfig

DEFAULT_STOP_TOKEN: List[str] = ["<EOS>"]

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIServiceConfig(LLMServiceConfig):
    def _configure(self) -> None:
        # shared common config
        self.api_type = self.llm_module_config.api_type
        assert self.api_type in ["openai", "azure", "azure_ad"], "Invalid API type"

        self._set_name(self.api_type)

        shared_api_base = self.llm_module_config.api_base
        self.api_base = self._get_str(
            "api_base",
            (shared_api_base if shared_api_base is not None else "https://api.openai.com/v1"),
        )
        shared_api_key = self.llm_module_config.api_key
        self.api_key = self._get_str(
            "api_key",
            (shared_api_key if shared_api_key is not None else ("" if self.api_type == "azure_ad" else None)),
        )

        shared_model = self.llm_module_config.model
        self.model = self._get_str(
            "model",
            shared_model if shared_model is not None else "gpt-4",
        )

        shared_embedding_model = self.llm_module_config.embedding_model
        self.embedding_model = self._get_str(
            "embedding_model",
            (shared_embedding_model if shared_embedding_model is not None else "text-embedding-ada-002"),
        )

        self.response_format = self.llm_module_config.response_format

        # openai specific config
        # ✅ NO FALLBACK - Must be provided by eclipse_adapter from .env
        self.api_version = self._get_str("api_version", None, required=False)
        is_azure_ad_login = self.api_type == "azure_ad"
        self.aad_auth_mode = self._get_enum(
            "aad_auth_mode",
            ["device_login", "aad_app", "default_azure_credential"],
            None if is_azure_ad_login else "device_login",
        )

        is_app_login = is_azure_ad_login and self.aad_auth_mode == "aad_app"
        self.aad_tenant_id = self._get_str(
            "aad_tenant_id",
            None if is_app_login else "common",
        )
        self.aad_api_resource = self._get_str(
            "aad_api_resource",
            None if is_app_login else "https://cognitiveservices.azure.com/",
        )
        self.aad_api_scope = self._get_str(
            "aad_api_scope",
            None if is_app_login else ".default",
        )
        self.aad_client_id = self._get_str(
            "aad_client_id",
            None if is_app_login else "",
        )
        self.aad_client_secret = self._get_str(
            "aad_client_secret",
            None if is_app_login else "",
        )
        self.aad_skip_interactive = self._get_bool(
            "aad_skip_interactive",
            # support interactive on macOS and Windows by default, skip on other platforms
            # could be overridden by config
            not (sys.platform == "darwin" or sys.platform == "win32"),
        )
        self.aad_use_token_cache = self._get_bool("aad_use_token_cache", True)
        self.aad_token_cache_path = self._get_str(
            "aad_token_cache_path",
            "cache/token_cache.bin",
        )
        self.aad_token_cache_full_path = os.path.join(
            self.src.app_base_path,
            self.aad_token_cache_path,
        )

        self.stop_token = self._get_list("stop_token", DEFAULT_STOP_TOKEN)
        self.temperature = self._get_float("temperature", 0)
        self.max_tokens = self._get_int("max_tokens", 1024)
        self.top_p = self._get_float("top_p", 0)
        self.frequency_penalty = self._get_float("frequency_penalty", 0)
        self.presence_penalty = self._get_float("presence_penalty", 0)
        self.seed = self._get_int("seed", 123456)

        self.require_alternative_roles = self._get_bool("require_alternative_roles", False)
        self.support_system_role = self._get_bool("support_system_role", True)
        self.support_constrained_generation = self._get_bool("support_constrained_generation", False)
        self.json_schema_enforcer = self._get_str("json_schema_enforcer", None, required=False)


class OpenAIService(CompletionService, EmbeddingService):
    @inject
    def __init__(self, config: OpenAIServiceConfig):
        self.config = config

        self.api_type = self.config.api_type

        assert self.api_type in ["openai", "azure", "azure_ad"], "Invalid API type"

        self._client: Optional[OpenAI] = None

    @property
    def client(self):
        from openai import AzureOpenAI, OpenAI

        if self._client is not None:
            return self._client

        if self.api_type == "openai":
            client = OpenAI(
                base_url=self.config.api_base,
                api_key=self.config.api_key,
            )
        elif self.api_type == "azure":
            client = AzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.api_base,
                api_key=self.config.api_key,
            )
        elif self.api_type == "azure_ad":
            client = AzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.api_base,
                azure_ad_token_provider=self._get_aad_token_provider(),
            )
        else:
            raise Exception(f"Invalid API type: {self.api_type}")

        self._client = client
        return client

    def chat_completion(
        self,
        messages: List[ChatMessageType],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageType, None, None]:
        import openai

        engine = self.config.model

        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        top_p = top_p if top_p is not None else self.config.top_p
        stop = stop if stop is not None else self.config.stop_token
        seed = self.config.seed

        try:
            tools_kwargs = {}
            if "tools" in kwargs and "tool_choice" in kwargs:
                tools_kwargs["tools"] = kwargs["tools"]
                tools_kwargs["tool_choice"] = kwargs["tool_choice"]

            if "response_format" in kwargs:
                response_format = kwargs["response_format"]
            elif self.config.response_format == "json_object":
                response_format = {"type": "json_object"}
            elif self.config.response_format == "json_schema":
                response_format = {"type": "json_schema"}
                assert "json_schema" in kwargs, "JSON schema is required for JSON schema response format"
                
                # Fix: Extract inner schema if wrapped (TaskWeaver wraps schema with {"properties":{"response":{...}}})
                schema = kwargs["json_schema"]
                if "properties" in schema and "response" in schema.get("properties", {}):
                    # Use inner "response" schema directly for Azure API
                    inner_schema = schema["properties"]["response"]
                    response_format["json_schema"] = {
                        "name": "response",
                        "strict": False,  # Azure 2025-01-01-preview compatibility
                        "schema": inner_schema,
                    }
                else:
                    response_format["json_schema"] = {
                        "name": "response",
                        "strict": False,
                        "schema": schema,
                    }
            else:
                response_format = None

            extra_body = {}
            if self.config.support_constrained_generation:
                if "json_schema" in kwargs:
                    extra_body["guided_json"] = kwargs["json_schema"]
                    assert isinstance(extra_body["guided_json"], dict), "JSON schema must be a dictionary"

                    assert self.config.json_schema_enforcer in [
                        "outlines",
                        "lm-format-enforcer",
                    ], f"Invalid JSON schema enforcer: {self.config.json_schema_enforcer}"
                    extra_body["guided_decoding_backend"] = self.config.json_schema_enforcer

                else:
                    raise Exception("Constrained generation requires a JSON schema")

            # Preprocess messages
            # 1. Change `system` to `user` if `support_system_role` is False
            # 2. Add dummy `assistant` messages if alternating user/assistant is required
            for i, message in enumerate(messages):
                if (not self.config.support_system_role) and message["role"] == "system":
                    message["role"] = "user"
                if self.config.require_alternative_roles:
                    if i > 0 and message["role"] == "user" and messages[i - 1]["role"] == "user":
                        messages.insert(
                            i,
                            {"role": "assistant", "content": "I get it."},
                        )

            res: Any = self.client.chat.completions.create(
                model=engine,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stop=stop,
                stream=stream,
                seed=seed,
                response_format=response_format,
                extra_body=extra_body,
                **tools_kwargs,
            )
            if stream:
                role: Any = None
                # ========================================================================
                # STREAMING WITH json_schema: Collect full response for SDK parsing
                # ========================================================================
                # Note: Streaming deltas don't have 'parsed' field, only final response does.
                # For json_schema mode, we need to wait for SDK to parse the full response.
                # For now, we stream raw content and let translator.py handle unwrapping.
                # TODO: Consider buffering for json_schema mode to leverage SDK's parsed field.
                # ========================================================================
                for stream_res in res:
                    if not stream_res.choices:
                        continue
                    delta = stream_res.choices[0].delta
                    if delta is None:
                        continue

                    role = delta.role if delta.role is not None else role
                    content = delta.content if delta.content is not None else ""
                    if content is None:
                        continue
                    yield format_chat_message(role, content)
            else:
                oai_response = res.choices[0].message
                if oai_response is None:
                    raise Exception("OpenAI API returned an empty response")
                
                # ========================================================================
                # ARCHITECTURAL FIX: Use OpenAI SDK's 'parsed' field (LangChain pattern)
                # ========================================================================
                # When using response_format json_schema, the OpenAI SDK automatically
                # unwraps Azure's {"type":"object","properties":{...}} format and stores
                # the unwrapped data in oai_response.parsed.
                #
                # This is how LangChain/LangGraph/AutoGen avoid manual unwrapping!
                # ========================================================================
                content_to_use = None
                if hasattr(oai_response, 'parsed') and oai_response.parsed is not None:
                    # ✅ SDK already unwrapped - use parsed field
                    import json
                    content_to_use = json.dumps({"response": oai_response.parsed})
                    print(f"[OPENAI_SERVICE] ✅ Using SDK's parsed field (unwrapped by SDK)")
                else:
                    # Fallback to raw content (for non-json_schema modes)
                    content_to_use = oai_response.content if oai_response.content is not None else ""
                    print(f"[OPENAI_SERVICE] ⚠️ Using raw content (no parsed field)")
                
                response: ChatMessageType = format_chat_message(
                    role=(oai_response.role if oai_response.role is not None else "assistant"),
                    message=content_to_use,
                )
                if oai_response.tool_calls is not None and len(oai_response.tool_calls) > 0:
                    import json

                    response["role"] = "function"
                    response["content"] = json.dumps(
                        [
                            {
                                "name": t.function.name,
                                "arguments": json.loads(t.function.arguments),
                            }
                            for t in oai_response.tool_calls
                        ],
                    )
                yield response

        except openai.APITimeoutError as e:
            # Handle timeout error, e.g. retry or log
            raise Exception(f"OpenAI API request timed out: {e}")
        except openai.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            raise Exception(f"OpenAI API request failed to connect: {e}")
        except openai.BadRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            raise Exception(f"OpenAI API request was invalid: {e}")
        except openai.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            raise Exception(f"OpenAI API request was not authorized: {e}")
        except openai.PermissionDeniedError as e:
            # Handle permission error, e.g. check scope or log
            raise Exception(f"OpenAI API request was not permitted: {e}")
        except openai.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            raise Exception(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            # Handle API error, e.g. retry or log
            raise Exception(f"OpenAI API returned an API Error: {e}")

    def chat_completion_with_function_calling(
        self,
        messages: List[ChatMessageType],
        functions: List[dict],
        function_call: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Call OpenAI with function calling (non-streaming).
        
        🔑 DESIGN: Deterministic function call for workflow IR generation.
        - Non-streaming only (function calling requires full response)
        - temperature=0.0 by default (deterministic output)
        - Validates exactly one tool call (no ambiguity)
        - Validates JSON before returning
        
        Args:
            messages: Chat messages
            functions: List of function schemas
            function_call: {"name": "function_name"} to force a specific function
            temperature: 0.0 for deterministic output (default)
            max_tokens: Max tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            {
                "function_name": str,
                "arguments": dict  # Already parsed and validated JSON
            }
            
        Raises:
            Exception: If no function call, multiple calls, or invalid JSON
        """
        import json
        import openai

        engine = self.config.model
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        try:
            # Preprocess messages (same as regular call)
            for i, message in enumerate(messages):
                if (not self.config.support_system_role) and message["role"] == "system":
                    message["role"] = "user"
                if self.config.require_alternative_roles:
                    if i > 0 and message["role"] == "user" and messages[i - 1]["role"] == "user":
                        messages.insert(
                            i,
                            {"role": "assistant", "content": "I get it."},
                        )

            # Call OpenAI with tools (new API format)
            response = self.client.chat.completions.create(
                model=engine,
                messages=messages,  # type: ignore
                tools=[{"type": "function", "function": f} for f in functions],
                tool_choice=(
                    {"type": "function", "function": function_call}
                    if function_call
                    else "auto"
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stream=False,  # Function calling requires non-streaming
            )

            # ✅ FIX 1: Validate exactly one tool call
            message_response = response.choices[0].message
            tool_calls = message_response.tool_calls or []

            if len(tool_calls) != 1:
                raise Exception(
                    f"Expected exactly one function call, got {len(tool_calls)}. "
                    f"This indicates either OpenAI API change or schema ambiguity."
                )

            tool_call = tool_calls[0]

            # ✅ FIX 2: Validate JSON before parsing
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Invalid JSON from function call '{tool_call.function.name}': {e}\n"
                    f"Raw arguments: {tool_call.function.arguments[:200]}"
                )

            return {
                "function_name": tool_call.function.name,
                "arguments": arguments,
            }

        except openai.APITimeoutError as e:
            raise Exception(f"OpenAI API request timed out: {e}")
        except openai.APIConnectionError as e:
            raise Exception(f"OpenAI API request failed to connect: {e}")
        except openai.BadRequestError as e:
            raise Exception(f"OpenAI API request was invalid: {e}")
        except openai.AuthenticationError as e:
            raise Exception(f"OpenAI API request was not authorized: {e}")
        except openai.PermissionDeniedError as e:
            raise Exception(f"OpenAI API request was not permitted: {e}")
        except openai.RateLimitError as e:
            raise Exception(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            raise Exception(f"OpenAI API returned an API Error: {e}")

    def get_embeddings(self, strings: List[str]) -> List[List[float]]:
        embedding_results = self.client.embeddings.create(
            input=strings,
            model=self.config.embedding_model,
        ).data
        return [r.embedding for r in embedding_results]

    def _get_aad_token_provider(self) -> Callable[[], str]:
        if self.config.aad_auth_mode == "default_azure_credential":
            return self._get_aad_token_provider_azure_identity()
        return lambda: self._get_aad_token_msal()

    def _get_aad_token_provider_azure_identity(self) -> Callable[[], str]:
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # type: ignore
        except ImportError:
            raise Exception(
                "AAD authentication requires azure-identity module to be installed, "
                "please run `pip install azure-identity`",
            )
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        print("Using DefaultAzureCredential for AAD authentication")
        scope = f"{self.config.aad_api_resource}/{self.config.aad_api_scope}"
        return get_bearer_token_provider(credential, scope)

    def _get_aad_token_msal(self) -> str:
        try:
            import msal  # type: ignore
        except ImportError:
            raise Exception(
                "AAD authentication requires msal module to be installed, please run `pip install msal`",
            )

        config = self.config

        cache: Any = msal.SerializableTokenCache()

        token_cache_file: Optional[str] = None
        if config.aad_use_token_cache:
            token_cache_file = config.aad_token_cache_full_path
            if not os.path.exists(token_cache_file):
                os.makedirs(os.path.dirname(token_cache_file), exist_ok=True)
            if os.path.exists(token_cache_file):
                with open(token_cache_file, "r") as cache_file:
                    cache.deserialize(cache_file.read())  # type: ignore

        def save_cache():
            if token_cache_file is not None and config.aad_use_token_cache:
                with open(token_cache_file, "w") as cache_file:
                    cache_file.write(cache.serialize())

        authority = f"https://login.microsoftonline.com/{config.aad_tenant_id}"
        api_resource = config.aad_api_resource
        api_scope = config.aad_api_scope
        auth_mode = config.aad_auth_mode

        if auth_mode == "aad_app":
            app: Any = msal.ConfidentialClientApplication(
                client_id=config.aad_client_id,
                client_credential=config.aad_client_secret,
                authority=authority,
                token_cache=cache,
            )
            result: Any = app.acquire_token_for_client(
                scopes=[
                    f"{api_resource}/{api_scope}",
                ],
            )
            if "access_token" in result:
                return result["access_token"]

            raise Exception(
                f"Authentication failed for acquiring AAD token for application login: {str(result)}",
            )

        scopes = [
            f"{api_resource}/{api_scope}",
        ]
        app: Any = msal.PublicClientApplication(
            "04b07795-8ddb-461a-bbee-02f9e1bf7b46",  # default id in Azure Identity module
            authority=authority,
            token_cache=cache,
        )
        result = None
        try:
            account = app.get_accounts()[0]
            result = app.acquire_token_silent(scopes, account=account)
            if result is not None and "access_token" in result:
                save_cache()
                return result["access_token"]
            result = None
        except Exception:
            pass

        try:
            account = cache.find(cache.CredentialType.ACCOUNT)[0]
            refresh_token = cache.find(
                cache.CredentialType.REFRESH_TOKEN,
                query={
                    "home_account_id": account["home_account_id"],
                },
            )[0]
            result = app.acquire_token_by_refresh_token(
                refresh_token["secret"],
                scopes=scopes,
            )
            if result is not None and "access_token" in result:
                save_cache()
                return result["access_token"]
            result = None
        except Exception:
            pass

        if not self.config.aad_skip_interactive:
            try:
                result = app.acquire_token_interactive(scopes=scopes)
                if result is not None and "access_token" in result:
                    save_cache()
                    return result["access_token"]
                result = None
            except Exception:
                pass

        flow = app.initiate_device_flow(scopes=scopes)
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow=flow)
        if result is not None and "access_token" in result:
            save_cache()
            return result["access_token"]

        error_details = "\n".join(
            [
                result.get("error"),
                result.get("error_description"),
            ],
        )
        raise Exception(
            f"Authentication failed for acquiring AAD token for AAD auth: {error_details}",
        )
