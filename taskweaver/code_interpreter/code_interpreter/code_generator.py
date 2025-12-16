import datetime
import json
import os
from typing import List, Optional, Dict, Any

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

# ✅ Consolidated placeholder validator (Pydantic-based)
from apps.analytics_conversation.services.workflow_placeholder_validator import validate_workflow_placeholders


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
        
        # Load Composio tool cache for placeholder validation
        self.composio_cache = self._load_composio_cache()

        self.logger.info("CodeGenerator initialized successfully")
    
    def _load_composio_cache(self) -> Dict[str, Any]:
        """
        Load Composio tool cache from composio_schemas_cache.json.
        Used by placeholder validator for deep schema validation.
        
        Returns:
            Dict mapping tool_id -> tool schema, or {} if not available
        """
        try:
            cache_file = 'composio_schemas_cache.json'
            # Navigate from code_interpreter/code_interpreter/ to project/plugins/
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_path = os.path.join(base_dir, 'project', 'plugins', cache_file)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    self.logger.info(f"✅ [TOOL_CACHE] Loaded {len(cache)} tools for validation")
                    return cache
            else:
                self.logger.warning(f"⚠️ [TOOL_CACHE] Cache not found at {cache_path} (optional for validation)")
                return {}
        except Exception as e:
            self.logger.warning(f"⚠️ [TOOL_CACHE] Failed to load cache: {e} (optional for validation)")
            return {}

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

    def get_env_context(self):
        # get date and time
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        return f"- Current time: {current_time}"

    def _extract_tool_ids_from_actions(self, composio_actions: str, init_plan: str = "") -> List[str]:
        """
        Extract tool IDs from select_composio_actions() formatted output.
        
        Input format:
        "Available Composio Actions:
         - APP1_ACTION_NAME (app: app1): Description [step_index: 1]
         - APP2_ACTION_NAME (app: app2): Description [step_index: 2]"
        
        Output: ["APP1_ACTION_NAME", "APP2_ACTION_NAME"]
        
        🔑 CRITICAL: This creates the filtered enum (5-50 tools, not 17k!)
        
        ✅ INDEX-BASED MAPPING (NEW):
        - When [step_index: N] is present, tools are already mapped by Composio
        - Extract tools in step_index order (explicit 1:1 correspondence)
        - No fuzzy matching needed - direct index-based mapping!
        
        ✅ FALLBACK (Legacy):
        - If no step_index, return all tools (let LLM decide)
        """
        import re
        
        # ✅ NEW: Check if this is index-based response (targeted mode)
        has_step_indices = '[step_index:' in composio_actions
        
        if has_step_indices:
            # ✅ INDEX-BASED MODE: Parse tool_id and step_index together
            self.logger.info(f"[FUNCTION_CALLING] 🎯 TARGETED MODE: Parsing index-based tool mapping...")
            
            # Match format: "- ACTION_ID (app: ...) ... [step_index: N]"
            pattern = r'^\s*-\s*([A-Z][A-Z0-9_]+)\s*\(app:.*?\).*?\[step_index:\s*(\d+)\]'
            matches = re.findall(pattern, composio_actions, re.MULTILINE)
            
            if matches:
                # Sort by step_index to preserve order
                indexed_tools = [(tool_id, int(index)) for tool_id, index in matches]
                indexed_tools.sort(key=lambda x: x[1])  # Sort by index
                
                tool_ids = [tool_id for tool_id, _ in indexed_tools]
                
                self.logger.info(f"[FUNCTION_CALLING] ✅ Extracted {len(tool_ids)} indexed tools:")
                for tool_id, index in indexed_tools:
                    self.logger.info(f"   [INDEX {index}] → {tool_id}")
                
                return tool_ids
            else:
                self.logger.warning(
                    "[FUNCTION_CALLING] ⚠️ step_index found but failed to parse - "
                    "falling back to simple extraction"
                )
        
        # ✅ LEGACY/FALLBACK MODE: Simple extraction (no filtering)
        self.logger.info(f"[FUNCTION_CALLING] 📝 LEGACY MODE: Extracting all tools (let LLM decide)")
        
        # Match format "- ACTION_ID (app: ...)" or "- ACTION_ID:"
        all_tool_ids = re.findall(r'^\s*-\s*([A-Z][A-Z0-9_]+)\s*(?:\(app:|:)', composio_actions, re.MULTILINE)
        
        self.logger.info(f"[FUNCTION_CALLING] Extracted {len(all_tool_ids)} tools: {all_tool_ids}")
        
        return all_tool_ids

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
        DEPRECATED - This method is no longer used.
        
        Function calling relies on JSON schema, not conversation examples.
        Examples are now embedded in the system prompt and planner_prompt.yaml.
        
        Keeping this for backwards compatibility but it returns empty string.
        """
        return ""

    def _auto_correct_workflow(
        self,
        workflow_json: dict,
        user_prompt: str,
        init_plan: str
    ) -> dict:
        """
        Phase 3: Auto-correction delegated to WorkflowAutoCorrector.
        
        Cross-reference: workflow_auto_corrector.py for implementation details
        """
        from taskweaver.code_interpreter.code_interpreter.workflow_auto_corrector import get_auto_corrector
        
        corrector = get_auto_corrector(logger=self.logger)
        return corrector.auto_correct(workflow_json, user_prompt, init_plan)

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

    # ============================================================================
    # LEGACY METHODS DELETED (unused after function calling migration)
    # - compose_prompt()
    # - compose_conversation()
    # - format_attachment()
    # These were only used by the non-function-calling path which has been removed.
    # ============================================================================

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
        
        # ✅ DEBUG: Log what queries we're working with
        self.logger.info(f"[QUERY_DEBUG] Planner message (query): {query[:150]}")
        self.logger.info(f"[QUERY_DEBUG] Original user query: {original_user_query[:150]}")
        self.logger.info(f"[QUERY_DEBUG] Has user_query attr: {hasattr(rounds[-1], 'user_query')}")
        if hasattr(rounds[-1], 'user_query'):
            self.logger.info(f"[QUERY_DEBUG] Round user_query value: {rounds[-1].user_query[:150] if rounds[-1].user_query else 'None'}")

        self.tracing.set_span_attribute("query", query)
        self.tracing.set_span_attribute("enable_auto_plugin_selection", self.config.enable_auto_plugin_selection)
        self.tracing.set_span_attribute("use_experience", self.config.use_experience)

        if self.config.enable_auto_plugin_selection:
            self.plugin_pool = self.select_plugins_for_prompt(query)

        self.role_load_experience(query=query, memory=memory)
        
        # ✅ LEGACY: Example loading removed (not used by function calling path)
        # Function calling relies on JSON schema, not conversation examples
        
        planning_enrichments = memory.get_shared_memory_entries(entry_type="plan")
        
        # ✅ Extract init_plan from Planner's attachments (has <interactive dependency> markers)
        init_plan_with_markers = ""
        for conversation_round in rounds:
            for post in conversation_round.post_list:
                if post.send_from == "Planner":
                    for attachment in post.attachment_list:
                        if attachment.type == AttachmentType.init_plan:
                            init_plan_with_markers = attachment.content
                            self.logger.info(f"[INIT_PLAN] Found init_plan with dependency markers: {init_plan_with_markers[:200]}")
                            break
        
        if not init_plan_with_markers:
            self.logger.warning(f"[INIT_PLAN] ⚠️ No init_plan attachment found in rounds")

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
                
                # Get session ID from environment (set by eclipse_adapter.py)
                import os
                session_id = os.environ.get('TASKWEAVER_SESSION_ID', None)
                
                # ✅ On retry (when query is "Failed to generate code..."), use original_user_query
                # This ensures cache lookup matches the cached key from first attempt
                query_for_composio = original_user_query if "failed" in query.lower() or "error" in query.lower() else query
                
                # ✅ NEW: Extract agent_with_tools steps from init_plan for targeted tool discovery
                structured_steps = None
                if init_plan_with_markers:
                    import re
                    structured_steps = []
                    for line in init_plan_with_markers.split('\n'):
                        # Match both top-level (1.) and nested (2.1.) step numbers
                        match = re.match(r'^\s*(\d+(?:\.\d+)?)\.\s+(.+?)\s*<([\w_]+)>', line)
                        if match:
                            step_num, description, step_type = match.groups()
                            if step_type == 'agent_with_tools':
                                structured_steps.append({
                                    'step_num': step_num,
                                    'description': description.strip(),
                                    'index': len(structured_steps) + 1  # 1-indexed for Composio
                                })
                    
                    if structured_steps:
                        self.logger.info(
                            f"[COMPOSIO_CALL] 🎯 Extracted {len(structured_steps)} agent_with_tools steps "
                            f"for targeted tool discovery: {[s['step_num'] for s in structured_steps]}"
                        )
                    else:
                        self.logger.warning("[COMPOSIO_CALL] ⚠️ No agent_with_tools steps found in init_plan")
                        structured_steps = None  # Fall back to full prompt
                
                # ✅ DEBUG: Log what's being sent to Composio
                self.logger.info(f"[COMPOSIO_CALL] user_query param: {query_for_composio[:150]}")
                self.logger.info(f"[COMPOSIO_CALL] context param (sent to batch API): {original_user_query[:150]}")
                self.logger.info(f"[COMPOSIO_CALL] session_id: {session_id}")
                
                composio_actions = select_composio_actions(
                    user_query=query_for_composio,  # Use original query on retry for cache hit
                    context=original_user_query,  # Full query for domain/app discovery
                    top_k=10,  # Balanced - enough for both read and write actions
                    adaptive_top_k=True,  # Enable automatic scaling based on detected apps
                    session_id=session_id,  # ✅ Enable batch API caching per session
                    structured_steps=structured_steps  # ✅ NEW: Per-step tool discovery
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
            
            # Use init_plan extracted from Planner's attachments
            tool_ids = self._extract_tool_ids_from_actions(composio_actions, init_plan_with_markers)
            
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
                    # Schema is now flat (no 'workflow' wrapper)
                    nodes_prop = params.get('properties', {}).get('nodes', {})
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
                # ✅ CRITICAL: Use init_plan_with_markers (detailed), not Planner's condensed plan!
                # The Planner's "plan" is a 3-step summary, but init_plan has all 6+ detailed steps
                plan = init_plan_with_markers if init_plan_with_markers else next(
                    (e for e in enrichment_contents if "Plan" in e or "plan" in e.lower()),
                    "No plan provided"
                )
                
                # ✅ DEBUG: Log what plan is being used
                plan_source = "init_plan_with_markers" if init_plan_with_markers else "enrichment_contents"
                self.logger.info(f"[PROMPT_BUILD] Using plan from: {plan_source}, length: {len(plan)} chars")
                self.logger.info(f"[PROMPT_BUILD] Plan preview (first 300 chars): {plan[:300]}")
                
                # ✅ SKIP EXAMPLES for function calling - schema is already very detailed
                # Loading 4 examples adds ~10K tokens and causes function calling issues
                # The JSON schema provides all the information the LLM needs
                example_context = ""
                yaml_examples = ""
                
                # ✅ Build step guidance: Map init_plan steps to node types AND extract control dependencies
                step_guidance = ""
                self.logger.info(f"[STEP_GUIDANCE] Building step guidance, init_plan length: {len(init_plan_with_markers)}")
                if init_plan_with_markers:
                    import re
                    step_hints = []
                    
                for line in init_plan_with_markers.split('\n'):
                    # 🎯 UNIVERSAL REGEX: Capture ENTIRE line after step number
                    # Handles ANY order: "1. Desc <marker> from $X" OR "1. Desc from $X <marker>"
                    match = re.match(r'^\s*(\d+(?:\.\d+)?)\.\s+(.+)', line)
                    if match:
                        step_num = match.group(1)
                        full_line = match.group(2).strip()  # Full line after step number
                        
                        # Extract marker from anywhere in the line
                        marker_match = re.search(r'<([^>]+)>', full_line)
                        node_type_marker = marker_match.group(1).strip() if marker_match else ""
                        
                        # Extract description (everything before first marker, or full line if no marker)
                        step_desc = full_line.split('<')[0].strip() if '<' in full_line else full_line
                        
                        # 🎯 UNIVERSAL DEPENDENCY EXTRACTION (works for ANY workflow pattern, ANY order)
                        # Search in FULL LINE (not just description) to handle both:
                        #   - "1. Desc from $X <marker>"  (old order)
                        #   - "1. Desc <marker> from $X"  (new order)
                        
                        # Pattern: Find ALL $X references after "after" keyword
                        # Handles: "after $1", "after $2 and $3", "after $1, $2, and $3"
                        after_deps = []
                        if 'after' in full_line:
                            after_clause = full_line.split('after', 1)[1]
                            after_matches = re.findall(r'\$(\d+)', after_clause)
                            if after_matches:
                                after_deps = [int(d) for d in after_matches]
                                self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'after' control deps: {after_deps}")
                        
                        # Pattern: Find ALL $X references after "from" keyword
                        # Handles: "from $1", "from $2 and $3", "from $1, $2, and $3"
                        from_deps = []
                        if 'from' in full_line:
                            from_clause = full_line.split('from', 1)[1]
                            if 'after' in from_clause:
                                from_clause = from_clause.split('after', 1)[0]  # Stop at "after"
                            from_matches = re.findall(r'\$(\d+)', from_clause)
                            if from_matches:
                                from_deps = [int(d) for d in from_matches]
                                self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'from' data deps: {from_deps}")
                        
                        # Pattern: Find ALL $X references after "in"/"over" keyword (for loops)
                        # Uses precise regex to avoid false matches like "included", "within"
                        in_deps = []
                        if not from_deps:  # Only check 'in'/'over' if 'from' wasn't found
                            in_pattern = r'\b(?:in|over)\s+\$(\d+)'
                            in_matches = re.findall(in_pattern, full_line)
                            if in_matches:
                                in_deps = [int(d) for d in in_matches]
                                self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'in/over' data deps (loop): {in_deps}")
                            # Also check for comma-separated items: "in $1, $2, and $3"
                            elif re.search(r'\b(?:in|over)\s+\$\d+', full_line):
                                match_in = re.search(r'\b(?:in|over)\s+(.+?)(?:\s+(?:after|<)|$)', full_line)
                                if match_in:
                                    in_clause = match_in.group(1)
                                    in_matches = re.findall(r'\$(\d+)', in_clause)
                                    if in_matches:
                                        in_deps = [int(d) for d in in_matches]
                                        self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'in/over' data deps (loop, multi): {in_deps}")
                        
                        # 🎯 UNIVERSAL DEPENDENCY LOGIC:
                        # Combine ALL dependencies from 'from', 'after', and 'in/over' clauses
                        # If step says "from $3 after $2" → dependencies = [2, 3] (BOTH!)
                        all_deps = set()
                        all_deps.update(from_deps)
                        all_deps.update(after_deps)
                        all_deps.update(in_deps)
                        control_deps = sorted(list(all_deps)) if all_deps else []
                        
                        deps_hint = ""
                        if control_deps:
                            deps_hint = f", dependencies={control_deps}"
                        else:
                            deps_hint = f", dependencies=[] (independent)"
                        
                        self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: {step_desc}, type: {node_type_marker}{deps_hint}")
                        
                        # Map specific node type markers to hints
                        if node_type_marker == "form":
                            hint = f"Step {step_num}: type='form' with fields{deps_hint}"
                            step_hints.append(hint)
                            self.logger.info(f"[STEP_GUIDANCE] Added hint: {hint}")
                        elif node_type_marker == "hitl":
                            hint = f"Step {step_num}: type='hitl' for approval/review{deps_hint}"
                            step_hints.append(hint)
                            self.logger.info(f"[STEP_GUIDANCE] Added hint: {hint}")
                        elif node_type_marker == "agent_with_tools":
                            hint = f"Step {step_num}: type='agent_with_tools' WITH tool_id{deps_hint}"
                            step_hints.append(hint)
                            self.logger.info(f"[STEP_GUIDANCE] Added hint: {hint}")
                        elif node_type_marker == "agent_only":
                            hint = f"Step {step_num}: type='agent_only' for analysis{deps_hint}"
                            step_hints.append(hint)
                            self.logger.info(f"[STEP_GUIDANCE] Added hint: {hint}")
                        elif node_type_marker == "loop":
                            hint = f"Step {step_num}: type='loop' with items{deps_hint}"
                            step_hints.append(hint)
                            self.logger.info(f"[STEP_GUIDANCE] Added hint: {hint}")
                    
                    if step_hints:
                        num_steps = len(step_hints)
                        step_guidance = f"\n\n**GENERATE ALL {num_steps} NODES:**\n" + "\n".join(f"- {h}" for h in step_hints)
                        step_guidance += f"\n\n⚠️ Generate a node for EVERY step above (ALL {num_steps} nodes required)!"
                        step_guidance += f"\n\n🚨 CRITICAL: Generate EXACTLY {num_steps} nodes - NO MORE, NO LESS!"
                        step_guidance += f"\n   Do NOT expand or add extra nodes beyond the {num_steps} steps listed above!"
                        step_guidance += f"\n   Each step in the plan = EXACTLY 1 node in the output!"
                        self.logger.info(f"[STEP_GUIDANCE] Generated {len(step_hints)} step hints")
                    else:
                        self.logger.warning(f"[STEP_GUIDANCE] ⚠️ No step hints generated!")
                
                minimal_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "You are a workflow compiler. Generate workflow NODES ONLY via function calls.\n\n"
                            "**CRITICAL RULES:**\n"
                            "1. Generate a node for EVERY step in the plan\n"
                            "2. Use ONLY tool_id values from the Available Composio Actions list\n"
                            "3. All placeholders use ${{...}} with VALID patterns:\n"
                            "   • ${{EXTRACT:param_name}} - Extract from user query\n"
                            "   • ${{from_step:node_id.field}} - Reference previous node\n"
                            "   • ${{from_loop:loop_id.results}} - Access loop results\n"
                            "   • ${{loop_item}} - Current iteration item\n"
                            "   • Static values - Use directly without brackets\n"
                            "   NEVER: ${{bare_name}}, ${{user.*}}, ${{context.*}}\n"
                            "4. Node types: agent_with_tools (REQUIRES tool_id), agent_only, form, hitl, loop\n"
                            "5. MANDATORY: Every node MUST have a 'description' field with clear, user-friendly text\n"
                            "   • Take the step text from the plan and make it user-friendly\n"
                            "   • Remove markers: <agent_with_tools>, <loop>, etc.\n"
                            "   • Remove technical references: Replace 'from $1' with the actual action\n"
                            "   • Example: Plan '1. Fetch 5 Jira tickets' → description='Fetch 5 Jira tickets'\n"
                            "   • Example: Plan '2. For each ticket in $1' → description='For each ticket, add comment'\n"
                            "   • Example: Plan '2.1. Add comment' → description='Add comment to ticket'\n\n"
                            "**DUAL DEPENDENCY SYSTEM (UNIVERSAL - Works for ANY Workflow):**\n"
                            "The plan uses TWO types of dependencies:\n"
                            "   • 'from $X' = DATA dependency (which data this step consumes)\n"
                            "   • 'after $Y' = CONTROL dependency (which steps must complete first)\n\n"
                            "**DEPENDENCIES FIELD EXTRACTION (PATTERN-AGNOSTIC):**\n"
                            "⚠️ EVERY node MUST have a 'dependencies' field with CONTROL dependencies (1-based step numbers):\n\n"
                            "RULE 1: If step says 'after $X' → dependencies=[X] (explicit control)\n"
                            "RULE 2: If step says 'after $X and $Y' → dependencies=[X, Y] (multiple control)\n"
                            "RULE 3: If step says 'from $X after $Y' → dependencies=[Y] (control ≠ data)\n"
                            "RULE 4: If step says 'from $X' (no 'after') → dependencies=[X] (shorthand: control = data)\n"
                            "RULE 5: If step has NO 'from' or 'after' → dependencies=[] (independent/parallel)\n\n"
                            "**UNIVERSAL EXAMPLES (Works for ANY scenario):**\n"
                            "   Plan: '1. Fetch data' → dependencies=[] (independent)\n"
                            "   Plan: '2. Process from $1' → dependencies=[1] (shorthand)\n"
                            "   Plan: '3. Analyze from $1 and $2' → dependencies=[1, 2] (multiple data = multiple control)\n"
                            "   Plan: '4. Send from $1 after $3' → dependencies=[3] (control ≠ data!)\n"
                            "   Plan: '5. Save after $2 and $4' → dependencies=[2, 4] (explicit control, no data ref)\n"
                            "   Plan: '6. Loop from $5' → dependencies=[5] (shorthand)\n\n"
                            "**PARALLEL EXECUTION (Automatic via dependencies):**\n"
                            "   • Same dependencies = parallel execution\n"
                            "   • Example: Steps 2&3 both have dependencies=[1] → Run in parallel!\n"
                            "   • Example: Steps 5,6,7 all have dependencies=[4] → All run in parallel!\n\n"
                            "**DATA FLOW (LangGraph-Native):**\n"
                            "✅ Use actual state field names with optional path traversal:\n\n"
                            "**SIMPLE (auto-extraction):**\n"
                            "   • loop_over='step_1_output'  # Auto-extracts array from response\n"
                            "   • loop_over='node_2'  # Auto-finds 'messages', 'data', 'items', etc.\n\n"
                            "**EXPLICIT (nested path):**\n"
                            "   • loop_over='node_1.messages'  # Gmail API: {messages: [...]}\n"
                            "   • loop_over='step_2_output.data'  # REST API: {data: [...]}\n"
                            "   • loop_over='node_3.items'  # Generic: {items: [...]}\n\n"
                            "**LOOP BODY ACCESS:**\n"
                            "   • '${{{{loop_item}}}}' - current item in iteration\n\n"
                            "**KEY INSIGHT:**\n"
                            "Control dependencies (dependencies field) define WHEN to execute.\n"
                            "Data references (step_N_output) define WHAT data to use.\n"
                            "These can be DIFFERENT! Example: 'Send email from $1 after $5' means:\n"
                            "   dependencies=[5] (wait for step 5 approval)\n"
                            "   params uses step_1_output (use data from step 1)\n\n"
                            "**EDGES:**\n"
                            "⚠️ DO NOT generate 'edges' field - automatically created from dependencies\n\n"
                            f"{example_context}\n"
                            f"{yaml_examples}"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Plan:\n{plan}\n\n"
                            f"{composio_actions}{step_guidance}\n\n"
                            f"Generate nodes for: {original_user_query}"
                        )
                    }
                ]
                
                self.logger.info(f"[PROMPT_BUILD] Final prompt length: {len(minimal_prompt[0]['content'])} chars (system) + {len(minimal_prompt[1]['content'])} chars (user)")
                
                # ✅ DEBUG: Log the FULL user prompt content
                user_prompt_content = minimal_prompt[1]['content']
                self.logger.info(f"[PROMPT_FULL] ===== FULL USER PROMPT =====")
                self.logger.info(f"[PROMPT_FULL] {user_prompt_content}")
                self.logger.info(f"[PROMPT_FULL] ===== END USER PROMPT =====")
                
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
                    
                    # Extract workflow JSON with error handling
                    if "arguments" not in result:
                        error_msg = (
                            f"[FUNCTION_CALLING] LLM did not call function correctly. "
                            f"Response keys: {list(result.keys())}. "
                            f"This usually means the LLM returned text instead of a function call."
                        )
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # ✅ COMPATIBILITY: Accept both flat structure and wrapped structure
                    # Flat: {"arguments": {"triggers": [], "nodes": []}} - edges auto-generated
                    # Wrapped: {"arguments": {"workflow": {"triggers": [], "nodes": []}}}
                    arguments = result["arguments"]
                    if "workflow" in arguments:
                        # Old wrapped format
                        workflow_json = arguments["workflow"]
                    elif "nodes" in arguments:
                        # New flat format - edges will be auto-generated
                        workflow_json = arguments
                        
                        # Set defaults
                        if "triggers" not in workflow_json:
                            workflow_json["triggers"] = []
                        
                        # 🎯 SINGLE SOURCE OF TRUTH: Generate edges from explicit dependencies only
                        if "edges" not in workflow_json or not workflow_json["edges"]:
                            self.logger.info(f"[EDGE_GEN] Generating edges from explicit dependencies for {len(workflow_json['nodes'])} nodes...")
                            
                            # Build step index to node ID mapping (1-based indexing)
                            step_to_node = {}
                            for i, node in enumerate(workflow_json['nodes'], 1):
                                step_to_node[i] = node['id']
                            
                            # Generate edges from dependencies field
                            edges = []
                            validation_errors = []
                            
                            for i, node in enumerate(workflow_json['nodes'], 1):
                                node_id = node['id']
                                dependencies = node.get('dependencies', None)
                                
                                # Validate dependencies field exists
                                if dependencies is None:
                                    validation_errors.append(
                                        f"Node '{node_id}' (step {i}) missing REQUIRED 'dependencies' field"
                                    )
                                    continue
                                
                                # Validate dependencies are valid
                                if not isinstance(dependencies, list):
                                    validation_errors.append(
                                        f"Node '{node_id}' (step {i}) has invalid dependencies: {dependencies} "
                                        f"(must be list of integers)"
                                    )
                                    continue
                                
                                # Create edges from each dependency to this node
                                for dep_idx in dependencies:
                                    if not isinstance(dep_idx, int):
                                        validation_errors.append(
                                            f"Node '{node_id}' (step {i}) has non-integer dependency: {dep_idx}"
                                        )
                                        continue
                                    
                                    if dep_idx >= i:
                                        validation_errors.append(
                                            f"Node '{node_id}' (step {i}) has forward/self dependency: {dep_idx} "
                                            f"(dependencies must be < {i})"
                                        )
                                        continue
                                    
                                    if dep_idx not in step_to_node:
                                        validation_errors.append(
                                            f"Node '{node_id}' (step {i}) depends on non-existent step {dep_idx}"
                                        )
                                        continue
                                    
                                    source_node_id = step_to_node[dep_idx]
                                    
                                    # ✅ Determine edge type: NESTED for loop→body, SEQUENTIAL otherwise
                                    edge_type = 'sequential'
                                    source_node = next((n for n in workflow_json['nodes'] if n['id'] == source_node_id), None)
                                    if source_node and source_node.get('type') == 'loop':
                                        # Check if target is in loop body
                                        loop_body = source_node.get('loop_body', [])
                                        if node_id in loop_body:
                                            edge_type = 'nested'
                                    
                                    # Valid dependency - create edge with correct type
                                    edges.append({
                                        'type': edge_type,
                                        'from': source_node_id,
                                        'to': node_id
                                    })
                                    self.logger.info(f"[EDGE_GEN] Created edge: {source_node_id} → {node_id} (type={edge_type})")
                            
                            # FAIL FAST if validation errors found
                            if validation_errors:
                                error_msg = (
                                    f"⚠️ Workflow dependency validation failed:\n\n"
                                    + "\n".join(f"  • {err}" for err in validation_errors)
                                    + "\n\n"
                                    f"💡 FIX: Every node MUST have a 'dependencies' field:\n"
                                    f"  - First/independent nodes: dependencies=[]\n"
                                    f"  - Sequential nodes: dependencies=[previous_step_number]\n"
                                    f"  - Example: Step 2 depends on Step 1 → dependencies=[1]\n"
                                )
                                self.logger.error(f"[EDGE_GEN] {error_msg}")
                                post_proxy.update_attachment(error_msg, AttachmentType.revise_message)
                                post_proxy.update_send_to("CodeInterpreter")  # Retry with error feedback
                                return post_proxy.end()
                            
                            workflow_json['edges'] = edges
                            self.logger.info(f"[EDGE_GEN] ✅ Generated {len(edges)} edges from explicit dependencies")
                    else:
                        error_msg = (
                            f"[FUNCTION_CALLING] Function call missing workflow structure. "
                            f"Arguments received: {list(arguments.keys())}. "
                            f"Expected either 'workflow' wrapper or 'nodes' field. "
                            f"Full result: {json.dumps(result, indent=2)}"
                        )
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    self.logger.info(
                        f"[FUNCTION_CALLING] Got workflow: "
                        f"{len(workflow_json.get('nodes', []))} nodes, "
                        f"{len(workflow_json.get('edges', []))} edges"
                    )
                    
                    # Debug: Log actual workflow JSON to see placeholder formats
                    # json is already imported at top of file
                    self.logger.info(f"[WORKFLOW_JSON] Full workflow:\n{json.dumps(workflow_json, indent=2)}")
                    
                    # ✅ VALIDATE: Node count must match init_plan structure
                    if init_plan_with_markers and len(step_hints) > 0:
                        actual_node_count = len(workflow_json.get('nodes', []))
                        expected_node_count = len(step_hints)
                        
                        if actual_node_count != expected_node_count:
                            error_msg = (
                                f"🚨 NODE COUNT MISMATCH!\n"
                                f"   Expected: {expected_node_count} nodes (from init_plan)\n"
                                f"   Generated: {actual_node_count} nodes\n"
                                f"   Difference: {actual_node_count - expected_node_count:+d}\n\n"
                                f"   The LLM must generate EXACTLY {expected_node_count} nodes - no more, no less!\n"
                                f"   Each step in init_plan = 1 node in output.\n\n"
                                f"   Expected steps:\n"
                                + "\n".join(f"      - {h}" for h in step_hints)
                            )
                            self.logger.error(f"[NODE_COUNT_VALIDATION] {error_msg}")
                            
                            # Return error for retry
                            post_proxy.update_message(
                                error_msg + "\n\n⚠️ Please regenerate with EXACTLY the right number of nodes."
                            )
                            post_proxy.update_send_to("CodeInterpreter")
                            return post_proxy.end()
                    
                    # ⚡ PHASE 3: AUTO-CORRECTION (Forms + HITL + Invalid Params)
                    # Run BEFORE validation so corrections can fix issues
                    original_node_count = len(workflow_json.get('nodes', []))
                    workflow_json = self._auto_correct_workflow(
                        workflow_json=workflow_json,
                        user_prompt=query,
                        init_plan=init_plan_with_markers
                    )
                    corrected_node_count = len(workflow_json.get('nodes', []))
                    
                    if corrected_node_count != original_node_count:
                        self.logger.info(
                            f"[AUTO_CORRECT] Phase 3 corrections applied: "
                            f"{original_node_count} → {corrected_node_count} nodes"
                        )
                    
                    # ✅ VALIDATION: Comprehensive placeholder validation (Pydantic-based)
                    # Validates:
                    # 1. Simple placeholders (${{bare_name}}) use valid prefixes
                    # 2. Cross-node references (${{from_step:node.field}}) point to valid fields
                    # 3. Nested paths and array access patterns
                    # 4. Runtime keywords (loop_item, loop_index) are properly handled
                    validation_result = validate_workflow_placeholders(
                        workflow_dict=workflow_json,
                        tool_cache=self.composio_cache
                    )
                    
                    if not validation_result.valid:
                        self.logger.error(f"[PLACEHOLDER_VAL] {len(validation_result.errors)} validation errors")
                        for error in validation_result.errors[:5]:  # Show first 5
                            self.logger.error(f"  {error}")
                        
                        # ⚠️ VALIDATION ERRORS: Send back to CodeInterpreter for retry with detailed feedback
                        error_summary = "\n".join(validation_result.errors[:10])  # Top 10 errors
                        
                        # Build a completely dynamic error message based on actual errors
                        error_message = (
                            f"⚠️ Workflow validation failed with {len(validation_result.errors)} placeholder errors:\n\n"
                            f"{error_summary}\n\n"
                            f"💡 Fix these issues and regenerate the workflow."
                        )
                        post_proxy.update_attachment(error_message, AttachmentType.revise_message)
                        post_proxy.update_send_to("CodeInterpreter")  # ✅ Send to self for retry with error feedback
                        return post_proxy.end()
                    elif validation_result.warnings:
                        self.logger.warning(f"[PLACEHOLDER_VAL] {len(validation_result.warnings)} warnings (non-blocking)")
                        for warning in validation_result.warnings[:3]:  # Show first 3
                            self.logger.warning(f"  {warning}")
                    
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
        # ⚠️ UNREACHABLE: Function calling is always enabled
        # =====================================================================
        # If execution reaches here, it means:
        # 1. is_workflow_generation=False (not in workflow mode), OR
        # 2. use_function_calling=False (config disabled)
        # 
        # In production, both conditions should never occur for workflow generation.
        # This is a safety fallback that should never execute.
        self.logger.error(
            "[CODE_GENERATOR] ⚠️ UNREACHABLE CODE PATH REACHED! "
            f"is_workflow_generation={is_workflow_generation}, "
            f"use_function_calling={self.config.use_function_calling}"
        )
        post_proxy.update_attachment(
            "Internal error: Workflow generation mode is misconfigured. "
            "Please contact support.",
            AttachmentType.revise_message
        )
        post_proxy.update_send_to("User")
        return post_proxy.end()

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
