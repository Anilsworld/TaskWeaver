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

    def _convert_retry_loops_to_conditional_edges(
        self, 
        workflow_json: dict, 
        init_plan: str
    ) -> dict:
        """
        Detect and convert retry patterns to conditional_edges (Autogen pattern).
        
        UNIVERSAL DETECTION: Works for ANY workflow by detecting:
        1. Duplicate nodes (same tool_id appearing multiple times)
        2. Retry keywords in node descriptions ("retry", "regenerate", "if rejected")
        3. HITL/decision nodes between duplicates
        4. Loop nodes with retry intent
        
        Converts them to conditional_edges with loop-back to existing nodes.
        
        Args:
            workflow_json: Workflow dict from LLM
            init_plan: Original plan text to extract retry intent
            
        Returns:
            Modified workflow_json with conditional_edges instead of duplicates
        """
        nodes = workflow_json.get('nodes', [])
        retry_keywords = ['retry', 'regenerate', 'redo', 'if reject', 'if fail', 'loop back']
        
        self.logger.info(
            f"[RETRY_DETECT] 🔍 Starting universal retry detection on {len(nodes)} nodes"
        )
        
        # ========================================================================
        # PATTERN 1: Loop nodes with retry intent (nested sub-steps)
        # ========================================================================
        retry_loop_nodes = []
        
        for node in nodes:
            if node.get('type') == 'loop':
                node_desc = str(node.get('description', '')).lower()
                node_id = node.get('id', '')
                
                # Check if this loop is actually a retry pattern
                is_retry = any(keyword in node_desc or keyword in init_plan.lower() for keyword in retry_keywords)
                
                loop_body = node.get('loop_body', [])
                if is_retry and loop_body:
                    retry_loop_nodes.append((node_id, node, loop_body))
                    self.logger.info(
                        f"[RETRY_DETECT] 🔍 Pattern 1: Loop node {node_id} with "
                        f"{len(loop_body)} child nodes (retry keywords found)"
                    )
        
        # ========================================================================
        # PATTERN 2: Duplicate tool_id nodes (same tool appearing twice)
        # ========================================================================
        duplicate_patterns = []
        tool_id_map = {}  # tool_id → [(node_id, node, index)]
        
        for idx, node in enumerate(nodes):
            tool_id = node.get('tool_id')
            if tool_id:  # Only check nodes with tools
                if tool_id not in tool_id_map:
                    tool_id_map[tool_id] = []
                tool_id_map[tool_id].append((node.get('id'), node, idx))
        
        # Find duplicate tool_ids (same tool used 2+ times)
        for tool_id, node_instances in tool_id_map.items():
            if len(node_instances) >= 2:
                # Check if later instances mention retry keywords
                for node_id, node, idx in node_instances[1:]:  # Skip first occurrence
                    node_desc = str(node.get('description', '')).lower()
                    is_retry = any(keyword in node_desc for keyword in retry_keywords)
                    
                    if is_retry:
                        # This is a duplicate for retry purposes!
                        original_node_id = node_instances[0][0]  # First occurrence
                        duplicate_patterns.append((node_id, node, original_node_id, tool_id))
                        self.logger.info(
                            f"[RETRY_DETECT] 🔍 Pattern 2: Duplicate tool {tool_id} - "
                            f"{node_id} is retry of {original_node_id}"
                        )
        
        # ========================================================================
        # CONVERSION: Create conditional_edges and remove duplicates
        # ========================================================================
        nodes_to_remove = set()
        edges_to_remove = set()
        
        # Convert Pattern 1 (loop nodes)
        for loop_id, loop_node, child_node_ids in retry_loop_nodes:
            loop_deps = loop_node.get('dependencies', [])
            if not loop_deps:
                continue
            
            decision_node_id = loop_deps[0]  # HITL approval node
            
            # Find which existing node to loop back to
            loop_back_target = None
            if child_node_ids:
                first_child_id = child_node_ids[0]
                first_child = next((n for n in nodes if n.get('id') == first_child_id), None)
                
                if first_child and first_child.get('tool_id'):
                    target_tool_id = first_child.get('tool_id')
                    for node in nodes:
                        if (node.get('tool_id') == target_tool_id and 
                            node.get('id') != first_child_id and
                            node.get('id') < loop_id):
                            loop_back_target = node.get('id')
                            break
            
            if loop_back_target:
                self._add_conditional_edge(
                    workflow_json, decision_node_id, loop_back_target
                )
                nodes_to_remove.add(loop_id)
                nodes_to_remove.update(child_node_ids)
                edges_to_remove.update([loop_id] + list(child_node_ids))
                
                self.logger.info(
                    f"[RETRY_CONVERT] ✅ Pattern 1: {decision_node_id} --[if Reject]--> {loop_back_target}"
                )
        
        # Convert Pattern 2 (duplicate tool nodes)
        for duplicate_id, duplicate_node, original_id, tool_id in duplicate_patterns:
            # Find the decision node between original and duplicate
            duplicate_deps = duplicate_node.get('dependencies', [])
            
            # Look for HITL node in dependencies
            decision_node_id = None
            for dep_id in duplicate_deps:
                dep_node = next((n for n in nodes if n.get('id') == dep_id), None)
                if dep_node and dep_node.get('type') == 'hitl':
                    decision_node_id = dep_id
                    break
            
            if decision_node_id and original_id:
                self._add_conditional_edge(
                    workflow_json, decision_node_id, original_id
                )
                nodes_to_remove.add(duplicate_id)
                edges_to_remove.add(duplicate_id)
                
                self.logger.info(
                    f"[RETRY_CONVERT] ✅ Pattern 2: {decision_node_id} --[if Reject]--> {original_id}"
                )
                self.logger.info(
                    f"[RETRY_CONVERT] 🗑️  Removed duplicate: {duplicate_id} (duplicate of {original_id})"
                )
        
        # ========================================================================
        # CLEANUP: Remove duplicate nodes and their edges
        # ========================================================================
        if nodes_to_remove:
            workflow_json['nodes'] = [
                n for n in nodes if n.get('id') not in nodes_to_remove
            ]
            
            workflow_json['edges'] = [
                e for e in workflow_json.get('edges', [])
                if e.get('from') not in edges_to_remove and e.get('to') not in edges_to_remove
            ]
            
            workflow_json['sequential_edges'] = [
                e for e in workflow_json.get('sequential_edges', [])
                if (isinstance(e, tuple) and e[0] not in edges_to_remove and e[1] not in edges_to_remove) or
                   (isinstance(e, dict) and e.get('from') not in edges_to_remove and e.get('to') not in edges_to_remove)
            ]
            
            self.logger.info(
                f"[RETRY_CONVERT] 🎉 Removed {len(nodes_to_remove)} duplicate nodes, "
                f"created {len(workflow_json.get('conditional_edges') or [])} conditional edges"
            )
        
        # ========================================================================
        # CLEANUP: Remove orphan nodes created after duplicate removal
        # ========================================================================
        # After removing duplicate nodes, some nodes may now have missing dependencies.
        # Re-validate and remove these orphans (e.g., node_7 depending on removed node_6).
        existing_node_ids = {n.get('id') for n in workflow_json.get('nodes', [])}
        orphan_cleanup_iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while orphan_cleanup_iterations < max_iterations:
            orphan_nodes = []
            
            for node in workflow_json.get('nodes', []):
                node_id = node.get('id')
                dependencies = node.get('dependencies', [])
                
                # Check if any dependency doesn't exist
                missing_deps = [dep for dep in dependencies if dep not in existing_node_ids]
                
                if missing_deps:
                    orphan_nodes.append((node_id, missing_deps))
                    self.logger.warning(
                        f"[RETRY_CLEANUP] Removing orphan node {node_id}: "
                        f"depends on non-existent {missing_deps}"
                    )
            
            # If no orphans found, we're done
            if not orphan_nodes:
                break
            
            # Remove orphan nodes
            orphan_ids = {node_id for node_id, _ in orphan_nodes}
            workflow_json['nodes'] = [
                n for n in workflow_json.get('nodes', [])
                if n.get('id') not in orphan_ids
            ]
            
            # Remove edges to/from orphans
            workflow_json['edges'] = [
                e for e in workflow_json.get('edges', [])
                if e.get('from') not in orphan_ids and e.get('to') not in orphan_ids
            ]
            
            # Update existing node IDs for next iteration
            existing_node_ids = {n.get('id') for n in workflow_json.get('nodes', [])}
            nodes_to_remove.update(orphan_ids)
            orphan_cleanup_iterations += 1
            
            self.logger.info(
                f"[RETRY_CLEANUP] Iteration {orphan_cleanup_iterations}: "
                f"Removed {len(orphan_nodes)} orphan node(s)"
            )
        
        # Final summary log
        total_conditionals = len(workflow_json.get('conditional_edges') or [])
        total_nodes_remaining = len(workflow_json.get('nodes', []))
        self.logger.info(
            f"[RETRY_DETECT] ✅ Universal retry detection complete: "
            f"{len(retry_loop_nodes)} loop patterns, "
            f"{len(duplicate_patterns)} duplicate patterns, "
            f"{len(nodes_to_remove)} nodes removed/cleaned, "
            f"{total_conditionals} conditional_edges, "
            f"{total_nodes_remaining} final nodes"
        )
        
        return workflow_json
    
    def _add_conditional_edge(
        self,
        workflow_json: dict,
        decision_node_id: str,
        loop_back_target: str
    ):
        """
        Add a conditional_edge for retry loop-back (Autogen pattern).
        
        Args:
            workflow_json: Workflow dict to modify
            decision_node_id: ID of the HITL/decision node
            loop_back_target: ID of the node to loop back to on rejection
        """
        if 'conditional_edges' not in workflow_json or workflow_json['conditional_edges'] is None:
            workflow_json['conditional_edges'] = []
        
        # Universal condition pattern (works for any HITL with 'decision' field)
        conditional_edge = {
            "source": decision_node_id,
            "condition": f"${{{{{decision_node_id}.decision}}}} == 'Reject'",
            "if_true": loop_back_target,  # If Reject (condition TRUE) → loop back
            "if_false": "END"  # If Approve (condition FALSE) → end workflow
        }
        
        workflow_json['conditional_edges'].append(conditional_edge)
        
        self.logger.info(
            f"[RETRY_CONVERT] 📌 Added conditional_edge: "
            f"{decision_node_id} --[condition: Reject]--> {loop_back_target}"
        )
    
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
                
                # ✅ Create user-friendly version of plan (remove technical $1, $2 syntax for display)
                # This is what the LLM will use for generating descriptions
                def clean_plan_for_display(plan_text: str) -> str:
                    """Remove technical dependency syntax from plan for user-friendly descriptions."""
                    lines = plan_text.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # Keep the step number and marker, remove technical refs
                        cleaned = re.sub(r'\s+from\s+\$\d+(?:\.\d+)*(?:\s*(?:,|and)\s*\$\d+(?:\.\d+)*)*', '', line)
                        cleaned = re.sub(r'\s+after\s+\$\d+(?:\.\d+)*(?:\s*(?:,|and)\s*\$\d+(?:\.\d+)*)*', '', cleaned)
                        cleaned = re.sub(r'\s+in\s+\$\d+(?:\.\d+)*', '', cleaned)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        cleaned_lines.append(cleaned)
                    return '\n'.join(cleaned_lines)
                
                plan_for_display = clean_plan_for_display(plan)
                self.logger.info(f"[PROMPT_BUILD] Cleaned plan for descriptions (first 300 chars): {plan_for_display[:300]}")
                
                # ✅ SKIP EXAMPLES for function calling - schema is already very detailed
                # Loading 4 examples adds ~10K tokens and causes function calling issues
                # The JSON schema provides all the information the LLM needs
                example_context = ""
                yaml_examples = ""
                
                # ✅ Helper function to clean step descriptions
                def clean_step_description(step_text: str) -> str:
                    """
                    Remove technical syntax from step descriptions to make them user-friendly.
                    - Removes: from $1, from $2, from $1 and $2, from $1, $2
                    - Removes: after $X, after $Y
                    - Preserves: Everything else
                    """
                    cleaned = step_text
                    
                    # Remove "from $X, $Y, ..." patterns (handles various formats)
                    cleaned = re.sub(r'\s+from\s+\$\d+(?:\.\d+)*(?:\s*(?:,|and)\s*\$\d+(?:\.\d+)*)*', '', cleaned)
                    
                    # Remove "after $X" patterns
                    cleaned = re.sub(r'\s+after\s+\$\d+(?:\.\d+)*(?:\s*(?:,|and)\s*\$\d+(?:\.\d+)*)*', '', cleaned)
                    
                    # Remove "in $X" patterns (for loops)
                    cleaned = re.sub(r'\s+in\s+\$\d+(?:\.\d+)*', '', cleaned)
                    
                    # Clean up extra whitespace
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                    
                    return cleaned
                
                # ✅ Build step guidance: Map init_plan steps to node types AND extract control dependencies
                step_guidance = ""
                self.logger.info(f"[STEP_GUIDANCE] Building step guidance, init_plan length: {len(init_plan_with_markers)}")
                if init_plan_with_markers:
                    import re
                    step_hints = []
                    
                    # ✅ Fix: Handle both real newlines AND escaped \n strings
                    # Sometimes the plan comes with escaped newlines from JSON serialization
                    plan_text = init_plan_with_markers.replace('\\n', '\n')
                    
                    for line in plan_text.split('\n'):
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
                                # 🎯 UPDATED: Support hierarchical steps (e.g., $2.1, $2.2.3)
                                after_matches = re.findall(r'\$(\d+(?:\.\d+)*)', after_clause)
                                if after_matches:
                                    after_deps = after_matches  # Keep as strings: ['2', '2.1']
                                    self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'after' control deps: {after_deps}")
                            
                            # Pattern: Find ALL $X references after "from" keyword
                            # Handles: "from $1", "from $2 and $3", "from $1, $2, and $3"
                            # 🎯 UPDATED: Supports hierarchical steps like "from $2, $2.1"
                            from_deps = []
                            if 'from' in full_line:
                                from_clause = full_line.split('from', 1)[1]
                                if 'after' in from_clause:
                                    from_clause = from_clause.split('after', 1)[0]  # Stop at "after"
                                # 🎯 UPDATED: Support hierarchical steps (e.g., $2.1, $2.2.3)
                                from_matches = re.findall(r'\$(\d+(?:\.\d+)*)', from_clause)
                                if from_matches:
                                    from_deps = from_matches  # Keep as strings: ['2', '2.1']
                                    self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'from' data deps: {from_deps}")
                            
                            # Pattern: Find ALL $X references after "in"/"over" keyword (for loops)
                            # Uses precise regex to avoid false matches like "included", "within"
                            # 🎯 UPDATED: Supports hierarchical steps
                            in_deps = []
                            if not from_deps:  # Only check 'in'/'over' if 'from' wasn't found
                                in_pattern = r'\b(?:in|over)\s+\$(\d+(?:\.\d+)*)'
                                in_matches = re.findall(in_pattern, full_line)
                                if in_matches:
                                    in_deps = in_matches  # Keep as strings
                                    self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'in/over' data deps (loop): {in_deps}")
                                # Also check for comma-separated items: "in $1, $2, and $3"
                                # 🎯 UPDATED: Support hierarchical steps (e.g., "in $2.1")
                                elif re.search(r'\b(?:in|over)\s+\$\d+(?:\.\d+)*', full_line):
                                    match_in = re.search(r'\b(?:in|over)\s+(.+?)(?:\s+(?:after|<)|$)', full_line)
                                    if match_in:
                                        in_clause = match_in.group(1)
                                        # 🎯 UPDATED: Support hierarchical steps
                                        in_matches = re.findall(r'\$(\d+(?:\.\d+)*)', in_clause)
                                        if in_matches:
                                            in_deps = in_matches  # Keep as strings
                                            self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found 'in/over' data deps (loop, multi): {in_deps}")
                            
                            # 🎯 UNIVERSAL DEPENDENCY LOGIC:
                            # Combine ALL dependencies from 'from', 'after', and 'in/over' clauses
                            # If step says "from $3 after $2" → dependencies = [2, 3] (BOTH!)
                            # If step says "from $2, $2.1" → dependencies = ['2', '2.1'] (hierarchical)
                            all_deps = set()
                            all_deps.update(from_deps)
                            all_deps.update(after_deps)
                            all_deps.update(in_deps)
                            
                            # Natural sort for hierarchical steps (e.g., '2' < '2.1' < '10')
                            def natural_sort_key(step_str):
                                """Convert '2.1.3' to [2, 1, 3] for natural sorting"""
                                return [int(x) for x in step_str.split('.')]
                            
                            control_deps = sorted(list(all_deps), key=natural_sort_key) if all_deps else []
                            
                            # 🎯 BUILD HINT: Show node IDs (not integers) to guide LLM
                            # The hint tells the LLM what to generate in the workflow JSON
                            # Map step numbers to node IDs: "2" → "node_2", "2.1" → "node_2_1"
                            deps_hint = ""
                            if control_deps:
                                # ✅ FIXED: Handle hierarchical steps properly (2.1 → node_2_1)
                                node_id_deps = [f"'node_{d.replace('.', '_')}'" for d in sorted(set(control_deps))]
                                deps_hint = f", dependencies=[{', '.join(node_id_deps)}]"
                            else:
                                deps_hint = f", dependencies=[]"
                            
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
                                # 🎯 NEW: Extract nested child steps for loop_body
                                # If this is step 6, find all steps like 6.1, 6.2, 6.3, etc.
                                child_steps = []
                                for check_line in plan_text.split('\n'):
                                    child_match = re.match(r'^\s*(' + re.escape(step_num) + r'\.(\d+))\.\s+', check_line)
                                    if child_match:
                                        child_step_num = child_match.group(1)  # e.g., "6.1"
                                        child_node_id = f"node_{child_step_num.replace('.', '_')}"  # "node_6_1"
                                        child_steps.append(child_node_id)
                                
                                # Build loop_body and loop_over hints
                                loop_body_hint = ""
                                loop_over_hint = ""
                                
                                if child_steps:
                                    loop_body_list = ", ".join(f"'{nid}'" for nid in child_steps)
                                    loop_body_hint = f", loop_body=[{loop_body_list}]"
                                    self.logger.info(f"[STEP_GUIDANCE] Step {step_num}: Found {len(child_steps)} nested steps: {child_steps}")
                                
                                # Determine loop_over from dependencies
                                if control_deps:
                                    # Loop depends on a previous node - use that for loop_over
                                    source_node = f"node_{control_deps[0].replace('.', '_')}"
                                    loop_over_hint = f", loop_over='{source_node}'"
                                
                                hint = f"Step {step_num}: type='loop'{loop_over_hint}{loop_body_hint}{deps_hint}"
                                step_hints.append(hint)
                                self.logger.info(f"[STEP_GUIDANCE] Added hint: {hint}")
                            
                            # ✅ Log after each step to track progress
                            self.logger.info(f"[STEP_GUIDANCE] Generated {len(step_hints)} step hints")
                    
                    if step_hints:
                        num_steps = len(step_hints)
                        step_guidance = f"\n\n**GENERATE ALL {num_steps} NODES:**\n" + "\n".join(f"- {h}" for h in step_hints)
                        step_guidance += f"\n\n⚠️ Generate a node for EVERY step above (ALL {num_steps} nodes required)!"
                        step_guidance += f"\n\n🚨 CRITICAL: Generate EXACTLY {num_steps} nodes - NO MORE, NO LESS!"
                        step_guidance += f"\n   Do NOT expand or add extra nodes beyond the {num_steps} steps listed above!"
                        step_guidance += f"\n   Each step in the plan = EXACTLY 1 node in the output!"
                        step_guidance += f"\n\n🚨 LOOP NESTING RULE: Nested steps (e.g., 6.1, 6.2) go in BOTH places:"
                        step_guidance += f"\n   1. As separate nodes in top-level 'nodes' array (e.g., node_6_1, node_6_2)"
                        step_guidance += f"\n   2. Referenced by parent loop's 'loop_body' field (e.g., loop_body=['node_6_1', 'node_6_2'])"
                        step_guidance += f"\n   Example: Step 6 <loop> with 6.1, 6.2 → Create node_6 with loop_body=['node_6_1', 'node_6_2'], PLUS node_6_1 and node_6_2 in nodes array"
                        step_guidance += f"\n\n🚨 DEPENDENCIES RULE: For EACH node, copy the EXACT 'dependencies=[...]' from its step above!"
                        step_guidance += f"\n   - DO NOT use empty dependencies=[] unless the step explicitly shows dependencies=[]"
                        step_guidance += f"\n   - The schema default is WRONG - you MUST follow the step guidance exactly!"
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
                            "**LOOP BODY (CRITICAL):**\n"
                            "   • loop_body=['node_6_1', 'node_6_2'] - List of node IDs to execute per iteration\n"
                            "   • These nodes are defined separately in top-level nodes array\n"
                            "   • If plan shows '6. Loop <loop>' with sub-steps '6.1', '6.2' → loop_body=['node_6_1', 'node_6_2']\n"
                            "   • Access current iteration value: '${{{{loop_item}}}}'\n\n"
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
                            f"Plan:\n{plan_for_display}\n\n"
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
                
                # ✅ NEW: Use Instructor with Intelligent Retry (Autogen-inspired)
                # Features:
                # - Schema generation from Pydantic models
                # - LLM-based retry decision (stops on unfixable errors)
                # - Dependencies validation
                # - App name extraction
                # - HITL decision field auto-add
                # - Params population checks
                # - Reasoning transparency (logs why retrying)
                try:
                    from taskweaver.code_interpreter.code_interpreter.instructor_workflow_generator import (
                        create_intelligent_instructor_client
                    )
                    
                    # Initialize Intelligent Retry Wrapper with our LLM client
                    # Access OpenAI client via: llm_api → completion_service → client
                    intelligent_instructor = create_intelligent_instructor_client(
                        openai_client=self.llm_api.completion_service.client,
                        max_retries=3
                    )
                    
                    # Generate workflow using Intelligent Retry
                    # LLM decides whether to retry on validation errors (not blind retry)
                    # ✅ CRITICAL FIX: Append step_guidance to composio_actions (like the old code did)
                    # The old code appended step_guidance to composio_actions at line 819:
                    # f"{composio_actions}{step_guidance}\n\n"
                    composio_with_guidance = composio_actions
                    if step_guidance:
                        composio_with_guidance += "\n\n" + step_guidance
                        self.logger.info(f"[CODE_GENERATOR] ✅ Appended step_guidance ({len(step_guidance)} chars) to composio_actions")
                    else:
                        self.logger.warning("[CODE_GENERATOR] ⚠️ NO step_guidance to append!")
                    
                    # Note: Complex dependency analysis is handled by separate services:
                    # - apps.py_workflows.generation.analysis.dependency_analyzer (schema-based analysis)
                    # - apps.py_workflows.generation.orchestrator.dependency_resolver (auto-injection)
                    # Instructor does Pydantic validation (field types, node existence, tool IDs, important params)
                    
                    # 🔧 CRITICAL FIX: Build tool_schemas BEFORE calling Instructor
                    # This enables Pydantic validators to run DURING generation (not after)
                    tool_schemas = []
                    if session_id:
                        try:
                            from TaskWeaver.project.plugins.composio_action_selector import _BATCH_CACHE
                            if session_id in _BATCH_CACHE:
                                for step_query, action_dicts in _BATCH_CACHE[session_id].items():
                                    tool_schemas.extend(action_dicts)
                                self.logger.info(f"[PRE-VALIDATION] ✅ Loaded {len(tool_schemas)} tool schemas for Instructor validation")
                        except Exception as e:
                            self.logger.debug(f"[PRE-VALIDATION] Could not load tool schemas: {e}")
                    
                    workflow_def, instructor_error = intelligent_instructor.generate_with_intelligent_retry(
                        user_request=original_user_query,
                        available_tools=tool_ids,
                        plan=plan,
                        model=self.llm_api.config.model,  # ✅ Access model via llm_api.config
                        temperature=0.0,
                        composio_actions_text=composio_with_guidance,  # ✅ Includes step_guidance
                        tool_schemas=tool_schemas if tool_schemas else None,  # 🔧 NEW: Enable validators
                        max_retries=3
                    )
                    
                    if instructor_error:
                        # Instructor failed after max retries
                        self.logger.error(f"[INSTRUCTOR] Generation failed: {instructor_error}")
                        post_proxy.update_attachment(
                            f"Workflow generation failed:\n{instructor_error}",
                            AttachmentType.revise_message
                        )
                        post_proxy.update_send_to("User")
                        return post_proxy.end()
                    
                    # Convert Pydantic model to dict for remaining checks
                    workflow_json = workflow_def.model_dump()
                    
                    # ✅ POST-PROCESSING: Clean up technical syntax from descriptions
                    def clean_description(desc: str) -> str:
                        """Remove technical $1, $2 syntax from descriptions."""
                        if not desc:
                            return desc
                        cleaned = re.sub(r'\s+from\s+\$\d+(?:\.\d+)*(?:\s*(?:,|and)\s*\$\d+(?:\.\d+)*)*', '', desc)
                        cleaned = re.sub(r'\s+after\s+\$\d+(?:\.\d+)*(?:\s*(?:,|and)\s*\$\d+(?:\.\d+)*)*', '', cleaned)
                        cleaned = re.sub(r'\s+in\s+\$\d+(?:\.\d+)*', '', cleaned)
                        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                        return cleaned
                    
                    for node in workflow_json.get('nodes', []):
                        if 'description' in node and node['description']:
                            original = node['description']
                            cleaned = clean_description(original)
                            if cleaned != original:
                                node['description'] = cleaned
                                self.logger.info(f"[CLEANUP] Cleaned description: '{original}' → '{cleaned}'")
                    
                    self.logger.info(
                        f"[INSTRUCTOR] ✅ Generated workflow: "
                        f"{len(workflow_json.get('nodes', []))} nodes, "
                        f"{len(workflow_json.get('edges', []))} edges"
                    )
                    
                    # ✅ VALIDATION NOTE: All validation (tool IDs, important params, dependencies)
                    # now happens DURING Instructor generation via Pydantic validators.
                    # No need for redundant post-generation validation.
                    # Optimization opportunities are detected later by WorkflowOptimizer.
                    
                    # ⚠️ KEPT FROM ORIGINAL: Edge generation (if edges not already generated)
                    # This is business logic, not validation, so it stays
                    # NOTE: Instructor may have already generated edges via dependencies
                    if "edges" not in workflow_json or not workflow_json["edges"]:
                        self.logger.info(f"[EDGE_GEN] Generating edges from node dependencies...")
                        
                        # Build node ID lookup for validation
                        all_node_ids = {node['id'] for node in workflow_json['nodes']}
                        self.logger.info(f"[EDGE_GEN] Available node IDs: {sorted(all_node_ids)}")
                        
                        # Generate edges from dependencies field
                        edges = []
                        validation_errors = []
                        
                        for node in workflow_json['nodes']:
                            node_id = node['id']
                            dependencies = node.get('dependencies', None)
                            
                            # Dependencies validation is now in Instructor, but double-check here
                            if dependencies is None:
                                validation_errors.append(
                                    f"Node '{node_id}' missing REQUIRED 'dependencies' field"
                                )
                                continue
                            
                            if not isinstance(dependencies, list):
                                validation_errors.append(
                                    f"Node '{node_id}' has invalid dependencies: {dependencies} "
                                    f"(must be list of node IDs)"
                                )
                                continue
                            
                            # Filter out self-dependencies (auto-fix)
                            valid_dependencies = [dep for dep in dependencies if dep != node_id]
                            
                            if len(valid_dependencies) < len(dependencies):
                                removed_self_deps = [dep for dep in dependencies if dep == node_id]
                                self.logger.warning(
                                    f"[EDGE_GEN] ⚠️ Auto-fixed: Node '{node_id}' had self-dependency {removed_self_deps} - removed automatically"
                                )
                                node['dependencies'] = valid_dependencies
                            
                            # Create edges from each dependency to this node
                            for dep_node_id in valid_dependencies:
                                if not isinstance(dep_node_id, str):
                                    validation_errors.append(
                                        f"Node '{node_id}' has non-string dependency: {dep_node_id} "
                                        f"(must be node ID like 'node_1', 'node_2_1')"
                                    )
                                    continue
                                
                                if dep_node_id not in all_node_ids:
                                    validation_errors.append(
                                        f"Node '{node_id}' depends on non-existent node: '{dep_node_id}'"
                                    )
                                    continue
                                
                                # Determine edge type: NESTED for loop→body, SEQUENTIAL otherwise
                                edge_type = 'sequential'
                                source_node = next((n for n in workflow_json['nodes'] if n['id'] == dep_node_id), None)
                                if source_node and source_node.get('type') == 'loop':
                                    loop_body = source_node.get('loop_body', [])
                                    if node_id in loop_body:
                                        edge_type = 'nested'
                                
                                # Valid dependency - create edge with correct type
                                edges.append({
                                    'type': edge_type,
                                    'from': dep_node_id,
                                    'to': node_id
                                })
                                self.logger.info(f"[EDGE_GEN] Created edge: {dep_node_id} → {node_id} (type={edge_type})")
                        
                        # FAIL FAST if validation errors found
                        if validation_errors:
                            error_msg = (
                                f"⚠️ Workflow dependency validation failed:\n\n"
                                + "\n".join(f"  • {err}" for err in validation_errors)
                                + "\n\n"
                                f"💡 FIX: Every node MUST have a 'dependencies' field with node IDs:\n"
                                f"  - First/independent nodes: dependencies=[]\n"
                                f"  - Sequential nodes: dependencies=['node_1']\n"
                                f"  - Nested nodes: dependencies=['node_2', 'node_2_1']\n"
                                f"  - Example: node_2 depends on node_1 → dependencies=['node_1']\n"
                            )
                            self.logger.error(f"[EDGE_GEN] {error_msg}")
                            post_proxy.update_attachment(error_msg, AttachmentType.revise_message)
                            post_proxy.update_send_to("CodeInterpreter")
                            return post_proxy.end()
                        
                        # Create sequential edges WITHIN loop bodies
                        for node in workflow_json['nodes']:
                            if node.get('type') == 'loop':
                                loop_body = node.get('loop_body', [])
                                if len(loop_body) > 1:
                                    self.logger.info(f"[EDGE_GEN] Analyzing loop body of {node['id']}: {loop_body}")
                                    
                                    for i in range(len(loop_body) - 1):
                                        current_node_id = loop_body[i]
                                        next_node_id = loop_body[i + 1]
                                        
                                        # Check if edge already exists
                                        existing_edge = any(
                                            e['from'] == current_node_id and e['to'] == next_node_id
                                            for e in edges
                                        )
                                        
                                        if not existing_edge:
                                            edges.append({
                                                'type': 'sequential',
                                                'from': current_node_id,
                                                'to': next_node_id
                                            })
                                            self.logger.info(f"[EDGE_GEN] Created inferred edge within loop: {current_node_id} → {next_node_id} (type=sequential)")
                        
                        workflow_json['edges'] = edges
                        self.logger.info(f"[EDGE_GEN] ✅ Generated {len(edges)} edges from node dependencies")
                    
                    # ⚠️ ORIGINAL CODE CONTINUES BELOW (node count validation, etc.)
                    # This code is intentionally preserved for gradual migration
                    
                    # FAKE: Create a result object that looks like function calling output
                    # This allows the rest of the code to work without changes
                    result = {"arguments": workflow_json}
                    
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
                        # ✅ SIMPLIFIED: Instructor already generated workflow_json with validation
                        # This branch is now handled by Instructor above
                        workflow_json = arguments
                        
                        # Set defaults
                        if "triggers" not in workflow_json:
                            workflow_json["triggers"] = []
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
                    
                    # ✅ Workflow JSON logging disabled to reduce verbosity
                    # Debug: Uncomment below to see full workflow structure if needed
                    # self.logger.info(f"[WORKFLOW_JSON] Full workflow:\n{json.dumps(workflow_json, indent=2)}")
                    
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
                    
                    # ✅ REMOVED: Auto-correction now handled by Instructor validators
                    # - App name extraction: EnhancedWorkflowNode.extract_app_name_from_tool_id
                    # - HITL decision field: EnhancedWorkflowDefinition.add_hitl_decision_field
                    # - Invalid params: EnhancedWorkflowNode.ensure_params_populated
                    
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
                    
                    # ✅ REMOVED: Basic Pydantic validation now handled by Instructor
                    # Instructor's EnhancedWorkflowDefinition already validates:
                    # - Node types (via Literal type hints)
                    # - Required fields (via Pydantic Field(...))
                    # - Edge structure (via base WorkflowDefinition)
                    # - Cross-node dependencies (via validate_all_dependencies_exist)
                    
                    # ✅ workflow_json now has complete edges from edge generation above
                    self.logger.info(f"[FUNCTION_CALLING] After validation: {len(workflow_json.get('edges', []))} complete edges")
                    
                    # 🎯 ATTACH TOOL SCHEMAS TO WORKFLOW (for WorkflowOptimizer)
                    # This makes tool schemas available across process boundaries
                    if session_id:
                        try:
                            from TaskWeaver.project.plugins.composio_action_selector import _BATCH_CACHE
                            if session_id in _BATCH_CACHE:
                                tool_schemas_metadata = []
                                for step_query, action_dicts in _BATCH_CACHE[session_id].items():
                                    tool_schemas_metadata.extend(action_dicts)
                                
                                # Add as metadata field (will be available in consumer)
                                workflow_json['_tool_schemas'] = tool_schemas_metadata
                                self.logger.info(
                                    f"[TOOL_SCHEMAS] Attached {len(tool_schemas_metadata)} tool schemas to workflow"
                                )
                        except Exception as e:
                            self.logger.debug(f"[TOOL_SCHEMAS] Could not attach schemas: {e}")
                    
                    # 🎯 POST-PROCESS: Convert retry loops to conditional_edges (Autogen pattern)
                    workflow_json = self._convert_retry_loops_to_conditional_edges(
                        workflow_json, 
                        init_plan_with_markers
                    )
                    
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
