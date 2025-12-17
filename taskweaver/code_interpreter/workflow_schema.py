"""
Pydantic-based Workflow Validation
===================================
Fully scalable, adaptive schema validation for WORKFLOW dicts.
No hardcoding - just update the schema when requirements change.

Integration with WorkflowIR:
- Pydantic validates schema (types, required fields)
- WorkflowIR validates DAG logic (cycles, connectivity, data flow)
"""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import logging

logger = logging.getLogger(__name__)


class WorkflowParams(BaseModel):
    """Dynamic parameters - allows any field."""
    class Config:
        extra = "allow"  # Allow any additional fields


class WorkflowNode(BaseModel):
    """Single node in workflow."""
    id: str = Field(..., description="Unique node identifier")
    type: Literal[
        "agent_with_tools", 
        "agent_only", 
        "hitl", 
        "form",
        "code_execution",
        "loop",
        "parallel"
    ] = Field(..., description="Node execution type")
    
    # Optional fields
    tool_id: Optional[str] = Field(None, description="Composio tool ID")
    app_name: Optional[str] = Field(None, description="App name (gmail, slack, etc)")
    params: Union[Dict[str, Any], WorkflowParams] = Field(default_factory=dict, description="Tool parameters")
    description: Optional[str] = Field(None, description="Node description")
    
    # ✅ REQUIRED: Explicit dependency declaration (single source of truth)
    dependencies: List[str] = Field(
        default_factory=list,
        description="REQUIRED: List of node IDs this node depends on (e.g., ['node_1', 'node_2']). Use [] for independent/first nodes."
    )
    
    # Advanced fields
    decision: Optional[str] = Field(None, description="HITL decision field")
    blocking: Optional[bool] = Field(None, description="Blocks workflow until completion")
    form_schema: Optional[Dict[str, Any]] = Field(None, description="Form schema for input nodes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    # ✅ Code execution specific (for code_execution type)
    code: Optional[str] = Field(None, description="Python code to execute")
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"
    
    @model_validator(mode='after')
    def validate_tool_and_decision(self):
        """Validate that agent_with_tools nodes have tool_id and HITL nodes have decision."""
        # Validate tool_id for agent_with_tools
        if self.type == 'agent_with_tools' and not self.tool_id:
            raise ValueError(f"agent_with_tools nodes must have tool_id")
        
        # Validate decision for blocking HITL nodes
        if self.type == 'hitl' and self.blocking and not self.decision:
            raise ValueError(f"Blocking HITL nodes must have 'decision' field for conditional routing")
        
        return self


class WorkflowDefinition(BaseModel):
    """
    Complete workflow structure.
    
    CRITICAL STRUCTURE:
    {
      "nodes": [{"id": "...", "type": "...", ...}, ...],  # Array of node objects
      "edges": [...],                                      # Top-level field (NOT inside nodes)
      "sequential_edges": [...],                           # Top-level field (NOT inside nodes)
      "parallel_edges": [...]                              # Top-level field (NOT inside nodes)
    }
    
    ❌ WRONG: {"nodes": [{...node1...}, {...node2...}, {"edges": []}]}
    ✅ RIGHT: {"nodes": [{...node1...}, {...node2...}], "edges": [...]}
    """
    nodes: List[WorkflowNode] = Field(..., min_items=1, description="Array of workflow node objects. Each node MUST have 'id' and 'type' fields.")
    edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="TOP-LEVEL field: Edges connecting workflow nodes (auto-inferred from dependencies). This is NOT part of the nodes array."
    )
    # Conditional routing for approvals/retries/loop-backs (Autogen-inspired pattern)
    conditional_edges: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description=(
            "Conditional routing for decision-based workflow branching (e.g., approval, retry, error handling). "
            "Each edge must have: 'source' (decision node ID), 'condition' (expression to evaluate), "
            "'if_true' (target node if condition is true), 'if_false' (target node if condition is false). "
            "CRITICAL: Cycles (loop-backs) REQUIRE a condition field to prevent infinite loops. "
            "Examples:\n"
            "- Approval branch: {'source': 'approval', 'condition': '${approval.decision} == \"Approve\"', "
            "'if_true': 'send_email', 'if_false': 'END'}\n"
            "- Retry loop: {'source': 'approval', 'condition': '${approval.decision} == \"Reject\"', "
            "'if_true': 'END', 'if_false': 'regenerate_node'} (loops back with condition)\n"
            "- Multi-target: {'source': 'check', 'if_true': ['action_a', 'action_b'], 'if_false': 'fallback'}"
        )
    )
    # Legacy fields for backward compatibility
    sequential_edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="DEPRECATED: Use 'edges' instead"
    )
    parallel_edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="DEPRECATED: Use 'edges' with type='parallel'"
    )
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"
    
    @model_validator(mode='before')
    @classmethod
    def validate_nodes_structure(cls, data):
        """
        Auto-fix LLM JSON malformation: Prevent edges from being nested inside nodes array.
        
        Common LLM mistake:
        {"nodes": [{...node1...}, {...node2...}, {"edges": [], "sequential_edges": []}]}
        
        Correct structure:
        {"nodes": [...], "edges": [...], "sequential_edges": [...]}
        
        Safety: Only hoists edges from nodes array to top-level. Doesn't affect edges
        added later by optimizer (START edges, parallel groups, etc.) since those run
        after this validator.
        """
        if isinstance(data, dict) and 'nodes' in data:
            nodes = data['nodes']
            if isinstance(nodes, list) and len(nodes) > 0:
                # Check if any item in nodes is actually edges metadata (LLM mistake)
                nodes_to_remove = []
                for i, node in enumerate(nodes):
                    if isinstance(node, dict):
                        # If a node has 'edges' or 'sequential_edges' as keys but no 'id' or 'type',
                        # it's likely the LLM put edges metadata inside nodes array
                        has_edge_keys = any(k in node for k in ['edges', 'sequential_edges', 'parallel_edges', 'conditional_edges'])
                        has_node_keys = any(k in node for k in ['id', 'type', 'tool_id'])
                        
                        if has_edge_keys and not has_node_keys:
                            # LLM mistake detected - hoist edge data to top level (only if not already set)
                            logger.warning(f"[SCHEMA_FIX] 🔧 LLM put edges inside nodes[{i}], hoisting to top level")
                            for edge_key in ['edges', 'sequential_edges', 'parallel_edges', 'conditional_edges']:
                                if edge_key in node:
                                    if edge_key not in data:
                                        # Hoist to top level
                                        data[edge_key] = node[edge_key]
                                        logger.info(f"[SCHEMA_FIX] ✅ Hoisted '{edge_key}' to top level")
                                    else:
                                        # Top level already has this - merge intelligently
                                        if isinstance(data[edge_key], list) and isinstance(node[edge_key], list):
                                            data[edge_key].extend(node[edge_key])
                                            logger.info(f"[SCHEMA_FIX] ✅ Merged '{edge_key}' from malformed node")
                            nodes_to_remove.append(i)
                
                # Remove malformed "nodes" in reverse order to preserve indices
                for i in reversed(nodes_to_remove):
                    nodes.pop(i)
                    logger.info(f"[SCHEMA_FIX] 🗑️  Removed malformed node at index {i}")
                
                if nodes_to_remove:
                    logger.info(f"[SCHEMA_FIX] ✅ Fixed: {len(nodes)} valid nodes remaining after cleanup")
        return data
    
    @field_validator('edges', 'sequential_edges', 'parallel_edges', mode='before')
    @classmethod
    def normalize_edges(cls, v):
        """Normalize edges to list of dicts."""
        if not v:
            return []
        
        normalized = []
        for edge in v:
            if isinstance(edge, tuple) and len(edge) == 2:
                normalized.append({'source': edge[0], 'target': edge[1]})
            elif isinstance(edge, dict):
                normalized.append(edge)
            else:
                raise ValueError(f"Invalid edge format: {edge}. Expected tuple (source, target) or dict with 'source' and 'target' keys.")
        return normalized
    
    @model_validator(mode='after')
    def validate_edges_and_dependencies(self):
        """Validate that all edges reference existing nodes."""
        node_ids = {node.id for node in self.nodes}
        
        # Validate edges (check all edge lists: edges, sequential_edges, parallel_edges)
        for edge_list_name in ['edges', 'sequential_edges', 'parallel_edges']:
            edges = getattr(self, edge_list_name, []) or []
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source') or edge.get('from')
                    target = edge.get('target') or edge.get('to')
                    # Allow "start" as a special sentinel node for workflow entry point
                    if source and source != 'start' and source not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{source}'. Available nodes: {', '.join(sorted(node_ids))}")
                    if target and target not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{target}'. Available nodes: {', '.join(sorted(node_ids))}")
        
        return self
    
    @model_validator(mode='wrap')
    @classmethod
    def validate_tool_ids(cls, values, handler, info):
        """
        DETERMINISTIC VALIDATION: Ensure all tool_ids exist in _tool_schemas.
        
        This prevents LLM from inventing/hallucinating tool IDs (e.g., "FLIGHT_SEARCH" 
        instead of "COMPOSIO_SEARCH_FLIGHTS"). Triggers Instructor retry on failure.
        """
        # First, let Pydantic validate the model
        self = handler(values)
        
        # 🔧 CRITICAL FIX: Access _tool_schemas from validation_context (passed by Instructor)
        # or fall back to extra fields (for legacy validation)
        tool_schemas = None
        
        # Try validation_context first (Pydantic v2: context is in info.context)
        if info and info.context:
            tool_schemas = info.context.get('_tool_schemas')
            if tool_schemas:
                logger.info(f"[PYDANTIC_VALIDATOR] ✅ Using {len(tool_schemas)} tool schemas from validation_context")
        
        # Fall back to extra fields
        if not tool_schemas:
            tool_schemas = getattr(self, '_tool_schemas', None) or []
        
        if not tool_schemas:
            logger.debug("[PYDANTIC_VALIDATOR] No _tool_schemas found - skipping tool ID validation")
            return self
        
        # Build set of valid tool IDs from schemas
        valid_tool_ids = {schema['action_id'] for schema in tool_schemas if isinstance(schema, dict) and 'action_id' in schema}
        
        if not valid_tool_ids:
            logger.warning("[PYDANTIC_VALIDATOR] No valid tool IDs found in _tool_schemas")
            return self
        
        # Check each agent_with_tools node
        invalid_tools = []
        for node in self.nodes:
            if node.type == 'agent_with_tools' and node.tool_id:
                if node.tool_id not in valid_tool_ids:
                    # Find fuzzy matches for helpful error message
                    fuzzy_matches = []
                    node_tool_lower = node.tool_id.lower()
                    for valid_id in valid_tool_ids:
                        if node_tool_lower in valid_id.lower() or valid_id.lower() in node_tool_lower:
                            fuzzy_matches.append(valid_id)
                    
                    invalid_tools.append({
                        'node_id': node.id,
                        'invalid_tool_id': node.tool_id,
                        'suggestions': fuzzy_matches[:3]  # Top 3 suggestions
                    })
        
        if invalid_tools:
            # Build detailed error message for Instructor retry
            error_parts = [
                "❌ TOOL VALIDATION FAILED: Invalid tool_id(s) detected.",
                "",
                "🚨 CRITICAL: You MUST use EXACT tool IDs from the available actions list.",
                "DO NOT invent or simplify tool names.",
                "",
                "Invalid tools found:"
            ]
            for invalid in invalid_tools:
                error_parts.append(f"  • Node '{invalid['node_id']}': '{invalid['invalid_tool_id']}' does NOT exist")
                if invalid['suggestions']:
                    error_parts.append(f"    Did you mean: {', '.join(invalid['suggestions'])}?")
                else:
                    error_parts.append(f"    Available tools: {', '.join(sorted(valid_tool_ids)[:5])}...")
            
            error_parts.append("")
            error_parts.append("✅ FIX: Replace invalid tool_id(s) with exact matches from the available actions.")
            
            raise ValueError("\n".join(error_parts))
        
        logger.debug(f"[PYDANTIC_VALIDATOR] ✅ Tool ID validation passed: {len([n for n in self.nodes if n.type == 'agent_with_tools'])} nodes checked")
        return self
    
    @model_validator(mode='wrap')
    @classmethod
    def validate_important_optional_params(cls, values, handler, info):
        """
        DETERMINISTIC VALIDATION: Ensure contextually important optional parameters are included.
        
        Rule: If a parameter has 'examples' in schema AND 'default' is null/nullable, 
        it's contextually important (e.g., email subject, task title, etc.).
        
        This catches when LLM omits important params. Triggers Instructor retry.
        """
        # First, let Pydantic validate the model
        self = handler(values)
        
        # 🔧 CRITICAL FIX: Access _tool_schemas from validation_context (passed by Instructor)
        # or fall back to extra fields (for legacy validation)
        tool_schemas = None
        
        # Try validation_context first (Pydantic v2: context is in info.context)
        if info and info.context:
            tool_schemas = info.context.get('_tool_schemas')
            if tool_schemas:
                logger.info(f"[PYDANTIC_VALIDATOR] ✅ Using {len(tool_schemas)} tool schemas from validation_context for param check")
        
        # Fall back to extra fields
        if not tool_schemas:
            tool_schemas = getattr(self, '_tool_schemas', None) or []
        
        if not tool_schemas:
            logger.debug("[PYDANTIC_VALIDATOR] No _tool_schemas found - skipping important param validation")
            return self
        
        # Build tool_id → schema mapping
        tool_id_to_schema = {schema['action_id']: schema for schema in tool_schemas if isinstance(schema, dict) and 'action_id' in schema}
        
        # Check each agent_with_tools node
        missing_params = []
        for node in self.nodes:
            if node.type != 'agent_with_tools' or not node.tool_id:
                continue
            
            if node.tool_id not in tool_id_to_schema:
                continue  # Tool validation will catch this
            
            tool_schema = tool_id_to_schema[node.tool_id]
            params_schema = tool_schema.get('parameters', {}).get('properties', {})
            required_params = tool_schema.get('parameters', {}).get('required', [])
            
            for param_name, param_props in params_schema.items():
                # Skip if required (required params are handled elsewhere)
                if param_name in required_params:
                    continue
                
                # Check if contextually important:
                # has examples + null default = contextually important
                has_examples = bool(param_props.get('examples'))
                has_null_default = param_props.get('default') is None
                
                # ✅ SCHEMA-DRIVEN: No hardcoding - just check schema signals
                # If Composio added examples AND default is null, it's contextually important
                if has_examples and has_null_default:
                    # This is a contextually important optional parameter
                    # Check if LLM included it in node params
                    node_params = node.params or {}
                    if param_name not in node_params:
                        missing_params.append({
                            'node_id': node.id,
                            'tool_id': node.tool_id,
                            'param_name': param_name,
                            'examples': param_props.get('examples', [])[:2]  # Show first 2 examples
                        })
        
        if missing_params:
            # 🎯 NON-BLOCKING: Log warning only, don't block workflow generation
            # Optional parameters can be omitted if not needed for the user's use case
            warning_parts = [
                f"⚠️ [PARAM_VALIDATION] Found {len(missing_params)} optional parameter(s) with examples that were omitted:",
            ]
            for missing in missing_params[:3]:  # Show first 3
                examples_str = ', '.join([f"'{ex}'" for ex in missing['examples']])
                warning_parts.append(f"  • Node '{missing['node_id']}' ({missing['tool_id']}): '{missing['param_name']}' (examples: {examples_str})")
            
            if len(missing_params) > 3:
                warning_parts.append(f"  ... and {len(missing_params) - 3} more")
            
            warning_parts.append("💡 These parameters have examples in the schema but were omitted. They may improve UX if applicable.")
            
            logger.warning("\n".join(warning_parts))
        
        logger.debug(f"[PYDANTIC_VALIDATOR] ✅ Important param validation passed: {len([n for n in self.nodes if n.type == 'agent_with_tools'])} nodes checked")
        return self
    
    def to_ir(self):
        """
        Convert WorkflowDefinition to WorkflowIR for DAG validation.
        
        Returns:
            WorkflowIR instance with cycle detection, edge inference, and topological ordering
        """
        from taskweaver.code_interpreter.workflow_ir import WorkflowIR
        
        # Convert Pydantic model to dict for WorkflowIR
        workflow_dict = {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'tool_id': node.tool_id,
                    'app_name': node.app_name,
                    'params': node.params if isinstance(node.params, dict) else node.params.dict(),
                    'description': node.description,
                    'code': node.code,  # For code_execution nodes
                    'decision': node.decision,
                    'blocking': node.blocking,
                    'form_schema': node.form_schema,
                    'metadata': node.metadata,
                    # Loop-specific fields (preserved from extra fields via Pydantic Config)
                    'loop_body': getattr(node, 'loop_body', None),
                    'loop_over': getattr(node, 'loop_over', None),
                    'nodes': getattr(node, 'nodes', None),  # For parallel groups
                }
                for node in self.nodes
            ],
            'edges': self.edges or [],  # Primary edges array from function calling
            'sequential_edges': self.sequential_edges or [],  # Legacy support
            'parallel_edges': self.parallel_edges or [],
        }
        
        # Copy over any extra fields from original dict
        for key, value in self.__dict__.items():
            if key not in ['nodes', 'edges', 'sequential_edges', 'parallel_edges'] and not key.startswith('_'):
                workflow_dict[key] = value
        
        return WorkflowIR(workflow_dict)


def validate_workflow_dict(workflow_dict: Dict[str, Any]) -> tuple[bool, Optional[WorkflowDefinition], List[str]]:
    """
    Validate WORKFLOW dict using Pydantic + WorkflowIR.
    
    Two-phase validation:
    1. Pydantic: Schema validation (types, required fields, basic structure)
    2. WorkflowIR: DAG validation (cycles, connectivity, edge inference, data flow)
    
    Args:
        workflow_dict: The WORKFLOW dictionary to validate
    
    Returns:
        Tuple of (is_valid, workflow_obj, error_messages)
        - is_valid: True if validation passed
        - workflow_obj: Validated WorkflowDefinition object (None if invalid)
        - error_messages: List of human-readable error messages
    """
    error_messages = []
    
    # Phase 1: Pydantic validation (schema)
    try:
        workflow = WorkflowDefinition(**workflow_dict)
    except Exception as e:
        if hasattr(e, 'errors'):
            # Pydantic ValidationError
            for error in e.errors():
                loc = " -> ".join(str(l) for l in error['loc'])
                msg = error['msg']
                error_type = error.get('type', '')
                
                # Check if this is an invalid node type error
                if 'type' in loc and 'literal_error' in error_type:
                    # Get the actual value that was provided
                    try:
                        # Navigate to the error location to get actual value
                        value = workflow_dict
                        for key in error['loc']:
                            if isinstance(key, int):
                                value = value[key]
                            else:
                                value = value.get(key, {})
                        actual_value = value if isinstance(value, str) else str(value)
                        
                        # Specific error for invalid node type containing dots
                        if '.' in actual_value or 'tool_use' in actual_value:
                            error_messages.append(
                                f"[!] INVALID NODE TYPE at {loc}: '{actual_value}' is not valid.\n"
                                f"    ℹ️  Valid types: agent_with_tools, agent_only, code_execution, form, hitl, loop\n"
                                f"    💡 For parallel execution: Use dependencies field (nodes with same dependencies run in parallel)"
                            )
                            continue
                    except:
                        pass
                
                # Make errors more actionable
                if 'missing' in msg.lower():
                    # 🔧 FIX: Show both location AND message for missing fields
                    error_messages.append(f"[!] Missing required field at {loc}: {msg}")
                elif 'value_error' in error_type:
                    error_messages.append(f"[!] Invalid value at {loc}: {msg}")
                else:
                    error_messages.append(f"[!] {loc}: {msg}")
        else:
            error_messages.append(f"[!] Validation error: {str(e)}")
        
        return False, None, error_messages
    
    # Phase 2: WorkflowIR validation (DAG logic)
    try:
        logger.info(f"[WORKFLOW_SCHEMA] Starting WorkflowIR validation for {len(workflow.nodes)} nodes...")
        workflow_ir = workflow.to_ir()
        edge_count = len(workflow_ir.edges) if hasattr(workflow_ir, 'edges') else 0
        logger.info(f"[WORKFLOW_SCHEMA] WorkflowIR validation PASSED: {len(workflow.nodes)} nodes, {edge_count} edges")
        
        # ✅ SINGLE SOURCE OF TRUTH: Export WorkflowIR's complete edges back to workflow dict
        # This includes auto-added parallel edges, so downstream consumers get the full edge list
        workflow_dict['edges'] = [
            {
                'from': edge.source,
                'to': edge.target,
                'type': edge.type.name.lower() if hasattr(edge.type, 'name') else str(edge.type)
            }
            for edge in workflow_ir.edges
        ]
        logger.info(f"[WORKFLOW_SCHEMA] ✅ Exported {len(workflow_dict['edges'])} complete edges from WorkflowIR")
        
    except ValueError as e:
        # WorkflowIR raises ValueError for cycles or invalid DAG structure
        logger.error(f"[WORKFLOW_SCHEMA] WorkflowIR DAG validation FAILED: {str(e)}")
        error_messages.append(f"[!] Workflow DAG validation failed: {str(e)}")
        return False, None, error_messages
    except Exception as e:
        # Unexpected errors during IR conversion
        logger.error(f"[WORKFLOW_SCHEMA] WorkflowIR conversion ERROR: {str(e)}")
        error_messages.append(f"[!] WorkflowIR conversion failed: {str(e)}")
        logger.error(f"WorkflowIR conversion error: {e}", exc_info=True)
        return False, None, error_messages
    
    return True, workflow, []


def format_workflow_validation_error(errors: List[str], code: str) -> str:
    """
    Format Pydantic validation errors for LLM consumption.
    
    Returns clear, actionable error messages with context.
    """
    error_text = "\n".join(errors)
    
    # Extract just the WORKFLOW dict for context (first 500 chars)
    code_preview = code[:500] if len(code) > 500 else code
    if len(code) > 500:
        code_preview += "..."
    
    return f"""[!] WORKFLOW Structure Validation Failed

{error_text}

**Common Fixes:**
1. **Bracket matching**: Check for ]{{}}}} (wrong) vs ]{{}}, (correct)
2. **Node references**: All node IDs in edges/depends_on must exist in nodes list
3. **Tool requirements**: agent_with_tools nodes MUST have tool_id field
4. **HITL requirements**: Blocking HITL nodes MUST have decision field
5. **Edge format**: Use tuples like ("source", "target") or dicts with 'source' and 'target' keys

**Your WORKFLOW dict (preview):**
```python
{code_preview}
```

**Action:** Fix the specific errors listed above and regenerate the WORKFLOW dict with correct structure.
"""
