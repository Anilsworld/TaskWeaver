"""
Pydantic-based Workflow Validation
===================================
Fully scalable, adaptive schema validation for WORKFLOW dicts.
No hardcoding - just update the schema when requirements change.

Integration with WorkflowIR:
- Pydantic validates schema (types, required fields)
- WorkflowIR validates DAG logic (cycles, connectivity, data flow)
- Placeholder validator validates reference resolution
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
        "parallel"
    ] = Field(..., description="Node execution type")
    
    # Optional fields
    tool_id: Optional[str] = Field(None, description="Composio tool ID")
    app_name: Optional[str] = Field(None, description="App name (gmail, slack, etc)")
    params: Union[Dict[str, Any], WorkflowParams] = Field(default_factory=dict, description="Tool parameters")
    description: Optional[str] = Field(None, description="Node description")
    
    # Parallel execution field
    parallel_nodes: Optional[List[str]] = Field(None, description="Node IDs to execute in parallel (for parallel type)")
    
    # Advanced fields
    parallel_group: Optional[int] = Field(None, description="Parallel execution group")
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
    """Complete workflow structure."""
    nodes: List[WorkflowNode] = Field(..., min_items=1, description="Workflow nodes")
    edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="Edges connecting workflow nodes"
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
                    if source and source not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{source}'. Available nodes: {', '.join(sorted(node_ids))}")
                    if target and target not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{target}'. Available nodes: {', '.join(sorted(node_ids))}")
        
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
                    'parallel_group': node.parallel_group,
                    'parallel_nodes': node.parallel_nodes,  # ✅ CRITICAL: Pass parallel_nodes to WorkflowIR!
                    'code': node.code,  # ✅ For code_execution nodes
                    'decision': node.decision,
                    'blocking': node.blocking,
                    'form_schema': node.form_schema,
                    'metadata': node.metadata,
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


def validate_workflow_dict(
    workflow_dict: Dict[str, Any],
    tool_cache: Optional[Dict[str, Any]] = None
) -> tuple[bool, Optional[WorkflowDefinition], List[str]]:
    """
    Validate WORKFLOW dict using three-phase validation.
    
    Three-phase validation:
    1. Pydantic: Schema validation (types, required fields, basic structure)
    2. WorkflowIR: DAG validation (cycles, connectivity, edge inference, data flow)
    3. Placeholder: Reference validation (node existence, field availability)
    
    SCALABLE: No domain knowledge, no keyword matching, no hardcoded tools.
    Works for ANY workflow, ANY tools, ANY complexity.
    
    Args:
        workflow_dict: The WORKFLOW dictionary to validate
        tool_cache: Optional tool schema cache for deep placeholder validation
    
    Returns:
        Tuple of (is_valid, workflow_obj, error_messages)
        - is_valid: True if validation passed
        - workflow_obj: Validated WorkflowDefinition object (None if invalid)
        - error_messages: List of human-readable error messages
    """
    error_messages = []
    
    # Phase 1: Pydantic validation (schema only - no semantic checking)
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
                                f"    ✅ CORRECT: Use type='parallel' with parallel_nodes=['child1', 'child2', ...]\n"
                                f"    📚 See example: Check stock prices from 3 financial APIs\n"
                                f"    ℹ️  Valid types: agent_with_tools, code_execution, form, hitl, parallel"
                            )
                            continue
                    except:
                        pass
                
                # Make errors more actionable
                if 'missing' in msg.lower():
                    error_messages.append(f"[!] Missing required field: {loc}")
                elif 'value_error' in error_type:
                    error_messages.append(f"[!] Invalid value at {loc}: {msg}")
                else:
                    error_messages.append(f"[!] {loc}: {msg}")
        else:
            error_messages.append(f"[!] Validation error: {str(e)}")
        
        return False, None, error_messages
    
    # Phase 2: WorkflowIR validation (DAG logic)
    try:
        print(f"[WORKFLOW_SCHEMA] 🔍 Starting WorkflowIR validation for {len(workflow.nodes)} nodes...")
        workflow_ir = workflow.to_ir()
        edge_count = len(workflow_ir.edges) if hasattr(workflow_ir, 'edges') else 0
        print(f"[WORKFLOW_SCHEMA] ✅ WorkflowIR validation PASSED: {len(workflow.nodes)} nodes, {edge_count} edges")
        logger.info(f"✅ WorkflowIR validation passed: {len(workflow.nodes)} nodes, {edge_count} edges")
        
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
        print(f"[WORKFLOW_SCHEMA] ❌ WorkflowIR DAG validation FAILED: {str(e)}")
        error_messages.append(f"[!] Workflow DAG validation failed: {str(e)}")
        return False, None, error_messages
    except Exception as e:
        # Unexpected errors during IR conversion
        print(f"[WORKFLOW_SCHEMA] ❌ WorkflowIR conversion ERROR: {str(e)}")
        error_messages.append(f"[!] WorkflowIR conversion failed: {str(e)}")
        logger.error(f"WorkflowIR conversion error: {e}", exc_info=True)
        return False, None, error_messages
    
    # Phase 3: Placeholder validation (reference resolution)
    try:
        from taskweaver.code_interpreter.workflow_placeholder_validator import validate_workflow_placeholders
        
        print(f"[WORKFLOW_SCHEMA] 🔍 Starting placeholder validation...")
        placeholder_result = validate_workflow_placeholders(workflow_dict, tool_cache=tool_cache)
        
        if not placeholder_result.valid:
            print(f"[WORKFLOW_SCHEMA] ❌ Placeholder validation FAILED: {len(placeholder_result.errors)} errors")
            logger.warning(f"❌ Placeholder validation failed: {len(placeholder_result.errors)} errors")
            error_messages.extend(placeholder_result.errors)
            return False, None, error_messages
        
        # Log warnings (non-blocking)
        if placeholder_result.warnings:
            print(f"[WORKFLOW_SCHEMA] ⚠️  Placeholder validation: {len(placeholder_result.warnings)} warnings")
            for warning in placeholder_result.warnings[:3]:  # Show first 3
                logger.warning(f"  {warning}")
        
        print(f"[WORKFLOW_SCHEMA] ✅ Placeholder validation PASSED: {placeholder_result.validated_references} validated, {placeholder_result.skipped_references} skipped")
        logger.info(
            f"✅ Placeholder validation passed: "
            f"{placeholder_result.validated_references} validated, "
            f"{placeholder_result.skipped_references} skipped"
        )
        
    except Exception as e:
        # Placeholder validation is a best-effort check - don't fail the entire workflow if it errors
        print(f"[WORKFLOW_SCHEMA] ⚠️  Placeholder validation ERROR (non-blocking): {str(e)}")
        logger.warning(f"Placeholder validation error (non-blocking): {e}")
    
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
2. **Node references**: All node IDs in edges must exist in nodes list
3. **Tool requirements**: agent_with_tools nodes MUST have tool_id field
4. **HITL requirements**: Blocking HITL nodes MUST have decision field
5. **Edge format**: Use tuples like ("source", "target") or dicts with 'source' and 'target' keys

**Your WORKFLOW dict (preview):**
```python
{code_preview}
```

**Action:** Fix the specific errors listed above and regenerate the WORKFLOW dict with correct structure.
"""
