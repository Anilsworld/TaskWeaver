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
        "code_execution"
    ] = Field(..., description="Node execution type")
    
    # Optional fields
    tool_id: Optional[str] = Field(None, description="Composio tool ID")
    app_name: Optional[str] = Field(None, description="App name (gmail, slack, etc)")
    params: Union[Dict[str, Any], WorkflowParams] = Field(default_factory=dict, description="Tool parameters")
    description: Optional[str] = Field(None, description="Node description")
    depends_on: List[str] = Field(default_factory=list, description="Dependencies")
    
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
    
    @model_validator(mode='before')
    @classmethod
    def move_tool_params_to_params_dict(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        AUTO-HEALING: Move top-level tool params into 'params' dict.
        
        LLM often puts tool params at top level (adults, arrival_id, etc.) 
        despite schema saying additionalProperties: false.
        
        This validator moves them to the correct location BEFORE validation.
        """
        if not isinstance(values, dict):
            return values
        
        # Define structural fields that should stay at top level
        structural_keys = {
            'id', 'type', 'tool_id', 'params', 'depends_on', 'description',
            'app_name', 'parallel_group', 'decision', 'blocking', 'form_schema',
            'metadata', 'code', 'fields', 'approval_type', 'loop_over', 'iterate_over',
            'loop_body', 'nodes', 'max_iterations', 'workflow_id', 'inputs'
        }
        
        # Find non-structural keys (likely tool params)
        top_level_params = {k: v for k, v in values.items() if k not in structural_keys}
        
        if top_level_params:
            # Ensure params dict exists
            if 'params' not in values:
                values['params'] = {}
            elif not isinstance(values['params'], dict):
                values['params'] = {}
            
            # Move top-level params into params dict
            moved_keys = []
            for key, value in top_level_params.items():
                if key not in values['params']:  # Don't overwrite existing params
                    values['params'][key] = value
                    moved_keys.append(key)
            
            # Remove from top level
            for key in moved_keys:
                del values[key]
            
            logger.info(
                f"[PYDANTIC_HEAL] Node '{values.get('id')}': Moved {len(moved_keys)} param(s) "
                f"from top-level to params dict: {moved_keys}"
            )
        
        return values
    
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
    sequential_edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="Sequential edges between nodes"
    )
    parallel_edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="Parallel edges (for parallel execution groups)"
    )
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"
    
    @field_validator('sequential_edges', 'parallel_edges', mode='before')
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
        """Validate that all edges and dependencies reference existing nodes."""
        node_ids = {node.id for node in self.nodes}
        
        # Validate edges
        for edge_list_name in ['sequential_edges', 'parallel_edges']:
            edges = getattr(self, edge_list_name, []) or []
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source')
                    target = edge.get('target')
                    if source not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{source}'. Available nodes: {', '.join(sorted(node_ids))}")
                    if target not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{target}'. Available nodes: {', '.join(sorted(node_ids))}")
        
        # Validate dependencies
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in node_ids:
                    raise ValueError(
                        f"Node '{node.id}' depends on non-existent node: '{dep}'. Available nodes: {', '.join(sorted(node_ids))}"
                    )
        
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
                    'depends_on': node.depends_on,
                    'parallel_group': node.parallel_group,
                    'decision': node.decision,
                    'blocking': node.blocking,
                    'form_schema': node.form_schema,
                    'metadata': node.metadata,
                }
                for node in self.nodes
            ],
            'sequential_edges': self.sequential_edges or [],
            'parallel_edges': self.parallel_edges or [],
        }
        
        # Copy over any extra fields from original dict
        for key, value in self.__dict__.items():
            if key not in ['nodes', 'sequential_edges', 'parallel_edges'] and not key.startswith('_'):
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
