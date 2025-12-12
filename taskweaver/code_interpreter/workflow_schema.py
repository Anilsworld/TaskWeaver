"""
Pydantic-based Workflow Validation
===================================
Fully scalable, adaptive schema validation for WORKFLOW dicts.
No hardcoding - just update the schema when requirements change.
"""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
import json


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
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"
    
    @validator('tool_id', always=True)
    def validate_tool_id(cls, v, values):
        """Validate that agent_with_tools nodes have tool_id."""
        node_type = values.get('type')
        if node_type == 'agent_with_tools' and not v:
            raise ValueError(f"agent_with_tools nodes must have tool_id")
        return v
    
    @validator('decision', always=True)
    def validate_hitl_decision(cls, v, values):
        """Validate that HITL nodes have decision field."""
        node_type = values.get('type')
        if node_type == 'hitl' and values.get('blocking') and not v:
            # Only require decision for blocking HITL nodes
            raise ValueError(f"Blocking HITL nodes must have 'decision' field for conditional routing")
        return v


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
    
    @validator('sequential_edges', 'parallel_edges', pre=True)
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
    
    @root_validator
    def validate_edges(cls, values):
        """Validate that all edges reference existing nodes."""
        nodes = values.get('nodes', [])
        node_ids = {node.id for node in nodes}
        
        for edge_list_name in ['sequential_edges', 'parallel_edges']:
            edges = values.get(edge_list_name, [])
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get('source')
                    target = edge.get('target')
                    if source not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{source}'. Available nodes: {', '.join(sorted(node_ids))}")
                    if target not in node_ids:
                        raise ValueError(f"Edge references non-existent node: '{target}'. Available nodes: {', '.join(sorted(node_ids))}")
        
        return values
    
    @root_validator
    def validate_dependencies(cls, values):
        """Validate that all depends_on reference existing nodes."""
        nodes = values.get('nodes', [])
        node_ids = {node.id for node in nodes}
        
        for node in nodes:
            for dep in node.depends_on:
                if dep not in node_ids:
                    raise ValueError(
                        f"Node '{node.id}' depends on non-existent node: '{dep}'. Available nodes: {', '.join(sorted(node_ids))}"
                    )
        
        return values


def validate_workflow_dict(workflow_dict: Dict[str, Any]) -> tuple[bool, Optional[WorkflowDefinition], List[str]]:
    """
    Validate WORKFLOW dict using Pydantic.
    
    Args:
        workflow_dict: The WORKFLOW dictionary to validate
    
    Returns:
        Tuple of (is_valid, workflow_obj, error_messages)
        - is_valid: True if validation passed
        - workflow_obj: Validated WorkflowDefinition object (None if invalid)
        - error_messages: List of human-readable error messages
    """
    try:
        workflow = WorkflowDefinition(**workflow_dict)
        return True, workflow, []
    except Exception as e:
        # Parse Pydantic validation errors into readable messages
        error_messages = []
        
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
            # Other errors (e.g., syntax errors caught before Pydantic)
            error_messages.append(f"[!] Validation error: {str(e)}")
        
        return False, None, error_messages


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
1. **Bracket matching**: Check for ]}} (wrong) vs ]}, (correct)
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
