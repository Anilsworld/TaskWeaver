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
        """
        ✅ SHIFT-LEFT VALIDATION: Enforce rules DURING generation.
        
        Instructor will automatically retry with detailed error feedback if validation fails.
        This prevents errors instead of patching them later!
        """
        # Validate tool_id for agent_with_tools
        if self.type == 'agent_with_tools' and not self.tool_id:
            raise ValueError(
                f"❌ Node '{self.id}': agent_with_tools nodes MUST have tool_id. "
                f"Select a valid tool from the available tools list."
            )
        
        # Validate decision for blocking HITL nodes
        if self.type == 'hitl' and self.blocking and not self.decision:
            raise ValueError(
                f"❌ Node '{self.id}': Blocking HITL nodes MUST have 'decision' field for conditional routing"
            )
        
        # ✨ NEW: Validate code_execution nodes have 'result =' in their code
        if self.type == 'code_execution':
            if not self.code or not self.code.strip():
                raise ValueError(
                    f"❌ Node '{self.id}': code_execution nodes MUST have non-empty 'code' field. "
                    f"Provide Python code that assigns final output to 'result' variable."
                )
            
            if 'result' not in self.code:
                raise ValueError(
                    f"❌ Node '{self.id}': code_execution code MUST assign to 'result' variable. "
                    f"Example: result = 'output value'\n"
                    f"Current code does not mention 'result' at all."
                )
            
            if 'result =' not in self.code and 'result=' not in self.code:
                raise ValueError(
                    f"❌ Node '{self.id}': code_execution code mentions 'result' but doesn't assign to it. "
                    f"Use 'result = <value>' to assign the final output.\n"
                    f"Example: result = formatted_text"
                )
        
        # ✨ NEW: Auto-extract app_name from tool_id if missing
        if self.type == 'agent_with_tools' and self.tool_id and not self.app_name:
            if '_' in self.tool_id:
                parts = self.tool_id.split('_')
                if parts[0] == 'COMPOSIO' and len(parts) > 1:
                    self.app_name = f"{parts[0].lower()}_{parts[1].lower()}"
                else:
                    self.app_name = parts[0].lower()
                logger.info(f"[PYDANTIC_HEAL] Auto-extracted app_name='{self.app_name}' from tool_id='{self.tool_id}'")
        
        return self


class WorkflowDefinition(BaseModel):
    """Complete workflow structure with comprehensive validation."""
    nodes: List[WorkflowNode] = Field(..., min_items=1, description="Workflow nodes")
    sequential_edges: Optional[List[Union[tuple, dict]]] = Field(
        default_factory=list,
        description="Sequential edges between nodes as [(source, target), ...] or [{'source': ..., 'target': ...}, ...]"
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
        """
        ✅ SHIFT-LEFT VALIDATION: Comprehensive validation DURING generation.
        
        Validates:
        1. Edge references point to existing nodes
        2. Dependencies point to existing nodes  
        3. Placeholder references are correct
        4. Code execution nodes are properly referenced
        
        Instructor will retry with detailed feedback if validation fails.
        """
        node_ids = {node.id for node in self.nodes}
        code_execution_nodes = {node.id for node in self.nodes if node.type == 'code_execution'}
        
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
        
        # ✨ NEW: Validate placeholder references
        errors = self._validate_all_placeholders(node_ids, code_execution_nodes)
        if errors:
            raise ValueError(
                f"❌ Placeholder validation failed:\n" + 
                "\n".join(f"  - {error}" for error in errors)
            )
        
        return self
    
    def _validate_all_placeholders(
        self,
        node_ids: set,
        code_execution_nodes: set
    ) -> List[str]:
        """
        Helper method to validate all placeholders in the workflow.
        
        Returns list of error messages (empty if no errors).
        """
        import re
        errors = []
        
        for node in self.nodes:
            # Check code field
            if node.code:
                errors.extend(self._validate_placeholders_in_text(
                    text=node.code,
                    node_id=node.id,
                    field_name='code',
                    node_ids=node_ids,
                    code_execution_nodes=code_execution_nodes
                ))
            
            # Check params
            if isinstance(node.params, dict):
                for param_key, param_value in node.params.items():
                    if isinstance(param_value, str):
                        errors.extend(self._validate_placeholders_in_text(
                            text=param_value,
                            node_id=node.id,
                            field_name=f'params.{param_key}',
                            node_ids=node_ids,
                            code_execution_nodes=code_execution_nodes
                        ))
        
        return errors
    
    @staticmethod
    def _validate_placeholders_in_text(
        text: str,
        node_id: str,
        field_name: str,
        node_ids: set,
        code_execution_nodes: set
    ) -> List[str]:
        """
        Validate placeholders in a text field.
        
        Returns list of error messages (empty if no errors).
        """
        import re
        errors = []
        
        # Pattern: ${node_id} or ${node_id.field}
        placeholder_pattern = r'\$\{([^}]+)\}'
        placeholders = re.findall(placeholder_pattern, text)
        
        for placeholder in placeholders:
            # Generic .response_field check
            if '.response_field' in placeholder:
                errors.append(
                    f"Node '{node_id}' {field_name}: Uses generic '.response_field' in ${{{placeholder}}}. "
                    f"Use actual field names from tool schemas (e.g., .data.results, .id, etc.)"
                )
                continue
            
            # Extract referenced node ID (before first dot)
            ref_parts = placeholder.split('.', 1)
            ref_node_id = ref_parts[0]
            
            # Skip system placeholders
            if ref_node_id in {'user_input', 'env', 'from_step', 'from_loop'}:
                continue
            
            # Check if referenced node exists
            if ref_node_id not in node_ids:
                errors.append(
                    f"Node '{node_id}' {field_name}: References non-existent node '{ref_node_id}' in ${{{placeholder}}}. "
                    f"Available nodes: {', '.join(sorted(node_ids))}"
                )
                continue
            
            # Check if referencing code_execution node without .execution_result
            if ref_node_id in code_execution_nodes:
                if len(ref_parts) == 1:
                    # ${code_node} with no field access
                    errors.append(
                        f"Node '{node_id}' {field_name}: References code_execution node '{ref_node_id}' without '.execution_result'. "
                        f"Use ${{{ref_node_id}.execution_result}} for primitive results or ${{{ref_node_id}.execution_result.field_name}} for dict results."
                    )
                elif not ref_parts[1].startswith('execution_result'):
                    # ${code_node.something} where something is not execution_result
                    errors.append(
                        f"Node '{node_id}' {field_name}: References code_execution node '{ref_node_id}' with invalid field '.{ref_parts[1]}'. "
                        f"Code execution outputs must be accessed via '.execution_result' or '.execution_result.field_name'."
                    )
        
        return errors
    
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
