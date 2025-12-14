"""
Pydantic-Based Placeholder Reference Validator

Validates that ${{from_step:source.field}} placeholders reference valid fields
in source nodes. Uses Pydantic for type-safe, declarative validation.

Only validates placeholders referencing form/HITL nodes (known schemas).
Tool output references are not validated since response schemas are often incomplete.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class PlaceholderReference(BaseModel):
    """A single placeholder reference extracted from a parameter value."""
    source_node_id: str
    field_name: str
    target_node_id: str  # Node that contains this reference
    param_path: str  # Path to the parameter (e.g., "params.recipient_email")
    full_placeholder: str  # Original placeholder text
    array_access: Optional[str] = None  # e.g., "messages[0]" if array indexing detected
    sub_field_path: Optional[str] = None  # Nested field path after array (e.g., ".subject", ".payload.body")


class NodeSchema(BaseModel):
    """Schema of a workflow node's available output fields."""
    node_id: str
    node_type: str
    fields: Optional[Set[str]] = None  # None = unknown schema (tool output)
    tool_id: Optional[str] = None  # Tool ID for agent_with_tools nodes
    tool_cache_ref: Optional[Dict[str, Any]] = None  # Reference to full tool cache for deep validation
    
    @property
    def has_known_schema(self) -> bool:
        """Whether this node has a known output schema."""
        return self.fields is not None
    
    def get_nested_fields(self, field_name: str) -> Optional[Set[str]]:
        """
        Get available fields for nested access (e.g., messages[0].field).
        
        SCALABLE: Works for ANY array field in ANY tool schema!
        
        Args:
            field_name: Base field name (e.g., "messages")
            
        Returns:
            Set of available sub-fields if schema available, None otherwise
        """
        if not self.tool_id or not self.tool_cache_ref or self.node_type != 'agent_with_tools':
            return None
        
        try:
            tool_schema = self.tool_cache_ref.get(self.tool_id, {})
            response_schema = tool_schema.get('response_schema', {})
            properties = response_schema.get('properties', {})
            
            # Navigate through Composio's data wrapper if present
            target_props = properties
            if 'data' in properties:
                data_schema = properties['data']
                if 'properties' in data_schema:
                    target_props = data_schema['properties']
            
            # Get the field schema (e.g., "messages")
            field_schema = target_props.get(field_name, {})
            
            # If it's an array, get the items schema
            if field_schema.get('type') == 'array' and 'items' in field_schema:
                items_schema = field_schema['items']
                if 'properties' in items_schema:
                    # Return all available sub-fields
                    return set(items_schema['properties'].keys())
            
            return None
        except Exception as e:
            logger.debug(f"Failed to get nested fields for {field_name}: {e}")
            return None
    
    def validate_nested_path(self, field_name: str, nested_path: str) -> Tuple[bool, Optional[str], Optional[Set[str]]]:
        """
        🚀 FULLY DYNAMIC: Recursively validate entire nested path against schema.
        
        SCALABLE: Works for ANY nested path in ANY tool schema!
        No hardcoded patterns - pure schema traversal.
        
        Args:
            field_name: Base field name (e.g., "messages")
            nested_path: Full nested path (e.g., ".payload.parts[0].body.data")
            
        Returns:
            (is_valid, error_field, available_fields)
            - is_valid: True if entire path is valid
            - error_field: The field that doesn't exist (if invalid)
            - available_fields: Available fields at the error point
        """
        if not self.tool_id or not self.tool_cache_ref or self.node_type != 'agent_with_tools':
            return (True, None, None)  # Can't validate without schema
        
        try:
            tool_schema = self.tool_cache_ref.get(self.tool_id, {})
            response_schema = tool_schema.get('response_schema', {})
            properties = response_schema.get('properties', {})
            
            # Navigate through Composio's data wrapper if present
            target_props = properties
            if 'data' in properties:
                data_schema = properties['data']
                if 'properties' in data_schema:
                    target_props = data_schema['properties']
            
            # Start with base field (e.g., "messages")
            field_schema = target_props.get(field_name, {})
            
            # If it's an array, get the items schema
            if field_schema.get('type') == 'array' and 'items' in field_schema:
                field_schema = field_schema['items']
            
            # Parse nested path (e.g., ".payload.parts[0].body.data")
            # Remove leading dot, split by dots, handle array notation
            path_parts = nested_path.lstrip('.').split('.')
            current_schema = field_schema
            
            for i, part in enumerate(path_parts):
                # Remove array notation (e.g., "parts[0]" → "parts")
                part_name = part.split('[')[0] if '[' in part else part
                
                if not part_name:  # Skip empty parts
                    continue
                
                # Get properties at current level
                current_props = current_schema.get('properties', {})
                if not current_props:
                    # No schema available at this level - can't validate further
                    return (True, None, None)
                
                # Check if field exists
                if part_name not in current_props:
                    # Field doesn't exist! Return available fields at this level
                    available = set(current_props.keys())
                    failed_path = '.'.join(path_parts[:i+1])
                    return (False, part_name, available)
                
                # Move to next level
                current_schema = current_props[part_name]
                
                # If it's an array, descend into items
                if current_schema.get('type') == 'array' and 'items' in current_schema:
                    current_schema = current_schema['items']
            
            # All parts validated successfully
            return (True, None, None)
            
        except Exception as e:
            logger.debug(f"Schema traversal failed for {field_name}{nested_path}: {e}")
            return (True, None, None)  # Can't validate, allow it (avoid false positives)


class PlaceholderValidationResult(BaseModel):
    """Result of placeholder validation."""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    total_references: int = 0
    validated_references: int = 0
    skipped_references: int = 0  # Tool outputs with unknown schemas


class WorkflowPlaceholderValidator(BaseModel):
    """
    Validates placeholder references in workflow nodes.
    
    Features:
    - Pydantic-based declarative validation
    - Type-safe schema extraction
    - Only validates known schemas (forms/HITL)
    - Auto-extracts references using regex
    - Validates placeholder syntax (single vs double braces)
    - Provides detailed error messages
    """
    nodes: List[Dict[str, Any]]
    tool_cache: Optional[Dict[str, Any]] = None
    
    # Computed fields
    _node_schemas: Dict[str, NodeSchema] = {}
    _placeholder_refs: List[PlaceholderReference] = []
    _syntax_errors: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True
    
    @model_validator(mode='after')
    def extract_node_schemas(self):
        """Extract output schemas from all nodes."""
        schemas = {}
        
        for node in self.nodes:
            node_id = node.get('id')
            if not node_id:
                continue
            
            node_type = node.get('type') or node.get('node_type', '')
            fields = None  # None = unknown schema
            
            # Form nodes: Extract field names from schema
            if node_type in ('form', 'form_node'):
                form_fields = node.get('fields', []) or node.get('form_schema', [])
                fields = set(f.get('name') for f in form_fields if f.get('name'))
            
            # HITL nodes: Extract field names + standard outputs
            elif node_type in ('hitl', 'hitl_form'):
                form_fields = node.get('fields', []) or node.get('form_schema', [])
                field_names = [f.get('name') for f in form_fields if f.get('name')]
                field_names.append('decision')  # Standard HITL output
                fields = set(field_names)
            
            # Tool nodes: Parse Composio response schema (SCALABLE!)
            tool_id = None
            if node_type == 'agent_with_tools':
                tool_id = node.get('tool_id')
                if tool_id and self.tool_cache:
                    tool_schema = self.tool_cache.get(tool_id, {})
                    response_schema = tool_schema.get('response_schema', {})
                    
                    # Extract response properties
                    properties = response_schema.get('properties', {})
                    if properties:
                        field_names = []
                        
                        # ✅ COMPOSIO PATTERN: Extract nested fields from 'data' wrapper
                        if 'data' in properties:
                            data_schema = properties['data']
                            
                            # If data is an object with properties, extract its fields
                            if 'properties' in data_schema:
                                data_props = data_schema['properties']
                                # Add both 'field' and 'data.field' for LLM flexibility
                                for field_name in data_props.keys():
                                    field_names.append(field_name)  # Direct access (LLM pattern)
                                    field_names.append(f"data.{field_name}")  # Correct Composio path
                        
                        # Also add top-level fields (error, successful, etc.)
                        field_names.extend(properties.keys())
                        
                        fields = set(field_names)
                        
                        logger.info(
                            f"[PLACEHOLDER_VALIDATOR] ✅ Tool '{tool_id}' has {len(fields)} response fields: {', '.join(list(fields)[:10])}"
                        )
            
            schemas[node_id] = NodeSchema(
                node_id=node_id,
                node_type=node_type,
                fields=fields,
                tool_id=tool_id,
                tool_cache_ref=self.tool_cache if tool_id else None
            )
        
        self._node_schemas = schemas
        return self
    
    @model_validator(mode='after')
    def extract_placeholder_references(self):
        """
        Extract all placeholder references from node params and check syntax.
        
        SCALABLE: Detects array access patterns for ANY field.
        Supports ${{from_step:...}}, ${{from_loop:...}}, and ${{@variable}} references.
        """
        double_brace_pattern = re.compile(r'\$\{\{from_(?:step|loop):([^.}]+)\.([^}]+)\}\}')
        single_brace_pattern = re.compile(r'\$\{from_(?:step|loop):')
        array_pattern = re.compile(r'^([^\[]+)\[(\d+)\](.*)$')  # Matches array[index].field
        auto_variable_pattern = re.compile(r'\$\{\{@(\w+)\}\}')  # ✅ @variable syntax
        refs = []
        syntax_errors = []
        
        for node in self.nodes:
            node_id = node.get('id')
            if not node_id:
                continue
            
            params = node.get('params', {})
            
            def _scan_params(val: Any, path: str = "params"):
                if isinstance(val, str):
                    # Check for single-brace syntax error
                    if single_brace_pattern.search(val) and not double_brace_pattern.search(val):
                        syntax_errors.append(
                            f"Node '{node_id}' uses single-brace ${{from_step/from_loop:...}}; "
                            f"use ${{{{from_step/from_loop:...}}}} instead in {path}"
                        )
                    
                    # ✅ Extract @variable references (auto-resolution at runtime)
                    auto_var_matches = auto_variable_pattern.findall(val)
                    for var_name in auto_var_matches:
                        refs.append(PlaceholderReference(
                            source_node_id="@auto",  # Special marker for auto-resolution
                            field_name=var_name,
                            target_node_id=node_id,
                            param_path=path,
                            full_placeholder=f"${{{{@{var_name}}}}}"
                        ))
                    
                    # ✅ SCALABLE FIX: Detect malformed placeholders (no dot, no @)
                    # This catches patterns like ${{user_email}} that should be ${{node.user_email}} or ${{@user_email}}
                    malformed_pattern = re.compile(r'\$\{\{([^}@]+)\}\}')  # No @ and no dot separator
                    malformed_matches = malformed_pattern.findall(val)
                    for match in malformed_matches:
                        # Skip if it's a valid format (has dot separator or starts with from_step:/from_loop:)
                        if '.' not in match and not match.startswith('from_step:') and not match.startswith('from_loop:'):
                            syntax_errors.append(
                                f"❌ Node '{node_id}' has malformed placeholder '${{{{{{match}}}}}}'.\n"
                                f"   Valid formats:\n"
                                f"   - ${{{{node_id.field}}}} (reference another node's field)\n"
                                f"   - ${{{{@variable}}}} (reference code node variable)\n"
                                f"   Hint: Did you mean '${{{{collect_passenger_details.{match}}}}}' or '${{{{@{match}}}}}'?"
                            )
                    
                    # Extract valid double-brace references
                    matches = double_brace_pattern.findall(val)
                    for source_node_id, field_path in matches:
                        # Parse array access
                        array_match = array_pattern.match(field_path)
                        
                        if array_match:
                            base_field = array_match.group(1)
                            index = array_match.group(2)
                            sub_path = array_match.group(3)
                            
                            refs.append(PlaceholderReference(
                                source_node_id=source_node_id,
                                field_name=base_field,
                                target_node_id=node_id,
                                param_path=path,
                                full_placeholder=f"${{{{from_step:{source_node_id}.{field_path}}}}}",
                                array_access=f"{base_field}[{index}]",
                                sub_field_path=sub_path if sub_path else None
                            ))
                        else:
                            # Simple field reference
                            refs.append(PlaceholderReference(
                                source_node_id=source_node_id,
                                field_name=field_path.split('.')[0],
                                target_node_id=node_id,
                                param_path=path,
                                full_placeholder=f"${{{{from_step:{source_node_id}.{field_path}}}}}"
                            ))
                elif isinstance(val, dict):
                    for k, v in val.items():
                        _scan_params(v, f"{path}.{k}")
                elif isinstance(val, list):
                    for i, v in enumerate(val):
                        _scan_params(v, f"{path}[{i}]")
            
            _scan_params(params)
        
        self._placeholder_refs = refs
        self._syntax_errors = syntax_errors
        return self
    
    def _analyze_array_access_patterns(self) -> Tuple[List[str], List[str]]:
        """
        🚀 SCALABLE: Analyze array access patterns across the ENTIRE workflow.
        
        Detects risky patterns that should use loops instead of hardcoded indices.
        Works for ANY tool, ANY array field!
        
        Returns:
            (errors, warnings) - Critical issues vs best practice suggestions
        """
        errors = []
        warnings = []
        
        # Group array accesses by (source_node, field_name)
        array_accesses: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = {}
        
        for ref in self._placeholder_refs:
            if ref.array_access:
                match = re.match(r'(.+?)\[(\d+)\]', ref.array_access)
                if match:
                    field = match.group(1)
                    index = int(match.group(2))
                    
                    key = (ref.source_node_id, field)
                    if key not in array_accesses:
                        array_accesses[key] = []
                    array_accesses[key].append((index, ref.target_node_id, ref.param_path))
        
        # Analyze patterns for each array
        for (source_node, field_name), accesses in array_accesses.items():
            indices = sorted(set(idx for idx, _, _ in accesses))
            unique_nodes = list(set(node for _, node, _ in accesses))
            
            # Critical: Accessing index > 0 without [0]
            if indices and min(indices) > 0:
                errors.append(
                    f"❌ ARRAY BOUNDS RISK: Nodes {unique_nodes} access '{field_name}[{min(indices)}]' "
                    f"from '{source_node}' but don't validate array length first. "
                    f"This will CRASH if the array has fewer than {min(indices) + 1} items!"
                )
            
            # Critical: Multiple sequential indices = should be a loop
            elif len(indices) >= 2:
                is_sequential = all(indices[i] == i for i in range(len(indices)))
                
                if is_sequential:
                    errors.append(
                        f"❌ HARDCODED ARRAY ITERATION DETECTED! "
                        f"Nodes {unique_nodes} access '{source_node}.{field_name}' with indices {indices}. "
                        f"Use a LOOP node instead for robustness!"
                    )
                else:
                    warnings.append(
                        f"⚠️  UNUSUAL PATTERN: Accessing '{source_node}.{field_name}' at non-sequential indices {indices}. "
                        f"Verify this is intentional."
                    )
            
            # Warning: Single access to [0] is OK but risky
            elif indices == [0]:
                warnings.append(
                    f"⚠️  Array bounds safety: Accessing '{source_node}.{field_name}[0]'. "
                    f"Ensure the array is not empty at runtime."
                )
        
        return errors, warnings
    
    def validate(self) -> PlaceholderValidationResult:
        """
        Validate all placeholder references.
        
        Returns:
            PlaceholderValidationResult with validation status and errors
        """
        errors = []
        warnings = []
        validated = 0
        skipped = 0
        
        # Add syntax errors first
        errors.extend(self._syntax_errors)
        
        # Analyze array access patterns
        array_errors, array_warnings = self._analyze_array_access_patterns()
        errors.extend(array_errors)
        warnings.extend(array_warnings)
        
        for ref in self._placeholder_refs:
            # ✅ Skip validation for @auto references (runtime-resolved)
            if ref.source_node_id == "@auto":
                skipped += 1
                logger.debug(
                    f"Skipping validation for @variable '{ref.field_name}' "
                    f"(will be auto-resolved at runtime)"
                )
                continue
            
            # Check if source node exists
            if ref.source_node_id not in self._node_schemas:
                errors.append(
                    f"Node '{ref.target_node_id}' references non-existent source node "
                    f"'{ref.source_node_id}' in {ref.param_path}: {ref.full_placeholder}"
                )
                continue
            
            source_schema = self._node_schemas[ref.source_node_id]
            
            # Skip validation for tool outputs without schema
            if not source_schema.has_known_schema:
                skipped += 1
                continue
            
            # Validate field exists in source schema
            if ref.field_name not in source_schema.fields:
                available = ', '.join(sorted(source_schema.fields))
                errors.append(
                    f"Node '{ref.target_node_id}' references non-existent field "
                    f"'{ref.field_name}' from node '{ref.source_node_id}'. "
                    f"Available fields: {available}"
                )
            else:
                # Validate nested paths
                if ref.sub_field_path:
                    is_valid, error_field, available_at_error = source_schema.validate_nested_path(
                        ref.field_name, 
                        ref.sub_field_path
                    )
                    
                    if not is_valid and available_at_error is not None:
                        available_list = ', '.join(sorted(available_at_error))
                        errors.append(
                            f"Node '{ref.target_node_id}' references non-existent nested field "
                            f"'{error_field}' in path '{ref.field_name}{ref.sub_field_path}' "
                            f"from node '{ref.source_node_id}'. Available fields: {available_list}"
                        )
                        continue
                    
                    # Warn about deep nested paths
                    nested_depth = ref.sub_field_path.count('.')
                    if nested_depth > 2:
                        warnings.append(
                            f"[BEST_PRACTICE] Node '{ref.target_node_id}' uses deep nested path "
                            f"'{ref.field_name}{ref.sub_field_path}'. Deep paths are fragile."
                        )
                    
                    # Warn about nested array access
                    if '[' in ref.sub_field_path:
                        warnings.append(
                            f"[BEST_PRACTICE] Node '{ref.target_node_id}' uses nested array access "
                            f"'{ref.field_name}{ref.sub_field_path}'. Consider using top-level fields."
                        )
                
                validated += 1
        
        # Log summary
        if self._placeholder_refs:
            logger.info(
                f"[PLACEHOLDER_VALIDATOR] Checked {len(self._placeholder_refs)} references: "
                f"{validated} validated, {skipped} skipped"
            )
        
        return PlaceholderValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            total_references=len(self._placeholder_refs),
            validated_references=validated,
            skipped_references=skipped
        )


def validate_workflow_placeholders(
    workflow_dict: Dict[str, Any],
    tool_cache: Optional[Dict[str, Any]] = None
) -> PlaceholderValidationResult:
    """
    Main entry point for placeholder validation.
    
    Args:
        workflow_dict: Workflow dictionary with nodes
        tool_cache: Optional tool schema cache for deep validation
    
    Returns:
        PlaceholderValidationResult with validation status
    """
    try:
        validator = WorkflowPlaceholderValidator(
            nodes=workflow_dict.get('nodes', []),
            tool_cache=tool_cache
        )
        return validator.validate()
    except Exception as e:
        logger.error(f"[PLACEHOLDER_VALIDATOR] Validation failed: {e}")
        return PlaceholderValidationResult(
            valid=False,
            errors=[f"Validation error: {str(e)}"]
        )

