"""
Dynamic Pydantic Model Validator
=================================
Generate tool-specific Pydantic models from Composio schemas for automatic validation.

THIS IS THE ULTIMATE SCALABLE SOLUTION:
- No manual auto-healing code needed
- No hardcoded patterns
- Pydantic automatically validates params, types, required fields
- Add new tools → validation scales automatically

Architecture:
    Tool Schema (JSON Schema)
          ↓
    create_model() → Dynamic Pydantic Model
          ↓
    Automatic Validation (required, types, constraints)
          ↓
    Type Coercion (string → int, etc.)
          ↓
    Clear Error Messages (which field, why it failed)
"""
from typing import Dict, Any, Optional, Type, get_args, get_origin
from pydantic import BaseModel, create_model, Field, ValidationError
import json
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)


class ToolSchemaCache:
    """Singleton cache for tool schemas."""
    _instance = None
    _schemas: Dict[str, Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_schemas(self, cache_path: str = None) -> bool:
        """Load schemas from cache file."""
        if self._schemas is not None:
            return True
        
        if cache_path is None:
            cache_path = Path(__file__).parent.parent.parent.parent / "project" / "plugins" / "composio_schemas_cache.json"
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                self._schemas = json.load(f)
            logger.info(f"[DYNAMIC_VAL] Loaded {len(self._schemas)} tool schemas")
            return True
        except Exception as e:
            logger.error(f"[DYNAMIC_VAL] Failed to load schemas: {e}")
            self._schemas = {}
            return False
    
    def get_schema(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get schema for tool."""
        if self._schemas is None:
            self.load_schemas()
        return self._schemas.get(tool_id)


@lru_cache(maxsize=1000)
def create_tool_param_model(tool_id: str) -> Optional[Type[BaseModel]]:
    """
    Create a dynamic Pydantic model for tool parameters from JSON schema.
    
    This is THE SCALABLE SOLUTION:
    - Reads tool schema from cache
    - Converts JSON schema to Pydantic model
    - Returns model class that validates params automatically
    
    Args:
        tool_id: Composio tool ID (e.g., GMAIL_SEND_EMAIL)
    
    Returns:
        Pydantic model class for tool params, or None if schema not found
    """
    cache = ToolSchemaCache()
    tool_schema = cache.get_schema(tool_id)
    
    if not tool_schema:
        logger.warning(f"[DYNAMIC_VAL] Schema not found for tool: {tool_id}")
        return None
    
    params_schema = tool_schema.get('parameters_schema', {})
    if not params_schema or 'properties' not in params_schema:
        logger.warning(f"[DYNAMIC_VAL] No parameter schema for tool: {tool_id}")
        return None
    
    # Convert JSON schema to Pydantic fields
    fields = {}
    properties = params_schema.get('properties', {})
    required_fields = set(params_schema.get('required', []))
    
    for field_name, field_schema in properties.items():
        field_type_str = field_schema.get('type')
        description = field_schema.get('description', '')
        default = field_schema.get('default')
        examples = field_schema.get('examples', [])
        
        # Map JSON schema types to Python types
        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        python_type = type_map.get(field_type_str, Any)
        
        # Build Field with metadata
        if field_name in required_fields:
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            # Use schema default if available
            if default is not None:
                fields[field_name] = (python_type, Field(default=default, description=description))
            else:
                fields[field_name] = (python_type, Field(default=None, description=description))
    
    # Create dynamic model
    model_name = f"{tool_id}_Params"
    DynamicModel = create_model(model_name, **fields)
    
    logger.info(f"[DYNAMIC_VAL] Created model for {tool_id}: {len(fields)} fields, {len(required_fields)} required")
    return DynamicModel


def validate_node_params(node: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """
    Validate node params using dynamically generated Pydantic model.
    
    This replaces ALL manual auto-healing with Pydantic's built-in validation.
    
    Args:
        node: Node dict with tool_id and params
    
    Returns:
        (normalized_node, warnings)
        - normalized_node: Node with validated and coerced params
        - warnings: List of validation warnings (empty if all good)
    """
    warnings = []
    node_id = node.get('id', 'unknown')
    tool_id = node.get('tool_id')
    node_type = node.get('type')
    
    # Only validate agent_with_tools nodes
    if node_type != 'agent_with_tools' or not tool_id:
        return node, warnings
    
    # Get or create dynamic model for this tool
    ParamModel = create_tool_param_model(tool_id)
    
    if not ParamModel:
        warnings.append(f"[WARNING] Node '{node_id}': No schema available for tool '{tool_id}'")
        return node, warnings
    
    # Validate params with dynamic model
    params = node.get('params', {})
    
    try:
        # Pydantic validates and coerces types automatically!
        validated_params = ParamModel(**params)
        
        # Convert back to dict (with coerced types)
        node['params'] = validated_params.model_dump(exclude_none=False)
        
        logger.info(
            f"[DYNAMIC_VAL] Node '{node_id}': Params validated successfully "
            f"({len(params)} provided)"
        )
        
    except ValidationError as e:
        # Pydantic validation failed - collect errors
        for error in e.errors():
            field = error['loc'][0] if error['loc'] else 'unknown'
            msg = error['msg']
            error_type = error['type']
            
            if error_type == 'missing':
                warnings.append(f"[ERROR] Node '{node_id}': Missing required param '{field}'")
            elif 'type' in error_type:
                warnings.append(f"[ERROR] Node '{node_id}': Param '{field}' has wrong type: {msg}")
            else:
                warnings.append(f"[ERROR] Node '{node_id}': Param '{field}': {msg}")
    
    except Exception as e:
        warnings.append(f"[ERROR] Node '{node_id}': Validation failed: {e}")
    
    return node, warnings


def validate_workflow_with_dynamic_models(workflow_json: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """
    Validate entire workflow using dynamic Pydantic models.
    
    THE ULTIMATE SCALABLE VALIDATION:
    - Generates tool-specific models on-the-fly
    - Pydantic handles all validation logic
    - No manual auto-healing needed
    - Infinitely scalable
    
    Args:
        workflow_json: Workflow dict from LLM
    
    Returns:
        (normalized_workflow, all_warnings)
    """
    all_warnings = []
    
    for node in workflow_json.get('nodes', []):
        normalized_node, warnings = validate_node_params(node)
        all_warnings.extend(warnings)
    
    if all_warnings:
        logger.warning(f"[DYNAMIC_VAL] {len(all_warnings)} validation warnings")
        for warning in all_warnings:
            logger.warning(f"  {warning}")
    else:
        logger.info(f"[DYNAMIC_VAL] All node params validated successfully")
    
    return workflow_json, all_warnings
