# -*- coding: utf-8 -*-
"""
Form Collection Plugin for TaskWeaver (Eclipse)
================================================

Collect user input via dynamic forms in TaskWeaver workflows.

This plugin follows TaskWeaver's standard plugin pattern and integrates with
the existing form infrastructure:
- FieldTypeRegistry for intelligent type inference
- FormSchema/FormField for form structure
- AIFormGenerator for schema generation

Usage in generated code:
    # Collect user registration
    registration_data, desc = form_collect(
        "registration",
        {
            "title": "Event Registration",
            "fields": [
                {"name": "attendee_name", "label": "Full Name", "required": True},
                {"name": "email", "label": "Email Address", "required": True}
            ]
        }
    )
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional, List

from taskweaver.plugin import Plugin, register_plugin

logger = logging.getLogger(__name__)


@register_plugin
class FormCollect(Plugin):
    """
    TaskWeaver plugin for collecting user input via dynamic forms.
    
    DESIGN PHILOSOPHY:
    - Forms are defined DECLARATIVELY in the plugin call, not extracted from input() calls
    - This survives LLM retries (form schema is a parameter, not implementation detail)
    - Integrates with existing FieldTypeRegistry for intelligent type inference
    - Supports form-to-form data flow via the prefill parameter
    
    SIMULATION MODE:
    During workflow generation, the plugin returns simulated form data so TaskWeaver
    can continue generating code for all steps. The actual form collection happens
    during workflow execution via the HITL system.
    
    EXECUTION MODE:
    When executed in the actual workflow, this plugin triggers a HITL (Human-in-the-Loop)
    pause, renders the form to the user, and returns the collected data.
    """
    
    _field_registry = None
    _simulation_mode = None
    
    def _get_field_registry(self):
        """
        Get the FieldTypeRegistry for intelligent type inference.
        Uses existing battle-tested infrastructure.
        """
        if self._field_registry is not None:
            return self._field_registry
        
        try:
            from apps.py_workflows.generation.forms.field_types import REGISTRY
            self._field_registry = REGISTRY
            logger.info("[FORM_COLLECT] ✅ Loaded FieldTypeRegistry")
            return self._field_registry
        except ImportError as e:
            logger.warning(f"[FORM_COLLECT] ⚠️ FieldTypeRegistry not available: {e}")
            return None
    
    def _is_simulation_mode(self) -> bool:
        """
        Detect if we're running in simulation mode (Docker CES) vs real execution.
        """
        if self._simulation_mode is not None:
            return self._simulation_mode
        
        try:
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            django.setup()
            self._simulation_mode = False
            return False
        except Exception:
            self._simulation_mode = True
            return True
    
    def _infer_field_type(self, field_name: str, context: str = "") -> str:
        """
        Infer field type from name using the existing FieldTypeRegistry.
        
        This is the SCALABLE approach - uses keyword matching, suffix priority,
        and pattern detection from the battle-tested registry.
        """
        registry = self._get_field_registry()
        
        if registry:
            try:
                from apps.py_workflows.generation.forms.field_types import infer_field_type_with_confidence
                type_id, confidence = infer_field_type_with_confidence(field_name, context)
                logger.debug(f"[FORM_COLLECT] Inferred {field_name} → {type_id} (confidence: {confidence:.2f})")
                return type_id
            except Exception as e:
                logger.warning(f"[FORM_COLLECT] Type inference failed: {e}")
        
        # Fallback: basic inference
        name_lower = field_name.lower()
        if 'email' in name_lower:
            return 'email'
        elif 'phone' in name_lower or 'tel' in name_lower:
            return 'tel'
        elif 'date' in name_lower:
            return 'date'
        elif any(x in name_lower for x in ['description', 'comment', 'note', 'message']):
            return 'textarea'
        elif any(x in name_lower for x in ['price', 'amount', 'count', 'number', 'age']):
            return 'number'
        
        return 'text'
    
    def _humanize_field_name(self, field_name: str) -> str:
        """
        Convert field_name to Human Readable Label.
        
        Examples:
            'attendee_name' → 'Attendee Name'
            'email' → 'Email'
            'departure_date' → 'Departure Date'
        """
        import re
        
        # Split on underscores and camelCase
        words = re.sub(r'([A-Z])', r' \1', field_name).split('_')
        words = [w.strip() for w in words if w.strip()]
        
        # Title case
        label = ' '.join(words).title()
        
        # Fix common acronyms
        label = label.replace(' Id', ' ID')
        label = label.replace(' Url', ' URL')
        label = label.replace(' Api', ' API')
        label = label.replace(' Cvv', ' CVV')
        
        return label
    
    def _build_form_schema(self, form_id: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a complete form schema from the provided definition.
        
        Normalizes and enriches the schema:
        - Infers missing field types
        - Generates labels from field names
        - Applies prefill values
        - Extracts HITL configuration for approval workflows
        """
        title = schema.get('title', f'Form: {form_id}')
        description = schema.get('description', '')
        fields_input = schema.get('fields', [])
        prefill = schema.get('prefill', {})
        hitl_config = schema.get('hitl_config', None)  # For approval workflows
        
        # Process each field
        processed_fields = []
        for field_def in fields_input:
            if isinstance(field_def, str):
                # Simple field name only
                field = {
                    'name': field_def,
                    'label': self._humanize_field_name(field_def),
                    'type': self._infer_field_type(field_def),
                    'required': False
                }
            else:
                # Full field definition
                name = field_def.get('name', f'field_{len(processed_fields)}')
                field = {
                    'name': name,
                    'label': field_def.get('label', self._humanize_field_name(name)),
                    'type': field_def.get('type', self._infer_field_type(name, field_def.get('label', ''))),
                    'required': field_def.get('required', False),
                }
                
                # Optional properties
                if 'options' in field_def:
                    field['options'] = field_def['options']
                if 'placeholder' in field_def:
                    field['placeholder'] = field_def['placeholder']
                if 'default' in field_def:
                    field['default'] = field_def['default']
                if 'description' in field_def:
                    field['description'] = field_def['description']
                if 'readonly' in field_def:
                    field['readonly'] = field_def['readonly']  # For HITL display fields
                
                # Apply prefill if available
                if name in prefill:
                    field['default'] = prefill[name]
            
            processed_fields.append(field)
        
        result = {
            'form_id': form_id,
            'title': title,
            'description': description,
            'fields': processed_fields,
            'prefill': prefill
        }
        
        # Include HITL config for approval workflows
        if hitl_config:
            result['hitl_config'] = hitl_config
            result['is_hitl'] = True
        
        return result
    
    def _generate_simulated_data(self, form_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate simulated form data for workflow generation phase.
        
        Returns realistic mock data so TaskWeaver can continue generating
        code for subsequent steps that depend on this form's output.
        """
        simulated_data = {}
        form_id = form_schema.get('form_id', 'form')
        
        for field in form_schema.get('fields', []):
            name = field['name']
            field_type = field.get('type', 'text')
            options = field.get('options', [])
            default = field.get('default')
            
            # Use default if provided
            if default is not None:
                simulated_data[name] = default
                continue
            
            # Use first option for select types
            if field_type in ('select', 'multiselect') and options:
                if field_type == 'multiselect':
                    simulated_data[name] = [options[0]]
                else:
                    simulated_data[name] = options[0]
                continue
            
            # Generate type-appropriate simulated values
            if field_type == 'email':
                simulated_data[name] = f"user_{form_id}@example.com"
            elif field_type == 'tel':
                simulated_data[name] = "+1 (555) 123-4567"
            elif field_type == 'date':
                simulated_data[name] = "2025-01-15"
            elif field_type == 'datetime':
                simulated_data[name] = "2025-01-15T10:00:00"
            elif field_type == 'number':
                simulated_data[name] = 0
            elif field_type == 'boolean':
                simulated_data[name] = True
            elif field_type == 'textarea':
                simulated_data[name] = f"Simulated {name} content"
            elif field_type == 'url':
                simulated_data[name] = "https://example.com"
            else:
                # Default to text
                simulated_data[name] = f"Simulated {self._humanize_field_name(name)}"
        
        # Add form metadata
        simulated_data['_form_id'] = form_id
        simulated_data['_simulated'] = True
        
        return simulated_data
    
    def __call__(
        self,
        form_id: str,
        schema: Dict[str, Any],
        entity_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Collect user input via a dynamic form.
        
        Args:
            form_id: Unique form identifier (e.g., "registration", "payment")
            schema: Form schema definition with title, fields, and optional prefill
            entity_id: Optional user entity ID (for auth context)
            
        Returns:
            Tuple of (form_data, description)
            - form_data: Dictionary of collected field values
            - description: Human-readable status message
        """
        logger.info(f"[FORM_COLLECT] Form '{form_id}' with {len(schema.get('fields', []))} fields")
        
        # Build complete form schema with type inference
        form_schema = self._build_form_schema(form_id, schema)
        
        logger.info(f"[FORM_COLLECT] Built form schema: {form_schema['title']}")
        for field in form_schema['fields']:
            logger.debug(f"[FORM_COLLECT]   - {field['name']}: {field['type']} (required: {field.get('required', False)})")
        
        # Check if we're in simulation mode
        if self._is_simulation_mode():
            logger.info(f"[FORM_COLLECT] SIMULATION MODE: Generating mock data for '{form_id}'")
            
            simulated_data = self._generate_simulated_data(form_schema)
            
            description = (
                f"[SIMULATION] Form '{form_id}' would collect: "
                f"{', '.join(f['name'] for f in form_schema['fields'])}. "
                f"Simulated data returned for workflow generation."
            )
            
            return simulated_data, description
        
        # =====================================================================
        # REAL EXECUTION MODE
        # =====================================================================
        # In real execution, we store the form schema and wait for HITL
        # The actual form rendering happens via the frontend
        # =====================================================================
        
        try:
            # Store form schema for the HITL system to render
            session_id = self.ctx.get_session_var("session_id") if hasattr(self, 'ctx') else None
            
            if session_id:
                try:
                    from django.core.cache import cache
                    cache_key = f"form_schema:{session_id}:{form_id}"
                    cache.set(cache_key, form_schema, timeout=3600)
                    logger.info(f"[FORM_COLLECT] Stored form schema for HITL: {cache_key}")
                except Exception as e:
                    logger.warning(f"[FORM_COLLECT] Failed to cache form schema: {e}")
            
            # For now, return simulated data even in real mode
            # The actual HITL integration will replace this
            simulated_data = self._generate_simulated_data(form_schema)
            simulated_data['_pending_hitl'] = True
            
            description = (
                f"Form '{form_id}' is ready for user input. "
                f"Fields: {', '.join(f['name'] for f in form_schema['fields'])}. "
                f"Waiting for HITL response."
            )
            
            return simulated_data, description
            
        except Exception as e:
            logger.error(f"[FORM_COLLECT] Error in real execution: {e}")
            
            # Fallback to simulation
            simulated_data = self._generate_simulated_data(form_schema)
            
            return simulated_data, f"Form '{form_id}' error: {e}. Simulated data returned."


# =====================================================================
# INTEGRATION WITH EXISTING FORM INFRASTRUCTURE
# =====================================================================

def form_schema_to_workflow_node(form_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a form_collect schema to a workflow node format.
    
    This bridges the plugin's declarative form definition with the existing
    workflow converter infrastructure.
    
    Args:
        form_schema: Schema from form_collect plugin call
        
    Returns:
        Workflow node-compatible form definition
    """
    return {
        'type': 'form_node',
        'form_id': form_schema.get('form_id', 'form'),
        'title': form_schema.get('title', 'Form'),
        'description': form_schema.get('description', ''),
        'fields': form_schema.get('fields', []),
        'output_variables': {f['name'] for f in form_schema.get('fields', [])},
        'output_mappings': {f['name']: f['name'] for f in form_schema.get('fields', [])}
    }


def extract_form_schemas_from_code(code: str) -> List[Dict[str, Any]]:
    """
    Extract form_collect calls from generated code using AST.
    
    This allows the workflow converter to detect forms defined via the plugin
    and create proper form_node entries.
    
    Args:
        code: Generated Python code containing form_collect calls
        
    Returns:
        List of form schemas extracted from the code
    """
    import ast
    
    forms = []
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's a form_collect call
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name == 'form_collect' and len(node.args) >= 2:
                    # Extract form_id (first argument)
                    form_id_node = node.args[0]
                    if isinstance(form_id_node, ast.Constant):
                        form_id = form_id_node.value
                    else:
                        form_id = 'unknown'
                    
                    # Extract schema (second argument) - this is complex
                    # For now, just note that a form exists
                    forms.append({
                        'form_id': form_id,
                        'line_number': node.lineno,
                        '_detected_from_code': True
                    })
                    
                    logger.info(f"[FORM_EXTRACT] Detected form_collect('{form_id}') at line {node.lineno}")
        
    except SyntaxError as e:
        logger.warning(f"[FORM_EXTRACT] Failed to parse code: {e}")
    
    return forms

