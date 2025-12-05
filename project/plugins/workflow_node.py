# -*- coding: utf-8 -*-
"""
Workflow Node Plugin for TaskWeaver (Eclipse)
==============================================

Declaratively define workflow nodes during code generation.

This plugin follows TaskWeaver's standard plugin pattern and provides:
- Explicit workflow structure declaration in generated code
- No plan/code reconciliation needed - code IS the workflow
- Automatic node ordering and dependency tracking
- Integration with existing tool, form, and HITL infrastructure

Usage in generated code:
    # Declare a tool node
    result = workflow_node(
        "send_email",
        {
            "type": "tool",
            "name": "Send notification",
            "tool": "GMAIL_SEND_EMAIL",
            "params": {"recipient_email": "user@example.com", "subject": "Hello"}
        }
    )
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional, List

from taskweaver.plugin import Plugin, register_plugin

logger = logging.getLogger(__name__)


@register_plugin
class WorkflowNode(Plugin):
    """
    TaskWeaver plugin for declaring workflow nodes.
    
    DESIGN PHILOSOPHY:
    - Code declares structure explicitly via workflow_node() calls
    - No need for plan/code reconciliation
    - Nodes are tracked in session variables for converter to read
    - Each call = one node in the final workflow graph
    
    SIMULATION MODE:
    During workflow generation, nodes are registered and mock data is returned
    so subsequent code can reference upstream node outputs.
    
    EXECUTION MODE:
    During real execution, this plugin delegates to the appropriate
    executor (ComposioService for tools, HITL for forms, etc.)
    """
    
    _simulation_mode = None
    _node_sequence = 0  # Track node order
    
    def _is_simulation_mode(self) -> bool:
        """Detect if running in simulation (generation) vs real execution."""
        if self._simulation_mode is not None:
            return self._simulation_mode
        
        # Check session variable first (most reliable)
        try:
            is_gen_mode = self.ctx.get_session_var("_workflow_generation_mode", None)
            if is_gen_mode == "true":
                self._simulation_mode = True
                return True
        except Exception:
            pass
        
        # Fallback: check Django availability
        try:
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            if not django.apps.apps.ready:
                django.setup()
            self._simulation_mode = False
            return False
        except Exception:
            self._simulation_mode = True
            return True
    
    def _register_node(self, node_id: str, config: Dict[str, Any], sequence: int) -> None:
        """
        Register node in session variables for converter to read.
        
        The workflow converter will read _workflow_nodes from session
        instead of parsing plan text and classifying nodes.
        """
        try:
            # Get existing nodes or initialize
            nodes_json = self.ctx.get_session_var("_workflow_nodes", None)
            
            if nodes_json:
                import json
                nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
            else:
                nodes = []
            
            # Build node definition
            node_def = {
                "id": node_id,
                "sequence": sequence,
                "type": config.get("type", "logic"),
                "name": config.get("name", node_id),
                "description": config.get("description", ""),
                "depends_on": config.get("depends_on", []),
            }
            
            # Add type-specific fields
            node_type = config.get("type", "logic")
            
            if node_type == "tool":
                node_def["tool"] = config.get("tool")
                node_def["params"] = config.get("params", {})
                
            elif node_type in ("form", "hitl"):
                node_def["fields"] = config.get("fields", [])
                node_def["prefill"] = config.get("prefill", {})
                
                if node_type == "hitl":
                    node_def["decision_field"] = config.get("decision_field")
                    node_def["routing"] = config.get("routing", {})
            
            # Append to nodes list
            nodes.append(node_def)
            
            # Store back in session
            import json
            self.ctx.set_session_var("_workflow_nodes", json.dumps(nodes))
            
            logger.info(f"[WORKFLOW_NODE] ✅ Registered node '{node_id}' (#{sequence}, type={node_type})")
            
        except Exception as e:
            logger.warning(f"[WORKFLOW_NODE] Failed to register node: {e}")
    
    def _generate_mock_result(self, node_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock result data for simulation mode.
        
        Returns realistic mock data so subsequent code can reference
        this node's outputs without errors.
        """
        node_type = config.get("type", "logic")
        
        if node_type == "tool":
            tool_id = config.get("tool", "UNKNOWN")
            # Generate tool-appropriate mock
            return {
                "success": True,
                "data": {
                    "items": [
                        {"id": "item_1", "name": "Sample Item 1", "value": "data_1"},
                        {"id": "item_2", "name": "Sample Item 2", "value": "data_2"}
                    ],
                    "count": 2,
                    "summary": f"Mock result from {tool_id}"
                },
                "_node_id": node_id,
                "_mock": True
            }
            
        elif node_type in ("form", "hitl"):
            # Generate form field values
            mock_data = {"_node_id": node_id, "_mock": True}
            
            for field in config.get("fields", []):
                field_name = field.get("name", "")
                field_type = field.get("type", "text")
                options = field.get("options", [])
                default = field.get("default")
                
                if default is not None:
                    mock_data[field_name] = default
                elif options:
                    mock_data[field_name] = options[0]
                elif field_type == "email":
                    mock_data[field_name] = f"{field_name}@example.com"
                elif field_type == "tel":
                    mock_data[field_name] = "+1 (555) 123-4567"
                elif field_type == "date":
                    mock_data[field_name] = "2025-01-15"
                elif field_type == "number":
                    mock_data[field_name] = 0
                elif field_type == "boolean":
                    mock_data[field_name] = True
                else:
                    mock_data[field_name] = f"Sample {field_name}"
            
            return mock_data
            
        else:  # logic
            return {
                "processed": True,
                "result": f"Processed by {node_id}",
                "_node_id": node_id,
                "_mock": True
            }
    
    def __call__(
        self,
        node_id: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Declare a workflow node.
        
        Args:
            node_id: Unique identifier for this node
            config: Node configuration (type, name, tool/fields, params, etc.)
            
        Returns:
            Tuple of (result_data, description)
        """
        # Increment sequence counter
        WorkflowNode._node_sequence += 1
        sequence = WorkflowNode._node_sequence
        
        node_type = config.get("type", "logic")
        node_name = config.get("name", node_id)
        
        logger.info(f"[WORKFLOW_NODE] Declaring node '{node_id}' (type={node_type}, seq={sequence})")
        
        # Register node in session for converter
        self._register_node(node_id, config, sequence)
        
        # Check mode
        if self._is_simulation_mode():
            logger.info(f"[WORKFLOW_NODE] SIMULATION MODE: Generating mock for '{node_id}'")
            
            mock_result = self._generate_mock_result(node_id, config)
            
            description = (
                f"[SIMULATION] Node '{node_id}' ({node_type}) registered. "
                f"Mock data returned for workflow generation."
            )
            
            return mock_result, description
        
        # =====================================================================
        # REAL EXECUTION MODE
        # =====================================================================
        # Delegate to appropriate executor based on node type
        # =====================================================================
        
        if node_type == "tool":
            return self._execute_tool_node(node_id, config)
        elif node_type in ("form", "hitl"):
            return self._execute_form_node(node_id, config)
        else:
            return self._execute_logic_node(node_id, config)
    
    def _execute_tool_node(
        self,
        node_id: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Execute a tool node via Composio."""
        tool_id = config.get("tool")
        params = config.get("params", {})
        
        logger.info(f"[WORKFLOW_NODE] Executing tool node '{node_id}': {tool_id}")
        
        try:
            # Use composio_action plugin
            from taskweaver.plugin import get_plugin
            composio = get_plugin("composio_action")
            
            if composio:
                result, desc = composio(tool_id, params)
                return result, f"Tool node '{node_id}' executed: {desc}"
            else:
                # Fallback to mock
                return self._generate_mock_result(node_id, config), f"Tool node '{node_id}': plugin not available"
                
        except Exception as e:
            logger.error(f"[WORKFLOW_NODE] Tool execution failed: {e}")
            return {"error": str(e), "_node_id": node_id}, f"Tool node '{node_id}' failed: {e}"
    
    def _execute_form_node(
        self,
        node_id: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Execute a form/HITL node."""
        logger.info(f"[WORKFLOW_NODE] Executing form node '{node_id}'")
        
        try:
            # Use form_collect plugin
            from taskweaver.plugin import get_plugin
            form_plugin = get_plugin("form_collect")
            
            if form_plugin:
                schema = {
                    "title": config.get("name", node_id),
                    "description": config.get("description", ""),
                    "fields": config.get("fields", []),
                    "prefill": config.get("prefill", {})
                }
                
                # Add HITL config if present
                if config.get("type") == "hitl":
                    schema["hitl_config"] = {
                        "decision_field": config.get("decision_field"),
                        "routing": config.get("routing", {})
                    }
                
                result, desc = form_plugin(node_id, schema)
                return result, f"Form node '{node_id}' collected: {desc}"
            else:
                return self._generate_mock_result(node_id, config), f"Form node '{node_id}': plugin not available"
                
        except Exception as e:
            logger.error(f"[WORKFLOW_NODE] Form execution failed: {e}")
            return {"error": str(e), "_node_id": node_id}, f"Form node '{node_id}' failed: {e}"
    
    def _execute_logic_node(
        self,
        node_id: str,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Execute a logic node (data transformation)."""
        logger.info(f"[WORKFLOW_NODE] Executing logic node '{node_id}'")
        
        # Logic nodes are typically inline code, just return success
        return {
            "processed": True,
            "node_id": node_id,
            "description": config.get("description", "")
        }, f"Logic node '{node_id}' executed"


# =====================================================================
# HELPER FUNCTIONS FOR CONVERTER
# =====================================================================

def get_workflow_nodes_from_session(session) -> List[Dict[str, Any]]:
    """
    Extract registered workflow nodes from TaskWeaver session.
    
    Called by workflow_converter_service to get the declared workflow
    structure instead of parsing plan text.
    
    Args:
        session: TaskWeaver session object
        
    Returns:
        List of node definitions in declaration order
    """
    import json
    
    try:
        # session.session_var is a dict, not a method
        nodes_json = session.session_var.get("_workflow_nodes", None)
        
        if not nodes_json:
            return []
        
        nodes = json.loads(nodes_json) if isinstance(nodes_json, str) else nodes_json
        
        # Sort by sequence number
        nodes.sort(key=lambda n: n.get("sequence", 0))
        
        logger.info(f"[WORKFLOW_NODE] Retrieved {len(nodes)} nodes from session")
        return nodes
        
    except Exception as e:
        logger.error(f"[WORKFLOW_NODE] Failed to get nodes from session: {e}")
        return []


def reset_node_sequence():
    """Reset the node sequence counter (for testing or new sessions)."""
    WorkflowNode._node_sequence = 0

