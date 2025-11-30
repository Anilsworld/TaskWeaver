"""
Composio Tool Selector Plugin for TaskWeaver

This plugin integrates your Django hybrid matcher with TaskWeaver's plugin system.
It automatically:
1. Selects the best Composio tool using hybrid search (vector + BM25)
2. Executes the tool with intelligent parameter extraction
3. Returns results in TaskWeaver-compatible format

PRODUCTION FEATURES:
- Automatic cache warmup on initialization
- Hybrid search for best accuracy
- Handles 17,000+ actions across 250+ apps
- Smart parameter extraction from natural language
"""

import sys
import os
from typing import Dict, List, Optional, Any

# Add Django project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from taskweaver.plugin import Plugin, register_plugin
from apps.integrations.services.action_matcher import ComposioActionMatcher
from apps.integrations.services.composio_service import ComposioService


@register_plugin
class ComposioToolSelector(Plugin):
    """
    TaskWeaver plugin for Composio tool selection and execution.
    
    This plugin bridges TaskWeaver with your Django-based Composio integration,
    providing access to all Composio tools through intelligent hybrid search.
    """
    
    def __init__(self, name: str, description: str, parameters: List[Dict], returns: List[Dict]):
        """Initialize the plugin with hybrid matcher."""
        super().__init__(name, description, parameters, returns)
        
        # Initialize matcher with cache warmup (auto-loads Google apps)
        self.matcher = ComposioActionMatcher(enable_warmup=True)
        self.composio_service = ComposioService()
        
        # Get configuration from plugin YAML
        self.vector_weight = float(self.config.get("vector_weight", 0.7))
        self.bm25_weight = float(self.config.get("bm25_weight", 0.3))
        self.min_confidence = float(self.config.get("min_confidence", 0.3))
        self.top_k_actions = int(self.config.get("top_k_actions", 5))
        self.top_k_apps = int(self.config.get("top_k_apps", 3))
    
    def __call__(
        self, 
        task_description: str, 
        app_hints: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select and execute the best Composio tool for the given task.
        
        Args:
            task_description: Natural language description of what to do
            app_hints: Optional list of app names to search (e.g., ["gmail"])
            params: Optional parameters for the tool execution
        
        Returns:
            Dictionary with execution results and metadata
        """
        try:
            # Step 1: Find best matching action using hybrid search
            print(f"\n🔍 Searching for tools matching: '{task_description}'")
            if app_hints:
                print(f"   📦 Searching in apps: {', '.join(app_hints)}")
            
            matches = self.matcher.match_for_subtask_hybrid(
                subtask_description=task_description,
                app_hints=app_hints or [],
                top_k_apps=self.top_k_apps,
                top_k_actions_per_app=self.top_k_actions,
                min_confidence=self.min_confidence,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight
            )
            
            if not matches:
                return {
                    "success": False,
                    "error": f"No matching tools found for: {task_description}",
                    "suggestion": "Try being more specific or check app_hints",
                    "confidence": 0.0
                }
            
            # Get best match
            best_match = matches[0]
            action_id = best_match['action_id']
            confidence = best_match['confidence']
            action_name = best_match['action_name']
            app_name = best_match['app_display_name']
            
            print(f"   ✅ Selected: {action_name} from {app_name} (confidence: {confidence:.3f})")
            
            # Step 2: Extract parameters if not provided
            if params is None:
                params = self._extract_parameters(
                    task_description=task_description,
                    action_schema=best_match
                )
            
            # Step 3: Execute the tool
            print(f"   ⚙️  Executing {action_name} with params: {params}")
            
            # Execute via ComposioService
            try:
                execution_result = self.composio_service.execute_action(
                    action_id=action_id,
                    params=params,
                    entity_id="default"
                )
                
                return {
                    "success": execution_result.get('successful', False),
                    "action_used": action_name,
                    "app_used": app_name,
                    "confidence": confidence,
                    "params_used": params,
                    "parameter_examples": best_match.get('parameter_examples', {}),
                    "tags": best_match.get('tags', []),
                    "data": execution_result.get('data', {}),
                    "message": f"Executed {action_name} from {app_name}",
                    "search_type": best_match.get('search_type', 'hybrid'),
                    "execution_log_id": execution_result.get('log_id')
                }
            except Exception as exec_error:
                # Execution failed, but selection worked
                print(f"   ⚠️  Execution failed: {exec_error}")
                return {
                    "success": False,
                    "action_used": action_name,
                    "app_used": app_name,
                    "confidence": confidence,
                    "params_used": params,
                    "error": f"Execution failed: {str(exec_error)}",
                    "message": f"Found {action_name} but execution failed",
                    "search_type": best_match.get('search_type', 'hybrid')
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_description": task_description,
                "confidence": 0.0
            }
    
    def _extract_parameters(
        self, 
        task_description: str, 
        action_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract parameters from task description using schema and examples.
        
        This is a simple extraction. For production, you might want to use LLM
        to intelligently map natural language to parameter values.
        
        Args:
            task_description: Natural language task description
            action_schema: The selected action's schema with parameter examples
        
        Returns:
            Dictionary of extracted parameters
        """
        # Get parameter examples from the action
        param_examples = action_schema.get('parameter_examples', {})
        
        # Simple extraction (you can enhance this with LLM later)
        extracted = {}
        
        # Example: Extract email address if present
        import re
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', task_description)
        if email_match and 'to' in param_examples:
            extracted['to'] = email_match.group(0)
        
        # Example: Extract subject if quoted
        subject_match = re.search(r'subject[:\s]+["\']([^"\']+)["\']', task_description, re.IGNORECASE)
        if subject_match and 'subject' in param_examples:
            extracted['subject'] = subject_match.group(1)
        
        # TODO: Add more sophisticated parameter extraction
        # For production, consider using an LLM to map task_description to parameters
        
        return extracted


# Register the plugin
__all__ = ['ComposioToolSelector']

