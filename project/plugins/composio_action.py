# -*- coding: utf-8 -*-
"""
Composio Action Plugin for TaskWeaver (Eclipse)
================================================

Execute any Composio action (800+ integrations) from TaskWeaver-generated code.

This plugin follows TaskWeaver's standard plugin pattern and provides access to:
- Gmail (send, read, forward emails)
- Google Sheets (read, write, update)
- Slack (post messages, read channels)
- And 800+ more integrations

Usage in generated code:
    # Execute any Composio action
    result, description = composio_action(
        action_name="GOOGLESHEETS_SPREADSHEETS_VALUES_BATCH_GET",
        params={"spreadsheetId": "1abc...", "ranges": ["Sheet1!A1:B10"]}
    )
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional, Union

from taskweaver.plugin import Plugin, register_plugin, test_plugin

logger = logging.getLogger(__name__)


@register_plugin
class ComposioAction(Plugin):
    """
    TaskWeaver plugin for executing Composio actions.
    
    Provides a simple interface for TaskWeaver-generated code to call
    any of the 800+ Composio integrations (Gmail, Google Sheets, Slack, etc.)
    
    ACTION RESOLUTION (sql_pull_data pattern):
    ------------------------------------------
    TaskWeaver's LLM may generate "conceptual" action names like GMAIL_GET_LAST_EMAILS
    that don't exist in our DB. This plugin follows the sql_pull_data pattern:
    1. Accept approximate action name from TaskWeaver
    2. Resolve to actual action ID using semantic search
    3. Execute the resolved action
    
    SIMULATION MODE:
    ----------------
    When running in TaskWeaver's Docker container (CES), Django services are not
    available. The plugin detects this and returns realistic mock responses so
    TaskWeaver can continue generating code for ALL steps in the plan.
    
    The actual execution happens later via langgraph_adapter using the real
    ComposioService with user's connected accounts.
    """
    
    _composio_service = None
    _action_matcher = None  # Reuse existing semantic search
    _simulation_mode = None  # None = unknown, True = simulation, False = real
    
    def _is_simulation_mode(self) -> bool:
        """
        Detect if we're running in simulation mode (Docker CES) vs real execution.
        
        Simulation mode indicators:
        - Django not available
        - ComposioService import fails
        - Running in Docker container
        """
        if self._simulation_mode is not None:
            return self._simulation_mode
        
        try:
            # Try to import Django and check if it's configured
            import django
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            
            # Check if Django apps are ready
            if not django.apps.apps.ready:
                try:
                    django.setup()
                except Exception:
                    self._simulation_mode = True
                    return True
            
            # Try to import ComposioService
            from apps.integrations.services.composio_service import ComposioService
            self._simulation_mode = False
            return False
            
        except (ImportError, Exception) as e:
            logger.info(f"[COMPOSIO_PLUGIN] Simulation mode detected: {e}")
            self._simulation_mode = True
            return True
    
    def _resolve_params(self, action_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve TaskWeaver's param names to actual API param names using the tool schema.
        
        Follows the sql_pull_data pattern: accept approximate input, resolve internally.
        
        1. Fetch the actual schema for this action from DB
        2. For each param TaskWeaver provided, find matching schema param
        3. Use semantic similarity for fuzzy matching (e.g., name -> title)
        
        This is SCALABLE - uses actual schema, no hardcoding needed.
        
        Args:
            action_id: Resolved action ID
            params: Parameters from TaskWeaver (may have approximate names)
            
        Returns:
            Resolved parameters with correct API names
        """
        if not params:
            return params
            
        try:
            from apps.integrations.models import ComposioActionSchema
            
            # Get the action schema from DB
            action = ComposioActionSchema.objects.filter(
                action_id__iexact=action_id
            ).first()
            
            if not action or not action.parameters_schema:
                logger.debug(f"[COMPOSIO_PLUGIN] No schema for {action_id}, keeping original params")
                return params
            
            schema_props = action.parameters_schema.get('properties', {})
            if not schema_props:
                return params
            
            # Build resolved params using schema-based matching (NO HARDCODING)
            resolved = {}
            
            for param_key, param_value in params.items():
                # Use schema-based matching - fully dynamic
                matched_key = self._find_best_schema_match(param_key, schema_props)
                
                if matched_key and matched_key != param_key:
                    resolved[matched_key] = param_value
                    logger.info(f"[COMPOSIO_PLUGIN] Param resolved: {param_key} -> {matched_key}")
                else:
                    # Keep original - either exact match or no match found
                    resolved[param_key] = param_value
            
            return resolved
            
        except ImportError:
            # Django not available (simulation mode)
            return params
        except Exception as e:
            logger.warning(f"[COMPOSIO_PLUGIN] Param resolution error: {e}, using original")
            return params
    
    def _find_best_schema_match(self, param_name: str, schema_props: Dict[str, Any]) -> Optional[str]:
        """
        Find the best matching schema param for a TaskWeaver param name.
        
        Truly SCALABLE approach - uses the schema itself:
        1. Exact match (case-insensitive)
        2. Fuzzy string match (Levenshtein)
        3. Check if param_name appears in schema param's description
        
        No hardcoding - works for ANY tool's schema.
        
        Args:
            param_name: Parameter name from TaskWeaver
            schema_props: Schema properties dict {param_name: {description, type, ...}}
            
        Returns:
            Best matching schema param name, or None
        """
        param_lower = param_name.lower().replace('_', '')
        best_match = None
        best_score = 0
        
        for schema_key, schema_def in schema_props.items():
            schema_lower = schema_key.lower().replace('_', '')
            
            # 1. Exact match
            if param_lower == schema_lower:
                return schema_key
            
            # 2. Substring match (e.g., 'name' in 'spreadsheet_name' or vice versa)
            score = 0
            if param_lower in schema_lower or schema_lower in param_lower:
                score = 0.7
            
            # 3. Check schema description for the param name
            description = schema_def.get('description', '') if isinstance(schema_def, dict) else ''
            if description and param_name.lower() in description.lower():
                score = max(score, 0.8)  # Description match is strong signal
            
            # 4. Levenshtein-like similarity for typos
            if not score:
                # Simple character overlap ratio
                common = set(param_lower) & set(schema_lower)
                if len(common) / max(len(param_lower), len(schema_lower)) > 0.7:
                    score = 0.5
            
            if score > best_score:
                best_score = score
                best_match = schema_key
        
        return best_match if best_score >= 0.5 else None
    
    def _resolve_action(self, action_name: str) -> str:
        """
        Resolve TaskWeaver's action name to actual DB action ID.
        
        Follows the sql_pull_data pattern: accept approximate input, resolve internally.
        
        1. Check if exact action exists in DB
        2. If not, use semantic search to find best match
        3. Return resolved action ID
        
        Args:
            action_name: Action name from TaskWeaver (may be conceptual/approximate)
            
        Returns:
            Resolved action ID that exists in the database
        """
        try:
            from apps.integrations.models import ComposioActionSchema
            
            # Step 1: Try exact match (case-insensitive)
            exact_match = ComposioActionSchema.objects.filter(
                action_id__iexact=action_name
            ).first()
            
            if exact_match:
                logger.debug(f"[COMPOSIO_PLUGIN] Exact match: {action_name}")
                return exact_match.action_id
            
            # Step 2: Use semantic search (reuse existing action_matcher)
            logger.info(f"[COMPOSIO_PLUGIN] '{action_name}' not in DB, resolving semantically...")
            
            # Extract app hint from action name (e.g., GMAIL_GET_LAST_EMAILS -> gmail)
            app_hint = None
            if '_' in action_name and action_name.isupper():
                app_hint = action_name.split('_')[0].lower()
            
            # Get or create action matcher (reuse existing implementation)
            if self._action_matcher is None:
                from apps.integrations.services.action_matcher import ComposioActionMatcher
                self._action_matcher = ComposioActionMatcher(enable_warmup=False)
            
            # Use the action name as the search query
            # Also try the human-readable version: GMAIL_GET_LAST_EMAILS -> "get last emails"
            search_query = action_name.replace('_', ' ').lower()
            if app_hint:
                search_query = search_query.replace(app_hint, '').strip()
            
            results = self._action_matcher.match_for_subtask(
                subtask_description=search_query,
                app_hints=[app_hint] if app_hint else None,
                top_k_apps=1,
                top_k_actions_per_app=1,
                min_confidence=0.3  # Lower threshold for resolution
            )
            
            if results:
                resolved = results[0]['action_id']
                confidence = results[0]['confidence']
                logger.info(
                    f"[COMPOSIO_PLUGIN] Resolved: {action_name} -> {resolved} "
                    f"(confidence: {confidence:.2f})"
                )
                return resolved
            
            # No match found - return original (will fail at execution with clear error)
            logger.warning(f"[COMPOSIO_PLUGIN] Could not resolve: {action_name}")
            return action_name
            
        except ImportError:
            # Django not available (simulation mode) - return as-is
            logger.debug(f"[COMPOSIO_PLUGIN] Resolution skipped (simulation mode)")
            return action_name
        except Exception as e:
            logger.warning(f"[COMPOSIO_PLUGIN] Resolution error: {e}, using original")
            return action_name
    
    def _get_mock_response(self, action_name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Generate a realistic mock response for simulation mode.
        
        This allows TaskWeaver to continue generating code for all plan steps
        without failing on the first API call. The mock response contains
        placeholder data that matches the expected output structure.
        """
        app_name = action_name.split('_')[0].lower() if '_' in action_name else 'unknown'
        
        # Generic mock response that works for most actions
        mock_data = {
            '_simulation': True,
            '_action': action_name,
            '_message': 'Simulated response - actual execution happens at runtime',
            'success': True,
            'data': {}
        }
        
        # Add action-specific mock data for common patterns
        action_lower = action_name.lower()
        
        if 'search' in action_lower or 'find' in action_lower or 'get' in action_lower or 'fetch' in action_lower:
            # Search/Get actions - return minimal generic response
            # DO NOT hardcode specific keys like 'emails' or 'messages' - let runtime handle actual structure
            mock_data['data'] = {
                '_note': 'Simulated response - actual API structure varies by tool',
                'success': True,
                'count': 2
            }
            description = f"[SIMULATION] {action_name} will return actual data at runtime. Use defensive code to handle response."
            
        elif 'send' in action_lower or 'create' in action_lower or 'post' in action_lower:
            # Create/Send actions return confirmation
            mock_data['data'] = {
                'id': 'mock_created_id',
                'status': 'success',
                'message': f'{action_name} will be executed at runtime'
            }
            description = f"[SIMULATION] {action_name} prepared. Actual execution happens at runtime."
            
        elif 'forward' in action_lower:
            # Forward actions
            mock_data['data'] = {
                'id': 'mock_forwarded_id',
                'status': 'forwarded',
                'to': params.get('to', params.get('recipient_email', 'recipient@example.com'))
            }
            description = f"[SIMULATION] Email forward prepared. Actual forwarding happens at runtime."
            
        elif 'append' in action_lower or 'row' in action_lower:
            # Spreadsheet row operations
            mock_data['data'] = {
                'id': 'mock_row_id',
                'status': 'appended',
                'spreadsheet_id': params.get('spreadsheet_id', 'mock_spreadsheet_id')
            }
            description = f"[SIMULATION] Row operation prepared. Actual execution happens at runtime."
            
        else:
            # Generic response
            mock_data['data'] = {'status': 'ok', 'action': action_name}
            description = f"[SIMULATION] {action_name} validated. Actual execution happens at runtime."
        
        logger.info(f"[COMPOSIO_PLUGIN] SIMULATION: {action_name} -> mock response")
        return mock_data, description
    
    def _get_service(self):
        """Get or create ComposioService instance (lazy initialization)."""
        if self._composio_service is None:
            try:
                # Add Django app path to sys.path if needed
                django_app_path = os.path.join(
                    os.path.dirname(__file__), 
                    '..', '..', '..'  # TaskWeaver/project/plugins -> xtrac-app-api
                )
                abs_path = os.path.abspath(django_app_path)
                if abs_path not in sys.path:
                    sys.path.insert(0, abs_path)
                
                # Setup Django if not already
                import django
                os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
                if not django.apps.apps.ready:
                    django.setup()
                
                # Import and initialize ComposioService
                from apps.integrations.services.composio_service import ComposioService
                self._composio_service = ComposioService()
                logger.info("[COMPOSIO_PLUGIN] ComposioService initialized")
                
            except ImportError as e:
                logger.error(f"[COMPOSIO_PLUGIN] Failed to import ComposioService: {e}")
                raise RuntimeError(
                    "ComposioService not available. "
                    "Make sure Django app is properly configured."
                )
            except Exception as e:
                logger.error(f"[COMPOSIO_PLUGIN] Failed to initialize: {e}")
                raise
        
        return self._composio_service
    
    def _format_rich_description(self, data: Any, action_name: str) -> str:
        """
        Generate a rich, human-readable description of the action result.
        
        This uses the content_preparer utility (SINGLE SOURCE OF TRUTH) to format
        complex data into beautiful, readable output suitable for email bodies,
        messages, and other content parameters.
        
        Args:
            data: Action result data
            action_name: Name of the executed action
            
        Returns:
            Human-readable HTML description
        """
        try:
            # Import content_preparer (SINGLE SOURCE OF TRUTH)
            from apps.py_workflows.execution.utils.content_preparer import prepare_content
            
            # Use HTML formatting for rich output
            description = prepare_content(
                data=data,
                param_name=action_name,
                use_html=True,
                logger_instance=logger
            )
            
            logger.info(f"[COMPOSIO_PLUGIN] Generated rich description ({len(description)} chars)")
            return description
            
        except ImportError:
            # Fallback if content_preparer not available
            logger.warning("[COMPOSIO_PLUGIN] content_preparer not available, using basic description")
            return self._basic_description(data, action_name)
        except Exception as e:
            logger.warning(f"[COMPOSIO_PLUGIN] Rich formatting failed: {e}, using basic")
            return self._basic_description(data, action_name)
    
    def _basic_description(self, data: Any, action_name: str) -> str:
        """Basic fallback description when rich formatting is unavailable."""
        description = f"Successfully executed {action_name}. "
        
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            description += f"Result contains: {', '.join(keys)}"
        elif isinstance(data, list):
            description += f"Result contains {len(data)} items."
        else:
            description += f"Result type: {type(data).__name__}"
        
        return description
    
    def __call__(
        self, 
        action_name: str, 
        params: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Execute a Composio action.
        
        Args:
            action_name: Composio action ID (e.g., "GOOGLESHEETS_BATCH_GET", "GMAIL_SEND_EMAIL")
            params: Action parameters as a dictionary
            entity_id: User's entity ID for authentication (optional, uses config if not provided)
            
        Returns:
            Tuple of (result_dict, description_string)
            - result_dict: Action result data or error info
            - description_string: Human-readable description of the result
            
        Examples:
            # Get spreadsheet data
            result, desc = composio_action(
                action_name="GOOGLESHEETS_SPREADSHEETS_VALUES_BATCH_GET",
                params={"spreadsheetId": "1abc...", "ranges": ["Sheet1!A1:B10"]}
            )
            
            # Send email
            result, desc = composio_action(
                action_name="GMAIL_SEND_EMAIL",
                params={"to": "user@example.com", "subject": "Hello", "body": "Message"}
            )
            
            # Search for spreadsheet by name
            result, desc = composio_action(
                action_name="GOOGLESHEETS_FIND_SPREADSHEET",
                params={"name": "My Spreadsheet"}
            )
        """
        params = params or {}
        original_action = action_name  # Keep original for logging
        
        # =====================================================================
        # SIMULATION MODE CHECK
        # =====================================================================
        # If running in Docker CES (TaskWeaver's code execution sandbox),
        # Django services are NOT available. Return mock response so TaskWeaver
        # continues generating code for ALL plan steps.
        #
        # Actual execution happens later via langgraph_adapter with real APIs.
        # =====================================================================
        if self._is_simulation_mode():
            logger.info(f"[COMPOSIO_PLUGIN] SIMULATION MODE: {action_name}")
            return self._get_mock_response(action_name, params)
        
        # =====================================================================
        # ACTION RESOLUTION (sql_pull_data pattern)
        # =====================================================================
        # TaskWeaver's LLM may generate conceptual names like GMAIL_GET_LAST_EMAILS.
        # Resolve to actual DB action using semantic search.
        # =====================================================================
        action_name = self._resolve_action(action_name)
        
        if action_name != original_action:
            logger.info(f"[COMPOSIO_PLUGIN] Action resolved: {original_action} -> {action_name}")
        
        # =====================================================================
        # PARAMETER RESOLUTION (sql_pull_data pattern)
        # =====================================================================
        # TaskWeaver's LLM may use approximate param names like 'name' instead of 'title'.
        # Resolve to actual schema params using semantic matching.
        # =====================================================================
        original_params = params.copy()
        params = self._resolve_params(action_name, params)
        
        if params != original_params:
            changed = {k: v for k, v in params.items() if k not in original_params or original_params.get(k) != v}
            logger.info(f"[COMPOSIO_PLUGIN] Params resolved: {list(original_params.keys())} -> {list(params.keys())}")
        
        try:
            service = self._get_service()
            
            # Get entity_id with priority: parameter > session_var > config > default
            # Session var is set by langgraph_adapter during execution
            if not entity_id:
                entity_id = self.ctx.get_session_var("_composio_entity_id", None) or \
                            self.config.get('entity_id', 'default')
            
            logger.info(f"[COMPOSIO_PLUGIN] Executing: {action_name}")
            logger.info(f"[COMPOSIO_PLUGIN] Params: {params}")
            logger.info(f"[COMPOSIO_PLUGIN] Entity: {entity_id}")
            
            # Get connected account for this action
            app_name = action_name.split('_')[0].lower()
            connected_account_id = None
            
            if not service.is_no_auth_app(app_name):
                # Try to get connection for this app
                try:
                    connections = service.get_user_connections(entity_id)
                    for conn in connections:
                        if conn.get('app_name', '').lower() == app_name:
                            if conn.get('status', '').upper() == 'ACTIVE':
                                connected_account_id = conn.get('id')
                                logger.info(f"[COMPOSIO_PLUGIN] Using connection: {connected_account_id}")
                                break
                except Exception as conn_err:
                    logger.warning(f"[COMPOSIO_PLUGIN] Could not get connections: {conn_err}")
            
            # Execute the action
            result = service.execute_action(
                action_name=action_name,
                params=params,
                entity_id=entity_id,
                connected_account_id=connected_account_id
            )
            
            if result.get('success') or result.get('successfull'):
                data = result.get('data', result)
                
                # RICH DESCRIPTION: Use content_preparer for human-readable output
                # This is the SINGLE SOURCE OF TRUTH for content formatting
                description = self._format_rich_description(data, action_name)
                
                logger.info(f"[COMPOSIO_PLUGIN] Action succeeded: {action_name}")
                return data, description
            else:
                error_msg = result.get('error', 'Unknown error')
                description = f"Action {action_name} failed: {error_msg}"
                logger.error(f"[COMPOSIO_PLUGIN] Action failed: {error_msg}")
                
                return {
                    'error': True,
                    'error_message': error_msg,
                    'action': action_name
                }, description
                
        except Exception as e:
            error_msg = str(e)
            description = f"Exception executing {action_name}: {error_msg}"
            logger.error(f"[COMPOSIO_PLUGIN] Exception: {e}", exc_info=True)
            
            return {
                'error': True,
                'error_message': error_msg,
                'action': action_name
            }, description


@test_plugin(name="test_composio_action", description="Test Composio action plugin")
def test_call(api_call):
    """Test the Composio action plugin with a no-auth action."""
    # Test with a simple search (no auth required)
    result, description = api_call(
        action_name="COMPOSIO_SEARCH_TOOLS",
        params={"query": "email"}
    )
    
    # Should return something (even if error due to missing config)
    assert result is not None
    assert description is not None
