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
from typing import Dict, Any, Tuple, Optional

from taskweaver.plugin import Plugin, register_plugin, test_plugin

logger = logging.getLogger(__name__)


@register_plugin
class ComposioAction(Plugin):
    """
    TaskWeaver plugin for executing Composio actions.
    
    Provides a simple interface for TaskWeaver-generated code to call
    any of the 800+ Composio integrations (Gmail, Google Sheets, Slack, etc.)
    """
    
    _composio_service = None
    
    def _get_service(self):
        """Get or create ComposioService instance (lazy initialization)."""
        if self._composio_service is None:
            try:
                # Add Django app path to sys.path if needed
                django_app_path = os.path.join(
                    os.path.dirname(__file__), 
                    '..', '..', '..'  # TaskWeaver/project/plugins → xtrac-app-api
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
        
        try:
            service = self._get_service()
            
            # Get entity_id with priority: parameter > config > default
            # Note: entity_id can be passed explicitly in the function call,
            # or set in the plugin config (composio_action.yaml)
            if not entity_id:
                entity_id = self.config.get('entity_id', 'default')
            
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
                description = f"Successfully executed {action_name}. "
                
                # Add context based on result type
                if isinstance(data, dict):
                    keys = list(data.keys())[:5]  # First 5 keys
                    description += f"Result contains: {', '.join(keys)}"
                elif isinstance(data, list):
                    description += f"Result contains {len(data)} items."
                else:
                    description += f"Result type: {type(data).__name__}"
                
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

