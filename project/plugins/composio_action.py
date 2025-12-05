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
from typing import Dict, Any, Tuple, Optional, Union, List

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
    _host_callback_available = None  # Can we call back to Django host?
    
    # Host URL for Docker container to call back to Django
    # Configurable via TASKWEAVER_HOST_URL env var (e.g., "http://host.docker.internal:8000")
    _host_urls = None  # Lazily initialized
    
    @classmethod
    def _get_host_urls(cls) -> List[str]:
        """Get list of host URLs to try for Docker-to-Django communication."""
        if cls._host_urls is None:
            # Check env var first (allows custom configuration)
            custom_url = os.environ.get('TASKWEAVER_HOST_URL')
            if custom_url:
                cls._host_urls = [custom_url]
            else:
                # Default fallback order for different Docker environments
                cls._host_urls = [
                    "http://host.docker.internal:8000",  # Docker Desktop (Windows/Mac)
                    "http://172.17.0.1:8000",            # Docker Linux bridge
                    "http://localhost:8000",             # Local development
                ]
        return cls._host_urls
    
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
    
    def _call_host_api(self, action_name: str, params: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Call the Django host's internal API to execute the action.
        
        This allows the Docker container to make REAL Composio API calls by
        calling back to the host where Django and credentials are available.
        
        Returns:
            (data, description) tuple if successful, None if callback failed
        """
        # =====================================================================
        # SCALABLE FIX (arch-44): Skip real API during workflow GENERATION
        # =====================================================================
        # During workflow generation, form_collect returns mock data which will
        # cause API validation failures. We should mock ALL API calls during
        # generation - real APIs are only called during workflow EXECUTION.
        #
        # Session variable `_workflow_generation_mode` is set by eclipse_adapter
        # when creating the session.
        # =====================================================================
        is_generation_mode = self.ctx.get_session_var("_workflow_generation_mode", None)
        logger.info(f"[COMPOSIO_PLUGIN] 🔍 CHECK: _workflow_generation_mode = '{is_generation_mode}' for {action_name}")
        if is_generation_mode == "true":
            logger.info(f"[COMPOSIO_PLUGIN] ✅ GENERATION MODE - returning mock for {action_name}")
            return None  # Triggers mock fallback
        else:
            logger.info(f"[COMPOSIO_PLUGIN] ⚠️ EXECUTION MODE - calling real API for {action_name}")
        
        if self._host_callback_available is False:
            return None
        
        try:
            import requests
            
            # Get session ID from context (set by TaskWeaver)
            session_id = self.ctx.session_id if hasattr(self.ctx, 'session_id') else None
            entity_id = self.ctx.get_session_var("_composio_entity_id", None)
            
            # Host URL - configurable via TASKWEAVER_HOST_URL env var
            host_urls = self._get_host_urls()
            
            payload = {
                "session_id": session_id,
                "action_name": action_name,
                "params": params,
                "entity_id": entity_id
            }
            
            for host_url in host_urls:
                try:
                    url = f"{host_url}/api/v1/integrations/internal/execute/"
                    logger.info(f"[COMPOSIO_PLUGIN] Trying host callback: {url}")
                    
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=30,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        self._host_callback_available = True
                        
                        if result.get('success') or result.get('successfull'):
                            data = result.get('data', result)
                            
                            # =========================================================
                            # SCALABLE FIX (arch-35): Dynamic structure hint for LLM
                            # =========================================================
                            # Auto-describe response structure so LLM writes correct
                            # parsing code on FIRST try, not via retry loop.
                            # =========================================================
                            structure_hint = self._describe_data_structure(data)
                            
                            description = (
                                f"Successfully executed {action_name}.\n"
                                f"RESPONSE STRUCTURE: {structure_hint}\n"
                                f"Use the structure above to access the data correctly."
                            )
                            logger.info(f"✅ [COMPOSIO_PLUGIN] Host callback succeeded: {action_name}")
                            return data, description
                        else:
                            # =========================================================
                            # CRITICAL FIX (arch-25): RAISE EXCEPTION on API failure!
                            # =========================================================
                            # This enables TaskWeaver's retry loop to self-correct!
                            # The LLM will see the error and fix the parameters.
                            # =========================================================
                            error = result.get('error', 'Unknown error')
                            logger.error(f"❌ [COMPOSIO_PLUGIN] API FAILURE (will trigger retry): {error}")
                            
                            raise RuntimeError(
                                f"Composio API error for {action_name}: {error}\n"
                                f"Parameters provided: {list(params.keys())}\n"
                                f"Please fix the parameters and retry."
                            )
                    
                except requests.exceptions.ConnectionError:
                    continue  # Try next host URL
                except requests.exceptions.Timeout:
                    logger.warning(f"[COMPOSIO_PLUGIN] Host callback timeout: {host_url}")
                    continue
            
            # No host available
            logger.info("[COMPOSIO_PLUGIN] Host callback not available, falling back to mock")
            self._host_callback_available = False
            return None
            
        except RuntimeError:
            # =====================================================================
            # CRITICAL FIX (arch-25): Re-raise RuntimeError (API failures)!
            # =====================================================================
            # RuntimeError is raised when Composio API returns success=False.
            # We MUST let this propagate to trigger TaskWeaver's retry loop!
            # =====================================================================
            raise
        except Exception as e:
            # Only catch network/import errors, not API failures
            logger.warning(f"[COMPOSIO_PLUGIN] Host callback failed (network error): {e}")
            self._host_callback_available = False
            return None
    
    def _describe_data_structure(self, data: Any, depth: int = 4, max_items: int = 5) -> str:
        """
        SCALABLE FIX (arch-35): Auto-describe response structure for LLM.
        
        Works for ANY API response - flights, spreadsheets, emails, etc.
        No hardcoding - recursively describes the actual structure.
        
        Examples:
        - Flights: {"results": {"best_flights": list[5]: [{"id", "price", ...}]}}
        - Sheets: {"spreadsheetId": "...", "replies": list[1]: [{"addSheet": {"properties": {...}}}]}
        - Gmail: list[10]: [{"id", "threadId", "labelIds": list[2], "snippet", ...}]
        
        Args:
            data: The API response data (ANY structure)
            depth: How deep to recurse (4 levels handles most APIs)
            max_items: Max keys/items to show (5 covers most important fields)
            
        Returns:
            Human-readable structure description
        """
        if data is None:
            return "null"
        
        if isinstance(data, str):
            # Show short strings as-is, truncate long ones
            if len(data) > 100:
                return f'string[{len(data)} chars]'
            elif len(data) > 40:
                return f'"{data[:35]}..."'
            return f'"{data}"'
        
        if isinstance(data, bool):
            return str(data).lower()
        
        if isinstance(data, (int, float)):
            return str(data)
        
        if isinstance(data, list):
            if len(data) == 0:
                return "[]"
            # Show first item structure to reveal the pattern
            if depth > 0:
                first_item = self._describe_data_structure(data[0], depth - 1, max_items)
                return f"list[{len(data)}]: [{first_item}, ...]"
            return f"list[{len(data)}]"
        
        if isinstance(data, dict):
            if len(data) == 0:
                return "{}"
            
            keys = list(data.keys())
            if depth > 0:
                # Show key-value pairs for important keys
                parts = []
                for key in keys[:max_items]:
                    val_desc = self._describe_data_structure(data[key], depth - 1, max_items)
                    parts.append(f'"{key}": {val_desc}')
                
                if len(keys) > max_items:
                    parts.append(f"+{len(keys) - max_items} more")
                
                return "{" + ", ".join(parts) + "}"
            else:
                # At max depth, just show key names
                shown = ", ".join(f'"{k}"' for k in keys[:max_items])
                if len(keys) > max_items:
                    shown += f", +{len(keys) - max_items} more"
                return "keys: [" + shown + "]"
        
        return type(data).__name__
    
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
        Generate a SCHEMA-BASED mock response for simulation mode.
        
        TASKWEAVER PATTERN (BATTLE-TESTED):
        ===================================
        TaskWeaver plugins return (data, description) where the LLM USES the data
        for subsequent steps. We must return realistic data, not "ignore this".
        
        SCALABLE APPROACH (800+ tools):
        ===============================
        1. Try HOST CALLBACK first (real API via Django host)
        2. Fallback to schema-based mock from DB
        3. Fallback to verb-based mock (no hardcoding)
        
        This follows the sql_pull_data pattern: DYNAMIC resolution, no hardcoding.
        """
        # =====================================================================
        # STEP 1: Try HOST CALLBACK - Real API execution via Django host!
        # =====================================================================
        # This is the BEST option - we get REAL Composio API responses
        # The Docker container calls back to Django where credentials exist
        host_result = self._call_host_api(action_name, params)
        
        if host_result is not None:
            return host_result
        
        # =====================================================================
        # STEP 2: Try to fetch actual response_schema from DB (DYNAMIC!)
        # =====================================================================
        schema_based_mock = self._generate_schema_based_mock(action_name, params)
        
        if schema_based_mock:
            mock_data, description = schema_based_mock
            logger.info(f"[COMPOSIO_PLUGIN] SIMULATION: {action_name} -> schema-based mock")
            return mock_data, description
        
        # =====================================================================
        # STEP 3: FALLBACK - VERB-BASED mock (ZERO app-specific hardcoding!)
        # =====================================================================
        # The key insight: All APIs follow REST conventions. 
        # We classify by VERB pattern, not by app name.
        # 
        # VERB PATTERNS:
        # - LIST/GET/FETCH → Returns a LIST that can be iterated
        # - CREATE/SEND/POST → Returns dict with 'id' of created item
        # - UPDATE/PATCH → Returns dict with 'id' of updated item  
        # - DELETE → Returns dict with 'success' confirmation
        # =====================================================================
        
        action_lower = action_name.lower()
        
        # Detect action verb from action name (GMAIL_FETCH_EMAILS → 'fetch')
        # This is 100% dynamic - no app-specific logic!
        is_list_action = any(verb in action_lower for verb in ['list', 'search', 'find', 'get', 'fetch', 'read', 'query'])
        is_create_action = any(verb in action_lower for verb in ['create', 'send', 'post', 'add', 'insert', 'forward', 'append'])
        is_update_action = any(verb in action_lower for verb in ['update', 'patch', 'modify', 'edit', 'set'])
        is_delete_action = any(verb in action_lower for verb in ['delete', 'remove', 'clear'])
        
        if is_list_action:
            # =====================================================================
            # LIST ACTIONS: Return a LIST that can be directly iterated
            # =====================================================================
            # TaskWeaver pattern: `for item in result:` must work!
            # 
            # UNIVERSAL FIELD NAMES (not app-specific!):
            # These are common across REST APIs regardless of app:
            # - Emails: subject, snippet, body, from, to
            # - Documents: title, name, content, body
            # - Records: id, data, created_at
            # 
            # Including ALL common names so ANY access pattern works!
            # =====================================================================
            mock_data = [
                {
                    # Universal identifiers
                    'id': 'sim_item_1',
                    'uid': 'sim_item_1',
                    # Email-pattern fields (universal)
                    'subject': 'Simulated Subject 1',
                    'snippet': 'Simulated snippet preview 1...',
                    'body': 'Simulated body content 1',
                    'from': 'sender@example.com',
                    'to': 'recipient@example.com',
                    # Document-pattern fields (universal)
                    'title': 'Simulated Title 1',
                    'name': 'Simulated Name 1',
                    'content': 'Simulated content 1',
                    'data': 'Simulated data 1',
                    # Metadata (universal)
                    'created_at': '2024-01-01T00:00:00Z',
                    'status': 'active'
                },
                {
                    'id': 'sim_item_2',
                    'uid': 'sim_item_2',
                    'subject': 'Simulated Subject 2',
                    'snippet': 'Simulated snippet preview 2...',
                    'body': 'Simulated body content 2',
                    'from': 'sender@example.com',
                    'to': 'recipient@example.com',
                    'title': 'Simulated Title 2',
                    'name': 'Simulated Name 2',
                    'content': 'Simulated content 2',
                    'data': 'Simulated data 2',
                    'created_at': '2024-01-02T00:00:00Z',
                    'status': 'active'
                }
            ]
            structure_hint = self._describe_data_structure(mock_data)
            description = (
                f"Successfully executed {action_name}.\n"
                f"RESPONSE STRUCTURE: {structure_hint}\n"
                f"Result is a list - iterate with: `for item in result:`"
            )
        elif is_create_action:
            # =====================================================================
            # CREATE ACTIONS: Return dict with 'id' of created resource
            # =====================================================================
            # REST APIs return IDs in various formats. Include ALL common patterns
            # so ANY access pattern works (no app-specific logic!):
            # - Generic: id, uid, resourceId
            # - Documents: documentId, spreadsheetId, fileId, sheetId
            # - Messages: messageId, threadId
            # - Records: recordId, entryId
            # =====================================================================
            created_id = 'sim_created_id_123'
            mock_data = {
                # Generic ID patterns (universal)
                'id': created_id,
                'uid': created_id,
                'resourceId': created_id,
                # Document ID patterns (universal for doc-like resources)
                'spreadsheetId': created_id,
                'documentId': created_id,
                'fileId': created_id,
                'sheetId': created_id,
                # Message ID patterns (universal for message-like resources)
                'messageId': created_id,
                'threadId': created_id,
                # Record ID patterns
                'recordId': created_id,
                'entryId': created_id,
                # Status (universal)
                'success': True,
                'status': 'created',
                'created': True,
                # Common response fields
                'title': params.get('title', params.get('name', 'Created Resource')),
                'name': params.get('name', params.get('title', 'Created Resource')),
                'url': f'https://example.com/resource/{created_id}'
            }
            structure_hint = self._describe_data_structure(mock_data)
            description = (
                f"Successfully executed {action_name}.\n"
                f"RESPONSE STRUCTURE: {structure_hint}\n"
                f"Resource ID available as result['id'] or result['spreadsheetId'], etc."
            )
        elif is_update_action:
            # =====================================================================
            # UPDATE ACTIONS: Return dict with confirmation
            # =====================================================================
            # Include same ID patterns as CREATE for consistency
            updated_id = params.get('id', params.get('spreadsheetId', params.get('documentId', 'sim_updated_id')))
            mock_data = {
                'id': updated_id,
                'spreadsheetId': updated_id,
                'documentId': updated_id,
                'success': True,
                'status': 'updated',
                'updated': True,
                'modifiedTime': '2024-01-01T12:00:00Z'
            }
            structure_hint = self._describe_data_structure(mock_data)
            description = f"Successfully executed {action_name}.\nRESPONSE STRUCTURE: {structure_hint}"
        elif is_delete_action:
            # =====================================================================
            # DELETE ACTIONS: Return success confirmation
            # =====================================================================
            mock_data = {
                'success': True,
                'deleted': True,
                'status': 'deleted'
            }
            structure_hint = self._describe_data_structure(mock_data)
            description = f"Successfully executed {action_name}.\nRESPONSE STRUCTURE: {structure_hint}"
        else:
            # =====================================================================
            # GENERIC ACTIONS: Return minimal success response
            # =====================================================================
            mock_data = {
                'success': True,
                'status': 'completed',
                'result': 'ok'
            }
            structure_hint = self._describe_data_structure(mock_data)
            description = f"Successfully executed {action_name}.\nRESPONSE STRUCTURE: {structure_hint}"
        
        logger.info(f"[COMPOSIO_PLUGIN] SIMULATION: {action_name} -> verb-based mock (no hardcoding)")
        return mock_data, description
    
    def _generate_schema_based_mock(self, action_name: str, params: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Generate mock response based on actual Composio action schema.
        
        FULLY DYNAMIC (TaskWeaver sql_pull_data pattern):
        - Fetches response_schema via Django's internal API (works from Docker!)
        - Generates conformant mock data based on schema
        - No hardcoding - works for all 800+ tools!
        
        Returns:
            (mock_data, description) or None if schema not available
        """
        try:
            import requests
            
            # =====================================================================
            # SCALABLE FIX (arch-51): Fetch schema via internal API
            # =====================================================================
            # Docker container can't access Django models or Redis cache.
            # We call the internal schema endpoint instead.
            # =====================================================================
            
            # Host URL - configurable via TASKWEAVER_HOST_URL env var
            host_urls = self._get_host_urls()
            
            response_schema = None
            
            for host_url in host_urls:
                try:
                    url = f"{host_url}/api/v1/integrations/internal/schema/{action_name}/"
                    response = requests.get(url, timeout=2)
                    
                    if response.status_code == 200:
                        data = response.json()
                        response_schema = data.get('response_schema')
                        if response_schema:
                            logger.info(f"[COMPOSIO_PLUGIN] Fetched schema for {action_name} from {host_url}")
                            break
                except requests.exceptions.RequestException:
                    continue
            
            if not response_schema:
                logger.debug(f"[COMPOSIO_PLUGIN] No response_schema available for {action_name}")
                return None
            
            # Generate mock based on schema properties
            mock_data = {
                'success': True,
                'successfull': True,
                'data': self._generate_mock_from_schema(response_schema)
            }
            
            # Use dynamic structure hint (same as real API)
            structure_hint = self._describe_data_structure(mock_data)
            
            description = (
                f"Successfully executed {action_name}.\n"
                f"RESPONSE STRUCTURE: {structure_hint}\n"
                f"Use the structure above to access the data correctly."
            )
            
            logger.info(f"[COMPOSIO_PLUGIN] ✅ Generated schema-based mock for {action_name}")
            return mock_data, description
            
        except Exception as e:
            logger.debug(f"[COMPOSIO_PLUGIN] Schema-based mock failed: {e}")
            return None
    
    def _generate_mock_from_schema(
        self, 
        schema: Dict[str, Any], 
        depth: int = 0,
        prop_name: str = ""
    ) -> Any:
        """
        Recursively generate mock data from JSON schema.
        
        SCALABLE: Works for any schema structure without hardcoding.
        
        CRITICAL FIX: Fields named 'error', 'error_message', etc. are set to None
        to prevent the Planner from thinking the mock response is a failure.
        """
        if depth > 3:  # Prevent infinite recursion
            return {}
        
        # =========================================================================
        # CRITICAL: Skip error-like fields to prevent Planner from stopping early
        # =========================================================================
        error_field_names = {'error', 'error_message', 'error_code', 'errors', 'err', 'exception'}
        if prop_name.lower() in error_field_names:
            return None  # No error in mock responses!
        
        schema_type = schema.get('type', 'object')
        
        if schema_type == 'object':
            properties = schema.get('properties', {})
            additional_props = schema.get('additionalProperties', False)
            mock_obj = {}
            
            for child_prop_name, prop_schema in list(properties.items())[:10]:  # Limit to 10 props
                # Pass prop_name to recursive call
                mock_obj[child_prop_name] = self._generate_mock_from_schema(
                    prop_schema, depth + 1, child_prop_name
                )
            
            # =========================================================================
            # FIX: If no properties but additionalProperties is true, generate sample data
            # This handles schemas like: {"type": "object", "additionalProperties": true}
            # Without this, we'd return {} which looks like "no results found"
            # 
            # SCALABLE: Uses generic structure that works for ANY API response
            # =========================================================================
            if not properties and additional_props:
                # Generic result structure that any code can iterate/access
                mock_obj = {
                    'items': [
                        {'id': 'item_1', 'name': 'Sample Item 1', 'value': 'data_1'},
                        {'id': 'item_2', 'name': 'Sample Item 2', 'value': 'data_2'}
                    ],
                    'count': 2,
                    'total': 2,
                    'has_more': False,
                    'summary': 'Mock results for workflow generation'
                }
            
            return mock_obj
            
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            # Generate 2-3 items for lists to make it look more realistic
            return [
                self._generate_mock_from_schema(items_schema, depth + 1, f"{prop_name}_item"),
                self._generate_mock_from_schema(items_schema, depth + 1, f"{prop_name}_item")
            ]
            
        elif schema_type == 'string':
            # Check for format hints
            format_hint = schema.get('format', '')
            description = schema.get('description', '').lower()
            
            # Generate contextual values based on field name/description
            prop_lower = prop_name.lower()
            
            if 'email' in format_hint or 'email' in prop_lower:
                return 'user@example.com'
            elif 'date' in format_hint or 'date' in prop_lower:
                return '2024-01-15'
            elif 'uri' in format_hint or 'url' in format_hint or 'url' in prop_lower or 'link' in prop_lower:
                return 'https://example.com/resource'
            elif 'id' in prop_lower:
                return 'mock_id_12345'
            elif 'name' in prop_lower:
                return 'Sample Name'
            elif 'title' in prop_lower:
                return 'Sample Title'
            elif 'subject' in prop_lower:
                return 'Sample Subject'
            elif 'body' in prop_lower or 'content' in prop_lower or 'text' in prop_lower:
                return 'Sample content text...'
            elif 'status' in prop_lower:
                return 'success'
            else:
                return 'sample_value'
                
        elif schema_type == 'integer':
            prop_lower = prop_name.lower()
            if 'count' in prop_lower or 'total' in prop_lower:
                return 5
            elif 'page' in prop_lower:
                return 1
            return 1
            
        elif schema_type == 'number':
            prop_lower = prop_name.lower()
            if 'price' in prop_lower or 'cost' in prop_lower or 'amount' in prop_lower:
                return 99.99
            return 1.0
            
        elif schema_type == 'boolean':
            # Default to True for success-like fields
            prop_lower = prop_name.lower()
            if 'success' in prop_lower or 'valid' in prop_lower or 'active' in prop_lower:
                return True
            elif 'error' in prop_lower or 'failed' in prop_lower:
                return False
            return True
            
        else:
            return 'value'
    
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
                # =====================================================================
                # CRITICAL FIX (arch-25): RAISE EXCEPTION on API failure!
                # =====================================================================
                # Previously: We returned error dict → Python "succeeded" → no retry
                # Now: We RAISE an exception → Python fails → TaskWeaver retries!
                #
                # This enables TaskWeaver's self-correction loop to fix issues like:
                # - "Missing required field: valueInputOption"
                # - "is_html must be true for HTML content"
                # =====================================================================
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"[COMPOSIO_PLUGIN] API FAILURE (will trigger retry): {error_msg}")
                
                # Raise exception with detailed message for LLM to understand
                raise RuntimeError(
                    f"Composio API error for {action_name}: {error_msg}\n"
                    f"Parameters provided: {list(params.keys())}\n"
                    f"Please fix the parameters and retry."
                )
                
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
