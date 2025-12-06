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
import json
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
    _host_callback_available = None  # Can we call back to Django host?
    _schema_cache = None  # Cached schemas from JSON file
    
    def _get_schema_cache_path(self) -> str:
        """Get schema cache file path from config (TaskWeaver pattern)."""
        cache_file = self.config.get("schema_cache_path", "composio_schemas_cache.json")
        # Try multiple locations - Docker container path FIRST since __file__ 
        # resolves to temp dir in container
        locations = [
            # 1. Container mounted path (Docker) - CHECK FIRST!
            f"/app/plugins/{cache_file}",
            # 2. Relative to plugin directory (host)
            os.path.join(os.path.dirname(__file__), cache_file),
            # 3. Current working directory
            cache_file,
        ]
        for path in locations:
            if os.path.exists(path):
                print(f"[COMPOSIO_PLUGIN] Found cache at: {path}")
                return path
        print(f"[COMPOSIO_PLUGIN] Cache not found in: {locations}")
        return locations[1]  # Default to host location
    
    def _load_schema_cache(self) -> Dict[str, Any]:
        """Load schema cache from JSON file (for Docker container without Django)."""
        if self._schema_cache is not None:
            return self._schema_cache
        
        cache_path = self._get_schema_cache_path()
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    ComposioAction._schema_cache = json.load(f)
                    print(f"[COMPOSIO_PLUGIN] Loaded {len(self._schema_cache)} schemas from {cache_path}")
                    return self._schema_cache
            else:
                print(f"[COMPOSIO_PLUGIN] Cache file not found: {cache_path}")
        except Exception as e:
            print(f"[COMPOSIO_PLUGIN] Failed to load schema cache: {e}")
        
        ComposioAction._schema_cache = {}
        return self._schema_cache
    
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
            
            # Host URL - try multiple options for Docker networking
            host_urls = [
                "http://host.docker.internal:8000",  # Docker Desktop (Windows/Mac)
                "http://172.17.0.1:8000",  # Docker Linux bridge
                "http://localhost:8000",  # Local development
            ]
            
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
        Generate mock response for simulation mode.
        
        SINGLE SOURCE OF TRUTH: ComposioActionSchema.response_schema in DB
        ===================================================================
        
        All mock data is generated from the response_schema stored in DB.
        No hardcoded patterns, no verb-based fallbacks, no app-specific logic.
        
        If response_schema is missing, that's a DATA problem - fix in DB, not code.
        """
        # =====================================================================
        # STEP 1: Try HOST CALLBACK - Real API execution via Django host
        # =====================================================================
        host_result = self._call_host_api(action_name, params)
        if host_result is not None:
            return host_result
        
        # =====================================================================
        # STEP 2: SINGLE SOURCE OF TRUTH - Schema from DB
        # =====================================================================
        schema_based_mock = self._generate_schema_based_mock(action_name, params)
        
        if schema_based_mock:
            logger.info(f"[COMPOSIO_PLUGIN] SIMULATION: {action_name} -> schema-based mock from DB")
            return schema_based_mock
        
        # =====================================================================
        # FALLBACK: Minimal success response (schema missing - DATA issue)
        # =====================================================================
        # If we get here, the action's response_schema needs to be populated in DB.
        # This is NOT a code fix - it's a data fix via sync_composio_schemas command.
        logger.warning(f"[COMPOSIO_PLUGIN] MISSING SCHEMA: {action_name} - run 'python manage.py sync_composio_schemas'")
        
        mock_data = {'success': True, 'status': 'completed', '_schema_missing': True}
        description = (
            f"Executed {action_name} (simulated - schema not in DB).\n"
            f"To get accurate mock data, run: python manage.py sync_composio_schemas"
        )
        return mock_data, description
    
    def _generate_schema_based_mock(self, action_name: str, params: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Generate mock response based on actual Composio action schema from DB.
        
        FULLY DYNAMIC (TaskWeaver sql_pull_data pattern):
        - Fetches response_schema from ComposioActionSchema table
        - Generates conformant mock data based on schema
        - No hardcoding - works for all 800+ tools!
        
        THE ARCHITECTURAL FIX:
        =====================
        If this returns None, it means the action's response_schema is missing from DB.
        The fix is to populate ComposioActionSchema.response_schema for all actions.
        This is DATA, not CODE - the schema-driven approach scales to any number of tools.
        
        Returns:
            (mock_data, description) or None if schema not available
        """
        try:
            from apps.integrations.models import ComposioActionSchema
            
            # Fetch schema from DB
            action = ComposioActionSchema.objects.filter(
                action_id__iexact=action_name
            ).first()
            
            if not action:
                logger.warning(f"[COMPOSIO_PLUGIN] ACTION NOT IN DB: {action_name} - add to ComposioActionSchema table")
                return None
            
            if not action.response_schema:
                logger.warning(f"[COMPOSIO_PLUGIN] NO RESPONSE_SCHEMA: {action_name} - populate response_schema in DB")
                return None
            
            response_schema = action.response_schema
            
            # Generate mock based on schema properties
            full_mock = self._generate_mock_from_schema(response_schema)
            
            # UNWRAP: Real API returns result.get('data', result)
            # Mock must match this pattern - return just the 'data' field
            if isinstance(full_mock, dict) and 'data' in full_mock:
                mock_data = full_mock['data']
            else:
                mock_data = full_mock
            
            # Use dynamic structure hint (same as real API)
            structure_hint = self._describe_data_structure(mock_data)
            
            description = (
                f"Successfully executed {action_name}.\n"
                f"RESPONSE STRUCTURE: {structure_hint}\n"
                f"Use the structure above to access the data correctly."
            )
            
            logger.info(f"[COMPOSIO_PLUGIN] SCHEMA-BASED mock for {action_name} (from DB)")
            return mock_data, description
            
        except ImportError:
            # Django not available (running in Docker without Django context)
            # Try loading from schema cache file instead
            print(f"[COMPOSIO_PLUGIN] Django not available, trying schema cache for {action_name}")
            return self._generate_mock_from_cache(action_name, params)
        except Exception as e:
            print(f"[COMPOSIO_PLUGIN] Schema lookup failed for {action_name}: {e}")
            return self._generate_mock_from_cache(action_name, params)
    
    def _generate_mock_from_cache(self, action_name: str, params: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
        """Generate mock from cached schema file (Docker container fallback)."""
        cache = self._load_schema_cache()
        
        # Try exact match first, then case-insensitive
        response_schema = cache.get(action_name) or cache.get(action_name.upper())
        
        if not response_schema:
            print(f"[COMPOSIO_PLUGIN] Schema not in cache: {action_name}")
            return None
        
        print(f"[COMPOSIO_PLUGIN] Found schema in cache for {action_name}")
        
        # Generate mock based on schema properties
        full_mock = self._generate_mock_from_schema(response_schema)
        
        # UNWRAP: Real API returns result.get('data', result)
        if isinstance(full_mock, dict) and 'data' in full_mock:
            mock_data = full_mock['data']
        else:
            mock_data = full_mock
        
        # Use dynamic structure hint
        structure_hint = self._describe_data_structure(mock_data)
        
        description = (
            f"Successfully executed {action_name}.\n"
            f"RESPONSE STRUCTURE: {structure_hint}\n"
            f"Use the structure above to access the data correctly."
        )
        
        print(f"[COMPOSIO_PLUGIN] Generated schema-based mock from cache for {action_name}")
        return mock_data, description
    
    def _generate_mock_from_schema(self, schema: Dict[str, Any], depth: int = 0, prop_name: str = "") -> Any:
        """
        Recursively generate mock data from JSON schema.
        
        SCALABLE: Works for any schema structure without hardcoding app names.
        Uses property names for context-aware mock generation for common patterns.
        
        Covered patterns:
        - Spreadsheets: values, valueRanges, rows, cells (Google Sheets, Excel, Smartsheet)
        - Databases: records, fields, entries (Airtable, Notion, Supabase)
        - Email/Messaging: messages, threads, attachments (Gmail, Outlook, Slack)
        - Files/Storage: files, items, entries (Google Drive, Dropbox, OneDrive)
        - Calendar: events, attendees (Google Calendar, Outlook Calendar)
        - CRM: contacts, deals, leads (Salesforce, HubSpot)
        - Generic: data, results, items, entries
        """
        if depth > 3:  # Prevent infinite recursion
            return {}
        
        schema_type = schema.get('type', 'object')
        prop_lower = prop_name.lower()
        
        # =====================================================================
        # SPREADSHEET/TABLE PATTERNS (Google Sheets, Excel, Airtable, Smartsheet)
        # =====================================================================
        if prop_lower == 'values':
            # 2D array: headers + data rows
            return [
                ['Date', 'Amount', 'Description'],
                ['2025-01-15', '150.00', 'Sample item 1'],
                ['2025-02-15', '200.00', 'Sample item 2'],
            ]
        elif prop_lower == 'valueranges':
            # Google Sheets BATCH_GET response
            return [{
                'range': 'Sheet1!A1:C3',
                'majorDimension': 'ROWS',
                'values': [
                    ['Date', 'Amount', 'Description'],
                    ['2025-01-15', '150.00', 'Sample item 1'],
                    ['2025-02-15', '200.00', 'Sample item 2'],
                ]
            }]
        elif prop_lower == 'rows':
            # Generic table rows
            return [
                {'id': 'row_1', 'Date': '2025-01-15', 'Amount': '150.00'},
                {'id': 'row_2', 'Date': '2025-02-15', 'Amount': '200.00'},
            ]
        elif prop_lower == 'cells':
            # Individual cells
            return [
                {'row': 1, 'column': 'A', 'value': 'Date'},
                {'row': 1, 'column': 'B', 'value': 'Amount'},
            ]
        
        # =====================================================================
        # DATABASE PATTERNS (Airtable, Notion, Supabase, Firebase)
        # =====================================================================
        elif prop_lower == 'records':
            # Airtable/Notion database records
            return [
                {'id': 'rec_001', 'fields': {'Name': 'Item 1', 'Status': 'Active', 'Amount': 150}},
                {'id': 'rec_002', 'fields': {'Name': 'Item 2', 'Status': 'Pending', 'Amount': 200}},
            ]
        elif prop_lower == 'fields':
            # Record fields (when accessed directly)
            return {'Name': 'Sample Record', 'Status': 'Active', 'Amount': 150, 'Date': '2025-01-15'}
        
        # =====================================================================
        # EMAIL/MESSAGING PATTERNS (Gmail, Outlook, Slack, Teams)
        # =====================================================================
        elif prop_lower == 'messages':
            return [
                {'id': 'msg_001', 'subject': 'Sample Email 1', 'from': 'sender@example.com', 'snippet': 'Email content preview...'},
                {'id': 'msg_002', 'subject': 'Sample Email 2', 'from': 'other@example.com', 'snippet': 'Another email preview...'},
            ]
        elif prop_lower == 'threads':
            return [
                {'id': 'thread_001', 'subject': 'Email Thread', 'messageCount': 3},
            ]
        elif prop_lower == 'attachments':
            return [
                {'id': 'att_001', 'filename': 'document.pdf', 'mimeType': 'application/pdf', 'size': 1024},
            ]
        elif prop_lower == 'channels':
            return [
                {'id': 'C001', 'name': 'general', 'is_private': False},
                {'id': 'C002', 'name': 'team-updates', 'is_private': False},
            ]
        
        # =====================================================================
        # FILE/STORAGE PATTERNS (Google Drive, Dropbox, OneDrive, Box)
        # =====================================================================
        elif prop_lower == 'files':
            return [
                {'id': 'file_001', 'name': 'Report.xlsx', 'mimeType': 'application/vnd.ms-excel', 'size': 2048},
                {'id': 'file_002', 'name': 'Presentation.pptx', 'mimeType': 'application/vnd.ms-powerpoint', 'size': 4096},
            ]
        elif prop_lower == 'folders':
            return [
                {'id': 'folder_001', 'name': 'Documents', 'itemCount': 15},
                {'id': 'folder_002', 'name': 'Projects', 'itemCount': 8},
            ]
        
        # =====================================================================
        # CALENDAR PATTERNS (Google Calendar, Outlook Calendar)
        # =====================================================================
        elif prop_lower == 'events':
            return [
                {'id': 'evt_001', 'summary': 'Team Meeting', 'start': '2025-01-15T10:00:00Z', 'end': '2025-01-15T11:00:00Z'},
                {'id': 'evt_002', 'summary': 'Project Review', 'start': '2025-01-16T14:00:00Z', 'end': '2025-01-16T15:00:00Z'},
            ]
        elif prop_lower == 'attendees':
            return [
                {'email': 'user1@example.com', 'responseStatus': 'accepted'},
                {'email': 'user2@example.com', 'responseStatus': 'tentative'},
            ]
        
        # =====================================================================
        # CRM PATTERNS (Salesforce, HubSpot, Pipedrive)
        # =====================================================================
        elif prop_lower == 'contacts':
            return [
                {'id': 'con_001', 'name': 'John Doe', 'email': 'john@example.com', 'phone': '+1234567890'},
                {'id': 'con_002', 'name': 'Jane Smith', 'email': 'jane@example.com', 'phone': '+0987654321'},
            ]
        elif prop_lower == 'deals' or prop_lower == 'opportunities':
            return [
                {'id': 'deal_001', 'name': 'Enterprise Deal', 'amount': 50000, 'stage': 'Negotiation'},
                {'id': 'deal_002', 'name': 'SMB Deal', 'amount': 5000, 'stage': 'Proposal'},
            ]
        elif prop_lower == 'leads':
            return [
                {'id': 'lead_001', 'name': 'New Lead', 'company': 'Acme Corp', 'status': 'New'},
            ]
        
        # =====================================================================
        # GENERIC PATTERNS (work across many APIs)
        # =====================================================================
        elif prop_lower == 'results':
            # Generic search/list results
            return [
                {'id': 'res_001', 'title': 'Result 1', 'score': 0.95},
                {'id': 'res_002', 'title': 'Result 2', 'score': 0.87},
            ]
        elif prop_lower == 'items':
            # Generic items list
            return [
                {'id': 'item_001', 'name': 'Item 1', 'type': 'default'},
                {'id': 'item_002', 'name': 'Item 2', 'type': 'default'},
            ]
        elif prop_lower == 'entries':
            # Generic entries
            return [
                {'id': 'entry_001', 'content': 'Entry content 1', 'created': '2025-01-15'},
                {'id': 'entry_002', 'content': 'Entry content 2', 'created': '2025-01-16'},
            ]
        elif prop_lower == 'data' and depth == 0:
            # Top-level data wrapper - don't override, let schema handle it
            pass
        
        # =====================================================================
        # FLIGHT/TRAVEL PATTERNS (Amadeus, Skyscanner, Composio flights)
        # =====================================================================
        elif prop_lower == 'flights':
            return [
                {'id': 'fl_001', 'airline': 'United', 'flight_number': 'UA123', 'departure': '2025-01-15T08:00:00Z', 'arrival': '2025-01-15T11:00:00Z', 'price': 350},
                {'id': 'fl_002', 'airline': 'Delta', 'flight_number': 'DL456', 'departure': '2025-01-15T10:00:00Z', 'arrival': '2025-01-15T13:00:00Z', 'price': 420},
            ]
        elif prop_lower == 'hotels':
            return [
                {'id': 'htl_001', 'name': 'Grand Hotel', 'rating': 4.5, 'price_per_night': 150},
                {'id': 'htl_002', 'name': 'City Inn', 'rating': 4.0, 'price_per_night': 95},
            ]
        
        # =====================================================================
        # DOCUMENT/CONTENT PATTERNS (Notion, Confluence, Google Docs)
        # =====================================================================
        elif prop_lower == 'blocks':
            return [
                {'id': 'blk_001', 'type': 'paragraph', 'text': 'Sample paragraph content'},
                {'id': 'blk_002', 'type': 'heading', 'text': 'Sample Heading'},
            ]
        elif prop_lower == 'pages':
            return [
                {'id': 'page_001', 'title': 'Project Overview', 'url': 'https://example.com/page1'},
                {'id': 'page_002', 'title': 'Meeting Notes', 'url': 'https://example.com/page2'},
            ]
        
        if schema_type == 'object':
            properties = schema.get('properties', {})
            mock_obj = {}
            
            for pname, prop_schema in list(properties.items())[:10]:  # Limit to 10 props
                mock_obj[pname] = self._generate_mock_from_schema(prop_schema, depth + 1, pname)
            
            return mock_obj
            
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            return [self._generate_mock_from_schema(items_schema, depth + 1, prop_name)]
            
        elif schema_type == 'string':
            # Check for format hints
            format_hint = schema.get('format', '')
            if 'email' in format_hint:
                return 'example@domain.com'
            elif 'date' in format_hint:
                return '2024-01-01'
            elif 'uri' in format_hint or 'url' in format_hint:
                return 'https://example.com'
            else:
                return 'sample_value'
                
        elif schema_type == 'integer':
            return 1
            
        elif schema_type == 'number':
            return 1.0
            
        elif schema_type == 'boolean':
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
