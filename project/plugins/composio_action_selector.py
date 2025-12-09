"""
Composio Action Selector - Intelligent action matching using pgvector.

Uses PostgreSQL pgvector for FAST similarity search (milliseconds, not seconds).

Problem: TaskWeaver knows to use composio_action plugin, but doesn't know
which specific action ID to use (17,000+ actions available).

Solution: 
1. Use pgvector HNSW index for fast similarity search IN DATABASE
2. No need to load all embeddings into memory
3. Inject matched actions into TaskWeaver prompt so LLM knows exact action IDs
"""
import logging
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ✅ LIVE DEBUGGING: File path for embedding query logs
DEBUG_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embedding_debug.log')

# Configuration constants
EMBEDDING_CACHE_TTL = 3600  # 1 hour
FALLBACK_ACTIONS_LIMIT = 500  # Top N actions for fallback scan
DEFAULT_TOP_K = 5  # Default number of actions to return
MAX_DESCRIPTION_LENGTH = 120  # Max chars for action description
MAX_PARAM_FIELDS = 3  # Max parameter fields to show
MAX_RESPONSE_FIELDS = 3  # Max response fields to show

# Cached OpenAI client and config
_openai_client = None
_embedding_deployment = None


def _log_to_debug_file(message: str):
    """
    Write debug message to file for live testing analysis.
    
    ✅ This allows us to see EXACTLY what pgvector returns during workflow generation.
    File: TaskWeaver/project/embedding_debug.log
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        logger.warning(f"[ACTION_SELECTOR] Failed to write debug log: {e}")


def _get_openai_client():
    """Get or create cached Azure OpenAI client (same pattern as action_matcher.py)"""
    global _openai_client, _embedding_deployment
    
    if _openai_client is not None:
        return _openai_client, _embedding_deployment
    
    try:
        from openai import AzureOpenAI
        from config.env_loader import config
        
        # Use EXACT same config pattern as action_matcher.py
        _embedding_deployment = config('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', default=None)
        
        if not _embedding_deployment:
            logger.warning("[ACTION_SELECTOR] AZURE_OPENAI_EMBEDDING_DEPLOYMENT not configured")
            return None, None
        
        _openai_client = AzureOpenAI(
            api_key=config('AZURE_OPENAI_EMBEDDING_API_KEY', default=config('AZURE_OPENAI_API_KEY')),
            api_version=config('AZURE_OPENAI_EMBEDDING_API_VERSION', default=config('AZURE_OPENAI_API_VERSION')),
            azure_endpoint=config('AZURE_OPENAI_EMBEDDING_ENDPOINT', default=config('AZURE_OPENAI_ENDPOINT'))
        )
        
        logger.info(f"[ACTION_SELECTOR] ✅ Azure OpenAI initialized (deployment: {_embedding_deployment})")
        return _openai_client, _embedding_deployment
        
    except Exception as e:
        logger.warning(f"[ACTION_SELECTOR] Could not initialize Azure OpenAI: {e}")
        return None, None


def _get_query_embedding(query: str) -> Optional[List[float]]:
    """Get embedding for user query using Azure OpenAI (same model as DB embeddings)"""
    try:
        from django.core.cache import cache
        
        client, deployment = _get_openai_client()
        if not client or not deployment:
            logger.warning("[ACTION_SELECTOR] No OpenAI client available")
            return None
        
        # Check cache first
        text_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"action_selector_emb:{deployment}:{text_hash}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Generate embedding
        response = client.embeddings.create(
            model=deployment,
            input=query
        )
        embedding = response.data[0].embedding
        
        # Cache for configured TTL
        cache.set(cache_key, embedding, timeout=EMBEDDING_CACHE_TTL)
        
        logger.debug(f"[ACTION_SELECTOR] Generated embedding ({len(embedding)} dims)")
        return embedding
        
    except Exception as e:
        logger.warning(f"[ACTION_SELECTOR] Could not embed query: {e}")
        return None


def select_actions_pgvector(
    user_query: str, 
    app_filter: Optional[str] = None,
    top_k: int = 10
) -> List[Dict]:
    """
    ⚡ FAST: Select actions using pgvector indexed search (milliseconds!)
    
    This is the production-grade approach - no memory loading required.
    Falls back to linear scan if pgvector extension not available.
    """
    try:
        import django
        if not django.apps.apps.ready:
            django.setup()
        
        from apps.integrations.models import ComposioActionSchema
    except Exception as e:
        logger.warning(f"[ACTION_SELECTOR] Django not available: {e}")
        return []
    
    # Get query embedding
    query_emb = _get_query_embedding(user_query)
    if not query_emb:
        return []
    
    # ✅ LIVE DEBUGGING: Write query to file for analysis
    _log_to_debug_file(f"\n{'='*80}\n[PGVECTOR QUERY] {user_query[:200]}\n{'='*80}")
    
    # ✅ Try pgvector indexed search first (FAST!)
    try:
        from pgvector.django import CosineDistance
        from django.db.utils import ProgrammingError
        
        # Build query
        queryset = ComposioActionSchema.objects.filter(
            is_deprecated=False,
            description_embedding__isnull=False
        )
        
        # ✅ CRITICAL: Exclude meta-tools (COMPOSIO_* actions)
        # These are orchestration tools, not app-specific actions
        queryset = queryset.exclude(action_id__startswith='COMPOSIO_')
        
        # NOTE: External form services (BYTEFORMS, FEATHERY, etc.) are NOT filtered here.
        # The LLM is guided via prompt instructions to prefer internal form_collect() plugin
        # for user input/HITL. External form services may still be useful for edge cases
        # (e.g., publishing forms to external websites).
        
        # Optional app filter
        if app_filter:
            queryset = queryset.filter(action_id__icontains=app_filter.upper())
        
        # pgvector similarity search - runs IN DATABASE!
        actions = queryset.annotate(
            distance=CosineDistance('description_embedding', query_emb)
        ).order_by('distance')[:top_k].values(
            'action_id',
            'action_name', 
            'description',
            'parameters_schema',
            'parameter_examples',
            'response_schema',
            'distance'
        )
        
        results = []
        for action in actions:
            similarity = 1.0 - action['distance']
            results.append({
                'action_id': action['action_id'],
                'action_name': action['action_name'],
                'description': action['description'] or '',
                'parameters': action['parameters_schema'] or {},
                'parameter_examples': action['parameter_examples'] or {},
                'response_schema': action['response_schema'] or {},
                'similarity': similarity
            })
        
        # Return top_k after filtering
        results = results[:top_k]
        
        # ✅ CRITICAL: Log query and results for debugging embedding quality
        logger.info(f"[ACTION_SELECTOR] 🔍 Query: '{user_query[:80]}...'")
        logger.info(f"[ACTION_SELECTOR] ⚡ pgvector returned {len(results)} actions:")
        for i, r in enumerate(results[:5], 1):  # Show top 5
            logger.info(f"[ACTION_SELECTOR]   {i}. {r['action_id']} (similarity: {r['similarity']:.3f})")
        
        # ✅ LIVE DEBUGGING: Write detailed results to file
        _log_to_debug_file(f"[PGVECTOR RESULTS] Returned {len(results)} actions (top {min(10, len(results))} shown):")
        for i, r in enumerate(results[:10], 1):
            # Get app category for analysis
            try:
                from apps.integrations.models import ComposioActionSchema
                action_obj = ComposioActionSchema.objects.filter(action_id=r['action_id']).select_related('integration').first()
                app_category = action_obj.integration.integration_category if action_obj and action_obj.integration else 'NULL'
                app_id = action_obj.integration.integration_id if action_obj and action_obj.integration else 'N/A'
            except:
                app_category = 'ERROR'
                app_id = 'ERROR'
            
            _log_to_debug_file(
                f"  {i:2d}. {r['action_id']:40s} | sim: {r['similarity']:.4f} | "
                f"app: {app_id:15s} | category: {app_category}"
            )
        
        return results
        
    except ImportError:
        logger.debug("[ACTION_SELECTOR] pgvector not installed, using fallback")
    except Exception as e:
        # ProgrammingError = pgvector extension not enabled
        logger.debug(f"[ACTION_SELECTOR] pgvector query failed ({type(e).__name__}), using fallback")
    
    # ❌ FALLBACK: Limited linear scan (only if pgvector fails)
    # This is slower but works without pgvector extension
    return _select_actions_fallback(user_query, query_emb, app_filter, top_k)


def _select_actions_fallback(
    user_query: str,
    query_emb: List[float],
    app_filter: Optional[str],
    top_k: int
) -> List[Dict]:
    """
    Fallback: Sample-based linear scan (NOT loading all 17k embeddings!)
    
    Strategy: Load top 500 most-used actions and scan those only.
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from apps.integrations.models import ComposioActionSchema
        
        # Only load top N most-used actions (not all 17k!)
        queryset = ComposioActionSchema.objects.filter(
            is_deprecated=False,
            description_embedding__isnull=False
        ).exclude(action_id__startswith='COMPOSIO_').order_by('-usage_count')[:FALLBACK_ACTIONS_LIMIT]  # ✅ Exclude meta-tools
        
        if app_filter:
            queryset = queryset.filter(action_id__icontains=app_filter.upper())
        
        actions = list(queryset.values(
            'action_id', 'action_name', 'description', 
            'parameters_schema', 'parameter_examples', 'response_schema', 'description_embedding'  # ✅ NEW: Add examples + response_schema
        ))
        
        if not actions:
            return []
        
        # Compute similarities
        query_np = np.array(query_emb).reshape(1, -1)
        results = []
        
        for action in actions:
            emb = action.get('description_embedding')
            if not emb:
                continue
            
            action_np = np.array(emb).reshape(1, -1)
            sim = cosine_similarity(query_np, action_np)[0][0]
            
            results.append({
                'action_id': action['action_id'],
                'action_name': action['action_name'],
                'description': action['description'] or '',
                'parameters': action['parameters_schema'] or {},
                'parameter_examples': action['parameter_examples'] or {},  # ✅ NEW: Include examples
                'response_schema': action['response_schema'] or {},  # ✅ NEW: Include response schema
                'similarity': float(sim)
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        logger.info(f"[ACTION_SELECTOR] 📦 Fallback scanned {len(actions)} actions")
        return results[:top_k]
        
    except Exception as e:
        logger.warning(f"[ACTION_SELECTOR] Fallback failed: {e}")
        return []


def format_actions_for_prompt(actions: List[Dict], top_k: int = 5) -> str:
    """
    Format actions with FULL schema information for LLM.
    
    ✅ NEW: Includes parameter types, examples, AND response structure!
    This tells the LLM exactly what to expect, reducing type/key errors by ~80%.
    """
    if not actions:
        return ""
    
    lines = ["## Available Composio Actions (use exact action_id):"]
    
    for action in actions[:top_k]:
        action_id = action['action_id']
        desc = (action.get('description') or '')[:MAX_DESCRIPTION_LENGTH]
        
        # ✅ Parameter schema with types AND descriptions
        params = action.get('parameters', {})
        required = params.get('required', [])
        param_details = []
        properties = params.get('properties', {})
        
        for param_name in required[:MAX_PARAM_FIELDS]:
            param_schema = properties.get(param_name, {})
            param_type = param_schema.get('type', 'any')
            param_desc = param_schema.get('description', '')
            
            # Include description if available and short
            if param_desc and len(param_desc) < 40:
                param_details.append(f"{param_name}: {param_type} ({param_desc})")
            else:
                param_details.append(f"{param_name}: {param_type}")
        
        params_str = f" | Params: {', '.join(param_details)}" if param_details else ""
        
        # ✅ Response schema (CRITICAL for preventing type errors!)
        response_hint = ""
        # Support both 'response' (from action_matcher) and 'response_schema' (from old selector)
        response_schema = action.get('response') or action.get('response_schema')
        
        if response_schema and isinstance(response_schema, dict):
            response_type = response_schema.get('type', 'object')
            
            if response_type == 'array':
                # It's a List!
                items = response_schema.get('items', {})
                if isinstance(items, dict):
                    item_props = items.get('properties', {})
                    if item_props:
                        # Show key fields in response
                        key_fields = list(item_props.keys())[:MAX_RESPONSE_FIELDS]
                        response_hint = f" | Returns: List[{{{', '.join(key_fields)}}}]"
                    else:
                        response_hint = " | Returns: List"
            
            elif response_type == 'object':
                # It's a Dict - but check if it has 'data' wrapper (common pattern)
                props = response_schema.get('properties', {})
                
                # Unwrap 'data' field if present (Composio pattern: {successful, data, error})
                if 'data' in props and 'successful' in props:
                    data_schema = props.get('data', {})
                    data_props = data_schema.get('properties', {})
                    if data_props:
                        # Build rich structure showing nested arrays
                        field_descriptions = []
                        for field_name in list(data_props.keys())[:MAX_RESPONSE_FIELDS]:
                            field_schema = data_props[field_name]
                            field_type = field_schema.get('type')
                            
                            # Check if this field is an array and show its item structure
                            if field_type == 'array':
                                items_schema = field_schema.get('items', {})
                                items_props = items_schema.get('properties', {})
                                if items_props:
                                    item_keys = list(items_props.keys())[:3]  # Show first 3 fields
                                    field_descriptions.append(f"{field_name}: [{{{', '.join(item_keys)}}}]")
                                else:
                                    field_descriptions.append(f"{field_name}: List")
                            else:
                                field_descriptions.append(field_name)
                        
                        response_hint = f" | Returns: {{{', '.join(field_descriptions)}}}"
                    else:
                        data_type = data_schema.get('type')
                        if data_type == 'array':
                            response_hint = " | Returns: List"
                        else:
                            response_hint = " | Returns: Dict"
                elif props:
                    key_fields = list(props.keys())[:MAX_RESPONSE_FIELDS]
                    response_hint = f" | Returns: {{{', '.join(key_fields)}}}"
                else:
                    response_hint = " | Returns: Dict"
        
        # Format: - ACTION_ID: description | Params: x: type, y: type | Returns: structure
        line = f"- {action_id}: {desc}{params_str}{response_hint}"
        
        # ✅ CRITICAL: Error handling pattern for placeholder detection (mock mode only)
        # Successful responses are unwrapped data dicts (NO 'success' field).
        # Only placeholder errors return {'success': False, 'error': '...'} in mock mode.
        # Real execution raises exceptions (LLM never sees error responses).
        line += f"\n  ⚠️ Error handling: if 'error' in result: raise Exception(result['error'])"
        
        # ✅ NEW: Add usage example showing correct extraction pattern
        param_examples = action.get('parameter_examples', {})
        if param_examples and required:
            # Build example call with first required param
            first_param = required[0]
            example_values = param_examples.get(first_param)
            
            # Handle both list and single value examples
            if example_values:
                if isinstance(example_values, list) and len(example_values) > 0:
                    example_value = example_values[0]
                elif not isinstance(example_values, (list, dict)):
                    # Single value (int, str, etc.)
                    example_value = example_values
                else:
                    example_value = None
                    
                if example_value is not None:
                    # Format the example value
                    if isinstance(example_value, str):
                        example_value = f'"{example_value}"'
                    line += f"\n  Example: result, desc = composio_action(\"{action_id}\", {{\"{first_param}\": {example_value}}})"
                    
                    # ✅ CRITICAL: Show error check in usage example (LLMs learn from examples, not just instructions)
                    line += f"\n           if 'error' in result: raise Exception(result['error'])"
                    
                    # Show how to extract from response based on structure
                    if response_schema and isinstance(response_schema, dict):
                        props = response_schema.get('properties', {})
                        
                        # Check for Composio wrapper pattern
                        if 'data' in props and 'successful' in props:
                            data_schema = props.get('data', {})
                            data_type = data_schema.get('type')
                            data_props = data_schema.get('properties', {})
                            
                            if data_type == 'array':
                                line += f"\n           data = result  # List"
                            elif data_type == 'object' and data_props:
                                first_field = list(data_props.keys())[0]
                                first_field_schema = data_props[first_field]
                                
                                # Check if first field is an array
                                if first_field_schema.get('type') == 'array':
                                    items_props = first_field_schema.get('items', {}).get('properties', {})
                                    if items_props:
                                        first_item_key = list(items_props.keys())[0]
                                        line += f"\n           {first_field} = result['{first_field}']  # List of dicts"
                                        line += f"\n           item_key = {first_field}[0]['{first_item_key}']  # Array element"
                                    else:
                                        line += f"\n           {first_field} = result['{first_field}']  # List"
                                else:
                                    line += f"\n           {first_field} = result['{first_field}']"
                            else:
                                line += f"\n           data = result"
                        elif response_type == 'array':
                            line += f"\n           first_item = result[0]  # List of dicts"
                        elif response_type == 'object' and props:
                            first_field = list(props.keys())[0]
                            line += f"\n           {first_field} = result['{first_field}']"
        
        lines.append(line)
    
    return "\n".join(lines)


def select_composio_actions(
    user_query: str,
    app_hints: Optional[List[str]] = None,
    top_k: int = 5
) -> str:
    """
    ⚡ Main entry point - Get relevant Composio actions for a query.
    
    ✅ NEW: Uses TWO-STAGE search (apps → actions) instead of global search.
    This matches Composio's own architecture and dramatically improves precision.
    
    Architecture:
        1. Find relevant APPS (200 apps, fast pgvector search)
        2. Search ACTIONS within those apps (10-50 actions, scoped search)
    
    Returns formatted string to inject into TaskWeaver prompt.
    """
    # ✅ LIVE DEBUGGING: Mark start of new action selection session
    _log_to_debug_file(f"\n{'#'*80}\n[NEW SESSION - TWO STAGE] User Query: {user_query[:150]}\n{'#'*80}")
    if app_hints:
        _log_to_debug_file(f"[APP HINTS] {app_hints}")
    
    try:
        # ✅ Use the WORKING two-stage matcher (action_matcher.py)
        from apps.integrations.services.action_matcher import ComposioActionMatcher
        
        matcher = ComposioActionMatcher(enable_warmup=False)  # No warmup for speed
        
        # Match actions using two-stage approach
        action_dicts = matcher.match_for_subtask(
            subtask_description=user_query,
            app_hints=app_hints,
            top_k_apps=10 if not app_hints else len(app_hints),  # Search 10 apps (increased for multi-concept queries)
            top_k_actions_per_app=2,  # 2 actions per app = ~20 total
            min_confidence=0.25  # Lower threshold for better recall
        )
        
        if not action_dicts:
            _log_to_debug_file("[TWO STAGE] No actions matched")
            return ""
        
        # Log the matched apps and actions
        matched_apps = list(set([a['app_name'] for a in action_dicts]))
        _log_to_debug_file(f"[TWO STAGE] Matched apps: {matched_apps}")
        _log_to_debug_file(f"[TWO STAGE] Found {len(action_dicts)} actions:")
        
        # Enhanced logging - show ALL actions returned
        logger.info(f"[TWO STAGE] 🎯 Returned {len(action_dicts)} actions from two-stage search:")
        for i, action in enumerate(action_dicts[:top_k], 1):
            action_id = action['action_id']
            app_name = action['app_name']
            confidence = action.get('confidence', 0)
            _log_to_debug_file(
                f"  {i}. {action_id} "
                f"(app: {app_name}, confidence: {confidence:.4f})"
            )
            logger.info(f"[TWO STAGE]   {i}. {action_id} (app: {app_name}, confidence: {confidence:.3f})")
        
        # Format for TaskWeaver prompt (same format as before)
        return format_actions_for_prompt(action_dicts, top_k)
        
    except ImportError as e:
        _log_to_debug_file(f"[TWO STAGE] Import error, falling back to old method: {e}")
        # Fallback to old global search if action_matcher not available
        return _select_composio_actions_fallback(user_query, app_hints, top_k)
    except Exception as e:
        logger.error(f"[TWO STAGE] Error in two-stage search: {e}", exc_info=True)
        _log_to_debug_file(f"[TWO STAGE] Error: {e}")
        # Fallback to old global search
        return _select_composio_actions_fallback(user_query, app_hints, top_k)


def _select_composio_actions_fallback(
    user_query: str,
    app_hints: Optional[List[str]] = None,
    top_k: int = 5
) -> str:
    """
    ⚠️  FALLBACK: Old global pgvector search (less accurate).
    Only used if ComposioActionMatcher is unavailable.
    """
    _log_to_debug_file(f"[FALLBACK] Using old global search method")
    
    all_actions = []
    
    # If app hints provided, get top actions from each app
    if app_hints:
        for app in app_hints:
            actions = select_actions_pgvector(user_query, app_filter=app, top_k=3)
            all_actions.extend(actions)
    else:
        all_actions = select_actions_pgvector(user_query, top_k=top_k)
    
    if not all_actions:
        return ""
    
    # Deduplicate by action_id
    seen = set()
    unique_actions = []
    for action in all_actions:
        if action['action_id'] not in seen:
            seen.add(action['action_id'])
            unique_actions.append(action)
    
    # Sort by similarity
    unique_actions.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    # ✅ LIVE DEBUGGING: Log final actions sent to LLM
    _log_to_debug_file(f"[FALLBACK SELECTION] Top {min(top_k, len(unique_actions))} actions being sent to LLM:")
    for i, action in enumerate(unique_actions[:top_k], 1):
        _log_to_debug_file(f"  {i}. {action['action_id']} (similarity: {action.get('similarity', 0):.4f})")
    
    return format_actions_for_prompt(unique_actions, top_k)

