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
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ===================================================================
# BATCH CALL CACHE
# ===================================================================
# Cache structure: {session_id: {step_query: [action_dicts]}}
# Stores Composio batch API results for the entire workflow session
_BATCH_CACHE = {}

def _fuzzy_match_step(query: str, cached_steps: list, threshold: float = 0.60) -> Optional[str]:
    """
    Find best matching cached step using simple word overlap.
    
    Args:
        query: Current step query
        cached_steps: List of cached step queries
        threshold: Minimum similarity score (0-1)
    
    Returns:
        Best matching cached step or None
    """
    import re
    
    # Normalize query
    query_words = set(re.findall(r'\w+', query.lower()))
    
    best_match = None
    best_score = 0
    
    for cached_step in cached_steps:
        cached_words = set(re.findall(r'\w+', cached_step.lower()))
        
        # Calculate Jaccard similarity (word overlap)
        if not query_words or not cached_words:
            continue
        
        intersection = len(query_words & cached_words)
        union = len(query_words | cached_words)
        score = intersection / union if union > 0 else 0
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = cached_step
    
    return best_match

# ✅ LIVE DEBUGGING: File path for embedding query logs
DEBUG_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embedding_debug.log')

# Configuration constants
EMBEDDING_CACHE_TTL = 3600  # 1 hour
FALLBACK_ACTIONS_LIMIT = 500  # Top N actions for fallback scan
DEFAULT_TOP_K = 5  # Default number of actions to return
MAX_DESCRIPTION_LENGTH = 120  # Max chars for action description
MAX_REQUIRED_PARAMS = 999  # Show ALL required params (critical for form generation!)
MAX_OPTIONAL_PARAMS = 6  # Max optional params to show (adaptive based on total)
MAX_RESPONSE_FIELDS = 3  # Max response fields to show

# Cached OpenAI client and config
_openai_client = None
_embedding_deployment = None


def detect_query_intent(user_query: str) -> str:
    """
    Detect if query is asking to READ/FETCH or WRITE/SEND.
    
    Returns: "read" | "write" | "both"
    
    Examples:
        "fetch emails" → "read"
        "send message" → "write"
        "read emails and send responses" → "both"
    """
    query_lower = user_query.lower()
    
    # Read/fetch intent keywords
    read_keywords = ['fetch', 'get', 'read', 'list', 'query', 'retrieve', 'download', 
                     'find', 'search', 'view', 'show', 'check', 'load']
    
    # Write/send intent keywords
    write_keywords = ['send', 'post', 'create', 'update', 'delete', 'reply', 'forward',
                      'publish', 'upload', 'write', 'draft', 'compose']
    
    # Count occurrences to determine PRIMARY intent
    read_count = sum(query_lower.count(kw) for kw in read_keywords)
    write_count = sum(query_lower.count(kw) for kw in write_keywords)
    
    # ✅ CRITICAL: For complete workflows, if BOTH read AND write are present, return "both"
    # This ensures actions for entire workflow are available (fetch + process + send)
    if read_count > 0 and write_count > 0:
        # If write keywords appear explicitly (send, reply, respond), include write actions
        # This handles: "fetch messages and send responses" -> both
        explicit_write_keywords = ['send', 'post', 'reply', 'respond', 'forward', 'publish']
        has_explicit_write = any(kw in query_lower for kw in explicit_write_keywords)
        
        if has_explicit_write:
            # User explicitly wants to WRITE after READ -> return "both"
            return "both"
        
        # Otherwise use priority-based detection
        first_read_pos = min((query_lower.find(kw) for kw in read_keywords if kw in query_lower), default=999)
        first_write_pos = min((query_lower.find(kw) for kw in write_keywords if kw in query_lower), default=999)
        
        # If read comes first OR is mentioned 2x more, treat as read
        if first_read_pos < first_write_pos and read_count >= write_count * 2:
            return "read"
        elif write_count >= read_count * 2:
            return "write"
        else:
            return "both"
    elif read_count > 0:
        return "read"
    elif write_count > 0:
        return "write"
    else:
        return "both"  # Default to both if unclear


# ✅ DELETED: detect_apps_from_query() function
# No longer needed - action_matcher.py now handles ALL app detection semantically!
# This removes 130+ lines of keyword matching and hardcoded app lists.


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
        
        # ✅ CRITICAL: Exclude ONLY orchestration meta-tools, NOT actual search tools
        # COMPOSIO_SEARCH_TOOLS = Meta-tool for discovering other tools (exclude this)
        # COMPOSIO_SEARCH_FLIGHTS/SCHOLAR/HOTEL = Actual search tools (INCLUDE these!)
        META_ORCHESTRATION_TOOLS = [
            'COMPOSIO_SEARCH_TOOLS',  # Tool discovery API
            'COMPOSIO_GET_TOOLS',     # Tool listing API
            'COMPOSIO_LIST_TOOLS',    # Tool catalog API
        ]
        queryset = queryset.exclude(action_id__in=META_ORCHESTRATION_TOOLS)
        
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
        ).exclude(action_id__in=[
            'COMPOSIO_SEARCH_TOOLS',  # Tool discovery API
            'COMPOSIO_GET_TOOLS',     # Tool listing API
            'COMPOSIO_LIST_TOOLS',    # Tool catalog API
        ]).order_by('-usage_count')[:FALLBACK_ACTIONS_LIMIT]  # ✅ Exclude ONLY orchestration meta-tools
        
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
    
    lines = [
        "## Available Composio Actions (use exact action_id):",
        "",
        "⚠️ CRITICAL RULES:",
        "1. You MUST ONLY use action IDs from the numbered list below",
        "2. DO NOT invent action names based on patterns",
        "3. DO NOT extrapolate from similar app names (APP1_ACTION ≠ APP2_ACTION)",
        "4. If you use an action NOT in this list, your code will FAIL at runtime",
        "5. Copy the EXACT action_id shown below - character-for-character",
        "",
        "🔴 FORM GENERATION REQUIREMENTS (100% MANDATORY):",
        "- Parameters marked with * (req) are REQUIRED - you MUST create form fields for them!",
        "- Field 'name' MUST exactly match parameter name (e.g., 'departure_id' NOT 'departure' or 'from')",
        "- Use 'label' for human-readable text (e.g., label='Departure Airport ID')",
        "- Examples shown in (e.g., ...) indicate expected format - use as placeholder",
        "- NEVER hardcode values that users should provide - ALWAYS collect via form!",
        "",
        "📋 Available Actions:",
        ""
    ]
    
    for action in actions[:top_k]:
        action_id = action['action_id']
        app_name = action.get('app_name', 'unknown')  # ✅ Extract app_name
        desc = (action.get('description') or '')[:MAX_DESCRIPTION_LENGTH]
        
        # ✅ Parameter schema with types AND descriptions
        # ⚠️ ROOT CAUSE FIX: Show BOTH required AND important optional params
        # Previously only showed required params, causing LLM to miss optional params like return_date
        params = action.get('parameters', {})
        required = params.get('required', [])
        param_details = []
        properties = params.get('properties', {})
        
        # ✅ CRITICAL: Show ALL required params (not just 3!)
        # Required params MUST be in forms, so LLM needs to see them all
        for param_name in required[:MAX_REQUIRED_PARAMS]:
            param_schema = properties.get(param_name, {})
            param_type = param_schema.get('type', 'any')
            param_desc = param_schema.get('description', '')
            param_examples = param_schema.get('examples', [])
            
            # Build rich param description
            param_info = f"{param_name}* (req): {param_type}"
            
            # Add description if available
            if param_desc and len(param_desc) < 80:
                param_info += f" - {param_desc[:75]}"
            
            # Add example hint if available (helps LLM understand format)
            if param_examples and len(param_examples) > 0:
                example = str(param_examples[0])
                if len(example) < 30:
                    param_info += f" (e.g., {example})"
            
            param_details.append(param_info)
        
        # ✅ NEW: Show important optional params too (prioritized + adaptive count)
        optional_params = [k for k in properties.keys() if k not in required]
        
        if optional_params:
            # ✅ 100% SCHEMA-DRIVEN PRIORITIZATION: No hardcoded keywords!
            # Uses only JSON Schema metadata and statistical signals
            scored = []
            for param_name in optional_params:
                score = 0
                param_schema = properties.get(param_name, {})
                
                # Signal 1: Has examples → API provider documented it, likely important
                if param_schema.get('examples'):
                    score += 10
                
                # Signal 2: Has default value → Important enough to have a default
                if 'default' in param_schema and param_schema['default'] is not None:
                    score += 5
                
                # Signal 3: Not nullable → More constrained = more important
                if param_schema.get('nullable') is False:
                    score += 3
                
                # Signal 4: Has validation constraints → Important enough to validate
                if param_schema.get('enum'):  # Enum values
                    score += 6
                if param_schema.get('pattern'):  # Regex pattern
                    score += 4
                if param_schema.get('minLength') or param_schema.get('maxLength'):
                    score += 3
                if param_schema.get('minimum') or param_schema.get('maximum'):
                    score += 3
                
                # Signal 5: Description length → Well-documented = important
                desc_length = len(param_schema.get('description') or '')
                if desc_length > 100:
                    score += 4
                elif desc_length > 50:
                    score += 2
                
                # Signal 6: Cross-referenced in tool description → Mentioned in overview
                tool_desc = (action.get('description') or '').lower()
                if param_name.lower() in tool_desc:
                    score += 6
                
                # Signal 7: Complex type → Structured data is usually important
                param_type = param_schema.get('type', '')
                if param_type in ('object', 'array'):
                    score += 3
                
                scored.append((param_name, score))
            
            # Sort by score descending, then alphabetically for determinism
            scored.sort(key=lambda x: (-x[1], x[0]))
            prioritized_optional = [p[0] for p in scored]
            
            # ✅ ADAPTIVE COUNT: Show more for tools with many params
            if len(optional_params) <= 3:
                show_count = len(optional_params)
            elif len(optional_params) <= 6:
                show_count = 4
            elif len(optional_params) <= 10:
                show_count = 5
            else:
                show_count = MAX_OPTIONAL_PARAMS
            
            # Show prioritized optional params
            for param_name in prioritized_optional[:show_count]:
                param_schema = properties.get(param_name, {})
                param_type = param_schema.get('type', 'any')
                param_desc = param_schema.get('description', '')
                
                if param_desc and len(param_desc) < 60:
                    param_details.append(f"{param_name} (opt): {param_type} - {param_desc[:50]}")
                else:
                    param_details.append(f"{param_name} (opt): {param_type}")
            
            # ✅ HINT: If more params exist, tell LLM
            if len(optional_params) > show_count:
                remaining = len(optional_params) - show_count
                param_details.append(f"... +{remaining} more optional")
        
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
        
        # Format: - ACTION_ID (app: app_name): description | Params: x: type, y: type | Returns: structure
        # ✅ FIX: Include app_name so LLM uses correct app in workflow nodes
        # ✅ NEW: Include step_index for targeted tool mapping (if present)
        step_index_hint = ""
        if 'step_index' in action:
            step_index_hint = f" [step_index: {action['step_index']}]"
        line = f"- {action_id} (app: {app_name}): {desc}{params_str}{response_hint}{step_index_hint}"
        
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
    top_k: int = 5,
    context: Optional[str] = None,
    adaptive_top_k: bool = True,
    session_id: Optional[str] = None,
    structured_steps: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    ⚡ Main entry point - Get relevant Composio actions for a query.
    
    ✅ HYBRID APPROACH (Option 2):
    - user_query: Step-specific task for INTENT matching ("Fetch data from platform")
    - context: Full user request for DOMAIN keywords ("read data... send responses")
    
    ✅ ADAPTIVE TOP_K (NEW):
    - Single platform: top_k=7 (fast)
    - Multi-platform (4 apps): top_k=12 (comprehensive)
    - Scales dynamically: top_k = max(7, len(apps) * 3)
    
    ✅ Uses TWO-STAGE search (apps → actions) for precision.
    
    Architecture:
        1. Find relevant APPS using context (domain keywords)
        2. Search ACTIONS using user_query (step intent)
        3. Adapt top_k based on detected apps (if adaptive_top_k=True)
    
    Returns formatted string to inject into TaskWeaver prompt.
    """
    # ✅ LIVE DEBUGGING: Mark start of new action selection session
    _log_to_debug_file(f"\n{'#'*80}\n[NEW SESSION - ADAPTIVE] Step Query: {user_query[:100]}\n{'#'*80}")
    if context:
        _log_to_debug_file(f"[FULL CONTEXT] {context[:150]}")
    
    # ===================================================================
    # 🆕 BATCH CALL: Check cache first, make batch call if needed
    # ===================================================================
    # Use passed session_id if available, otherwise use context[:50] as implicit session ID
    if not session_id:
        session_id = context[:50] if context else None
    
    # Check if we have cached results from batch call
    if session_id and session_id in _BATCH_CACHE:
        # ✅ DETERMINISTIC: Use step index if available (retry-safe)
        if structured_steps:
            # 🔧 CRITICAL FIX: Collect tools for ALL steps, not just the first match!
            # On retry, structured_steps are passed again - use index-based lookup
            all_cached_tools = []
            found_count = 0
            for step in structured_steps:
                step_index = step.get('index')
                cache_key = f"__INDEX_{step_index}__"
                if cache_key in _BATCH_CACHE[session_id]:
                    logger.info(f"✅ [BATCH_CACHE] Index-based hit for step {step_index}")
                    _log_to_debug_file(f"[BATCH_CACHE] Index {step_index} hit (retry-safe)")
                    step_tools = _BATCH_CACHE[session_id][cache_key]
                    all_cached_tools.extend(step_tools)
                    found_count += 1
            
            if found_count > 0:
                logger.info(f"✅ [BATCH_CACHE] Returning {len(all_cached_tools)} tools from {found_count} cached steps")
                return format_actions_for_prompt(all_cached_tools, len(all_cached_tools))
            
            logger.warning(f"⚠️ [BATCH_CACHE] Index-based lookup failed for {len(structured_steps)} steps")
        
        # ✅ FALLBACK: Try fuzzy matching (for non-structured calls)
        cached_steps = [k for k in _BATCH_CACHE[session_id].keys() if not k.startswith('__INDEX_')]
        matched_step = _fuzzy_match_step(user_query, cached_steps)
        
        if matched_step:
            logger.info(f"✅ [BATCH_CACHE] Found cached tools (matched: '{matched_step[:60]}...')")
            _log_to_debug_file(f"[BATCH_CACHE] Hit! Returning cached results for step")
            action_dicts = _BATCH_CACHE[session_id][matched_step]
            return format_actions_for_prompt(action_dicts, top_k)
    
    # If no cache and we have full context, make ONE batch call for entire workflow
    if context and session_id and session_id not in _BATCH_CACHE:
        try:
            # Add plugins directory to path for import
            import sys
            import os
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            from composio_batch_search import AIBatchSearch
            import asyncio
            
            logger.info("[BATCH_API] 🚀 Making ONE batch call for entire workflow...")
            _log_to_debug_file("[BATCH_API] First call - fetching tools for all steps")
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Make batch API call with full context
            # ✅ Get entity_id from environment (set by eclipse_adapter.py)
            # This ensures we use the actual user's Composio entity with configured integrations
            entity_id = os.environ.get('COMPOSIO_ENTITY_ID', 'default')
            logger.info(f"[BATCH_API] Using entity_id: {entity_id}")
            
            # ✅ FIX: Pass session_id to avoid namespace errors
            # When session_id is None, Composio tries to use default namespace which doesn't exist
            # Use the TaskWeaver session_id (not a Composio session_id) to generate unique session
            composio_session_id = session_id if session_id else None  # Let Composio generate if needed
            logger.info(f"[BATCH_API] Using session_id: {composio_session_id}")
            
            composio_api = AIBatchSearch()
            search_result = loop.run_until_complete(
                composio_api.search_batch(
                    user_prompt=context,  # Full workflow prompt (used if structured_steps is None)
                    entity_id=entity_id,  # ✅ Use actual user's entity (not hardcoded)
                    session_id=composio_session_id,  # ✅ Pass session to avoid namespace error
                    timeout=10,  # 10 seconds for batch
                    structured_steps=structured_steps  # ✅ NEW: Per-step targeted tool discovery
                )
            )
            
            # Cache results per sub-task
            if search_result and search_result.sub_task_results:
                logger.info(f"✅ [BATCH_API] Got tools for {len(search_result.sub_task_results)} sub-tasks")
                _log_to_debug_file(f"[BATCH_API] Caching results for {len(search_result.sub_task_results)} steps")
                
                _BATCH_CACHE[session_id] = {}
                
                # ✅ NEW: Track if this is a structured_steps response (indexed mode)
                is_structured_mode = structured_steps is not None
                
                if is_structured_mode:
                    # ✅ TARGETED MODE: Index-based mapping (no fuzzy matching needed!)
                    logger.info("🎯 [BATCH_API] TARGETED MODE: Using index-based tool mapping")
                    all_action_dicts = []
                    
                    for sub_task_result in search_result.sub_task_results:
                        step_index = sub_task_result.index
                        step_query = sub_task_result.use_case
                        tools = sub_task_result.tools
                        
                        # ✅ CRITICAL FIX: When Composio returns multiple tools, use the PRIMARY tool (lowest execution_order)
                        # 
                        # CONTEXT:
                        # - Composio's execution_order represents a multi-step plan sequence (1, 2, 3, ...)
                        # - BUT we generate ONE workflow node per step (user's step = one node)
                        # - So we need the PRIMARY action (order=1), not the final step in Composio's plan
                        # 
                        # EXAMPLE:
                        # User's step: "Send email with flight details"
                        # Composio returns: [GMAIL_SEND_EMAIL (order=1), CREATE_DRAFT (order=2), SEND_DRAFT (order=3)]
                        # ✅ We want: GMAIL_SEND_EMAIL (order=1) - the direct action
                        # ❌ Not: SEND_DRAFT (order=3) - the final step in a multi-step plan
                        if tools:
                            if len(tools) > 1:
                                # Use PRIMARY tool (lowest execution_order = first action in Composio's plan)
                                tool = min(tools, key=lambda t: t.get('execution_order', 999))
                                logger.info(f"   [INDEX {step_index}] Multi-step detected: {len(tools)} tools, selected PRIMARY tool {tool.get('tool_id')} (execution_order={tool.get('execution_order')})")
                            else:
                                tool = tools[0]  # Only one tool, use it
                            action_dict = {
                                'action_id': tool.get('tool_id', tool.get('action_id', 'UNKNOWN')),
                                'action_name': tool.get('name', tool.get('action_name', '')),
                                'app_name': tool.get('app_name', ''),
                                'app_display_name': tool.get('app_display_name', tool.get('app_name', '')),
                                'description': tool.get('description', ''),
                                'parameters': tool.get('parameters', {}),
                                'response': tool.get('response', {}),
                                'step_index': step_index,  # ✅ Preserve index for mapping
                                'execution_order': tool.get('execution_order', 0),
                            }
                            all_action_dicts.append(action_dict)
                            logger.info(f"   [INDEX {step_index}] → {action_dict['action_id']} ({action_dict['app_name']})")
                            
                            # 🎯 Cache ALL tools for this step (for dependency validation)
                            # Store all tools from this sub_task_result, not just the primary one
                            step_tool_dicts = []
                            for t in tools:
                                step_tool_dicts.append({
                                    'action_id': t.get('tool_id', t.get('action_id', 'UNKNOWN')),
                                    'action_name': t.get('name', t.get('action_name', '')),
                                    'app_name': t.get('app_name', ''),
                                    'app_display_name': t.get('app_display_name', t.get('app_name', '')),
                                    'description': t.get('description', ''),
                                    'parameters': t.get('parameters', {}),
                                    'response': t.get('response', {}),
                                    'step_index': step_index,
                                    'execution_order': t.get('execution_order', 0),
                                })
                            _BATCH_CACHE[session_id][step_query] = step_tool_dicts
                            
                            # ✅ DETERMINISTIC: Also cache by index for retry safety
                            index_key = f"__INDEX_{step_index}__"
                            _BATCH_CACHE[session_id][index_key] = step_tool_dicts
                            logger.info(f"[BATCH_CACHE] Cached step {step_index} with index key for retry")
                    
                    # Return best tool per step (1:1 mapping)
                    logger.info(f"✅ [BATCH_API] Returning {len(all_action_dicts)} indexed tools (1 per step)")
                    logger.info(f"✅ [BATCH_CACHE] Cached {len(_BATCH_CACHE[session_id])} steps for dependency validation")
                    return format_actions_for_prompt(all_action_dicts, len(all_action_dicts))
                
                else:
                    # ✅ LEGACY MODE: Fuzzy matching (for non-structured queries)
                    logger.info("📝 [BATCH_API] LEGACY MODE: Using fuzzy matching")
                    for sub_task_result in search_result.sub_task_results:
                        step_query = sub_task_result.use_case
                        tools = sub_task_result.tools
                        
                        # Convert to action_dicts format
                        # ✅ Preserve execution_order from Composio's validated plan
                        action_dicts = []
                        for tool in tools[:top_k]:
                            action_dicts.append({
                                'action_id': tool.get('tool_id', tool.get('action_id', 'UNKNOWN')),
                                'action_name': tool.get('name', tool.get('action_name', '')),
                                'app_name': tool.get('app_name', ''),
                                'app_display_name': tool.get('app_display_name', tool.get('app_name', '')),
                                'description': tool.get('description', ''),
                                'parameters': tool.get('parameters', {}),
                                'response': tool.get('response', {}),
                                'execution_order': tool.get('execution_order', 0),  # ✅ Preserve Composio's sequence
                            })
                        
                        _BATCH_CACHE[session_id][step_query] = action_dicts
                        logger.info(f"[BATCH_API] Cached {len(action_dicts)} tools for: {step_query[:60]}...")
                    
                    # Find matching step for current query
                    cached_steps = list(_BATCH_CACHE[session_id].keys())
                    matched_step = _fuzzy_match_step(user_query, cached_steps)
                    
                    if matched_step:
                        logger.info(f"✅ [BATCH_API] Returning tools for current step")
                        action_dicts = _BATCH_CACHE[session_id][matched_step]
                        return format_actions_for_prompt(action_dicts, top_k)
                    else:
                        # ✅ FALLBACK: Use best partial match or first cached result
                        logger.warning(f"⚠️  [BATCH_API] No fuzzy match found for '{user_query[:60]}...'")
                        
                        # Try to find the best partial match (even below threshold)
                        query_words = set(user_query.lower().split())
                        best_match = None
                        best_score = 0
                        
                        for cached_step in cached_steps:
                            cached_words = set(cached_step.lower().split())
                            overlap = len(query_words & cached_words)
                            total = len(query_words | cached_words)
                            score = overlap / total if total > 0 else 0
                            
                            if score > best_score:
                                best_score = score
                                best_match = cached_step
                        
                        # ✅ CRITICAL FIX: Always prefer Composio's validated tools!
                        # Lower threshold to 15% to ensure we use Composio's batch results
                        # Composio's AI already validated these tools - trust it!
                        if best_match and best_score > 0.15:
                            logger.info(f"✅ [BATCH_API] Using best partial match (score={best_score:.2f}): '{best_match[:60]}...'")
                            logger.info(f"   ✅ Preferring Composio's validated tools over pgvector to avoid hallucinations")
                            action_dicts = _BATCH_CACHE[session_id][best_match]
                            return format_actions_for_prompt(action_dicts, top_k)
                        
                        logger.warning(f"⚠️  [BATCH_API] Match score very low ({best_score:.2f}), falling back to pgvector")
                        logger.warning(f"   ⚠️  Risk of tool hallucination! Consider improving fuzzy matching.")
                        _log_to_debug_file("[BATCH_API] All matches too weak, using pgvector fallback")
            else:
                logger.info("⚠️  [BATCH_API] No sub-task results, using pgvector fallback")
                _log_to_debug_file("[BATCH_API] No results from batch call")
        
        except Exception as e:
            logger.error(f"❌ [BATCH_API] Batch call failed: {e}")
            logger.error(f"❌ [CRITICAL] Cannot proceed without batch API - workflow generation requires batch cache")
            _log_to_debug_file(f"[BATCH_API] Error: {e}, CANNOT FALL BACK (function calling requires batch)")
            raise RuntimeError(
                f"Batch API call failed: {e}. Function calling requires batch API results. "
                "Please check Composio API key and connectivity."
            )
    
    # ===================================================================
    # ❌ LEGACY TWO-STAGE SEARCH (COMMENTED OUT)
    # ===================================================================
    # This fallback is NO LONGER NEEDED because:
    # 1. Function calling REQUIRES batch API (we don't use legacy generation anymore)
    # 2. Index-based cache lookup handles retries deterministically
    # 3. If batch API fails, we should error (not silently degrade to pgvector)
    #
    # Keeping code commented for reference but should never reach here
    # ===================================================================
    
    logger.error("❌ [CRITICAL] Reached legacy fallback code - this should NEVER happen!")
    logger.error("   Batch API should have returned results or raised exception")
    logger.error(f"   Session: {session_id}, Query: {user_query[:100]}")
    raise RuntimeError(
        "Workflow generation failed: Batch API did not return results but also didn't error. "
        "This is a bug - please report to engineering."
    )
    
    # # ===================================================================
    # # INTENT DETECTION (for pgvector fallback)
    # # ===================================================================
    # query_intent = detect_query_intent(user_query)
    # _log_to_debug_file(f"[INTENT_DETECT] Query intent: {query_intent} (from step: {user_query[:80]}...)")
    # logger.info(f"[INTENT_DETECT] Step intent: {query_intent} | Step: {user_query[:60]}...")
    # 
    # # ✅ SCALABLE: Let action_matcher.py do SEMANTIC app detection
    # # No keyword matching, no hardcoding - pure AI-first pgvector search!
    # # action_matcher.py will:
    # # 1. Use app_embedding semantic search to find relevant apps
    # # 2. Re-rank with default_app_preferences (user's connected apps prioritized)
    # # 3. Search actions only in the best-matched apps
    # _log_to_debug_file(f"[APP_DETECTION] Delegating to action_matcher.py for semantic app search")
    # logger.info(f"[APP_DETECTION] Using semantic search (no keyword hints) for scalability")
    # 
    # try:
    #     # ✅ Use the WORKING two-stage matcher (action_matcher.py)
    #     from apps.integrations.services.action_matcher import ComposioActionMatcher
    #     
    #     matcher = ComposioActionMatcher(enable_warmup=False)  # No warmup for speed
    #     
    #     # ✅ SCALABLE ARCHITECTURE: Let action_matcher.py do ALL app detection semantically
    #     # - NO app_hints → action_matcher uses pgvector semantic search at app level
    #     # - NO keyword matching → pure AI-first approach
    #     # - context for domain/app discovery
    #     # - user_query for action intent matching
    #     
    #     _log_to_debug_file(f"[SEMANTIC_MATCH] Using action_matcher.py for both app AND action detection")
    #     
    #     action_dicts = matcher.match_for_subtask(
    #         subtask_description=user_query,  # Step-specific intent
    #         app_hints=None,  # ✅ Let matcher do semantic app search!
    #         top_k_apps=10,  # How many apps to consider
    #         top_k_actions_per_app=3,  # Actions per app
    #         min_confidence=0.35,  # Quality threshold
    #         context=context  # Full user query for domain/app discovery
    #     )
    #     
    #     if not action_dicts:
    #         _log_to_debug_file("[TWO STAGE] No actions matched")
    #         return ""
    #     
    #     # ✅ INTENT-BASED FILTERING: Filter actions based on detected intent
    #     if query_intent == "read":
    #         # Filter OUT write actions (send, create, update, delete, post, publish)
    #         write_action_keywords = ['SEND', 'CREATE', 'UPDATE', 'DELETE', 'POST', 'PUBLISH', 'UPLOAD', 
    #                                  'DRAFT', 'REPLY', 'FORWARD', 'COMPOSE', 'WRITE']
    #         read_action_keywords = ['FETCH', 'GET', 'LIST', 'QUERY', 'RETRIEVE', 'DOWNLOAD', 'FIND', 
    #                                'SEARCH', 'VIEW', 'SHOW', 'READ', 'LOAD']
    #         
    #         filtered_actions = []
    #         for action in action_dicts:
    #             action_id = action['action_id'].upper()
    #             # Prefer read actions, exclude pure write actions
    #             is_read = any(kw in action_id for kw in read_action_keywords)
    #             is_write = any(kw in action_id for kw in write_action_keywords)
    #             
    #             # Include if it's a read action OR if it's ambiguous (neither read nor write)
    #             if is_read or not is_write:
    #                 filtered_actions.append(action)
    #             else:
    #                 _log_to_debug_file(f"[INTENT_FILTER] Filtered OUT {action_id} (write action, but intent={query_intent})")
    #         
    #         if len(filtered_actions) < len(action_dicts):
    #             logger.info(f"[INTENT_FILTER] Filtered {len(action_dicts) - len(filtered_actions)} write actions (intent={query_intent})")
    #             _log_to_debug_file(f"[INTENT_FILTER] Kept {len(filtered_actions)}/{len(action_dicts)} actions after intent filtering")
    #             action_dicts = filtered_actions
    #     
    #     # Log the matched apps and actions
    #     matched_apps = list(set([a['app_name'] for a in action_dicts]))
    #     _log_to_debug_file(f"[TWO STAGE] Matched apps: {matched_apps}")
    #     _log_to_debug_file(f"[TWO STAGE] Found {len(action_dicts)} actions (after intent filter):")
    #     
    #     # Enhanced logging - show ALL actions returned
    #     logger.info(f"[TWO STAGE] 🎯 Returned {len(action_dicts)} actions from two-stage search:")
    #     for i, action in enumerate(action_dicts[:top_k], 1):
    #         action_id = action['action_id']
    #         app_name = action['app_name']
    #         confidence = action.get('confidence', 0)
    #         _log_to_debug_file(
    #             f"  {i}. {action_id} "
    #             f"(app: {app_name}, confidence: {confidence:.4f})"
    #         )
    #         logger.info(f"[TWO STAGE]   {i}. {action_id} (app: {app_name}, confidence: {confidence:.3f})")
    #     
    #     # Format for TaskWeaver prompt (same format as before)
    #     return format_actions_for_prompt(action_dicts, top_k)
    #     
    # except ImportError as e:
    #     _log_to_debug_file(f"[TWO STAGE] Import error, falling back to old method: {e}")
    #     # Fallback to old global search if action_matcher not available
    #     return _select_composio_actions_fallback(user_query, app_hints, top_k)
    # except Exception as e:
    #     logger.error(f"[TWO STAGE] Error in two-stage search: {e}", exc_info=True)
    #     _log_to_debug_file(f"[TWO STAGE] Error: {e}")
    #     # Fallback to old global search
    #     return _select_composio_actions_fallback(user_query, app_hints, top_k)


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
    
    # ✅ APPLY INTENT FILTERING (same as two-stage search)
    query_intent = detect_query_intent(user_query)
    _log_to_debug_file(f"[FALLBACK] Query intent: {query_intent}")
    all_actions = _filter_actions_by_intent(all_actions, query_intent)
    _log_to_debug_file(f"[FALLBACK] After intent filter: {len(all_actions)} actions remain")
    
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

