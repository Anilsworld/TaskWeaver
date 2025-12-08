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
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cached OpenAI client and config
_openai_client = None
_embedding_deployment = None


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
        
        # Cache for 1 hour
        cache.set(cache_key, embedding, timeout=3600)
        
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
    
    # ✅ Try pgvector indexed search first (FAST!)
    try:
        from pgvector.django import CosineDistance
        from django.db.utils import ProgrammingError
        
        # Build query
        queryset = ComposioActionSchema.objects.filter(
            is_deprecated=False,
            description_embedding__isnull=False
        )
        
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
            'distance'
        )
        
        results = []
        for action in actions:
            similarity = 1.0 - action['distance']  # Convert distance to similarity
            results.append({
                'action_id': action['action_id'],
                'action_name': action['action_name'],
                'description': action['description'] or '',
                'parameters': action['parameters_schema'] or {},
                'similarity': similarity
            })
        
        logger.info(f"[ACTION_SELECTOR] ⚡ pgvector returned {len(results)} actions in ms")
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
        
        # Only load top 500 most-used actions (not all 17k!)
        queryset = ComposioActionSchema.objects.filter(
            is_deprecated=False,
            description_embedding__isnull=False
        ).order_by('-usage_count')[:500]
        
        if app_filter:
            queryset = queryset.filter(action_id__icontains=app_filter.upper())
        
        actions = list(queryset.values(
            'action_id', 'action_name', 'description', 
            'parameters_schema', 'description_embedding'
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
    """Format actions for injection into TaskWeaver prompt"""
    if not actions:
        return ""
    
    lines = ["## Available Composio Actions (use exact action_id):"]
    
    for action in actions[:top_k]:
        params = action.get('parameters', {})
        required = params.get('required', [])
        required_str = f" (required: {', '.join(required[:3])})" if required else ""
        desc = (action.get('description') or '')[:100]  # Truncate long descriptions
        lines.append(f"- {action['action_id']}: {desc}{required_str}")
    
    return "\n".join(lines)


def select_composio_actions(
    user_query: str,
    app_hints: Optional[List[str]] = None,
    top_k: int = 5
) -> str:
    """
    ⚡ Main entry point - Get relevant Composio actions for a query.
    
    Uses pgvector for fast indexed search (milliseconds).
    Returns formatted string to inject into TaskWeaver prompt.
    """
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
    
    return format_actions_for_prompt(unique_actions, top_k)

