"""
Composio Action Selector - Intelligent action matching using embeddings.

Follows TaskWeaver's PluginSelector pattern but for Composio ACTIONS.

Problem: TaskWeaver knows to use composio_action plugin, but doesn't know
which specific action ID to use (800+ actions available).

Solution: 
1. Load all Composio actions from DB with embeddings
2. Use semantic similarity to find best matching actions
3. Inject matched actions into TaskWeaver prompt so LLM knows exact action IDs
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Cache for action embeddings
_action_embeddings_cache: Dict[str, List[float]] = {}
_action_catalog_cache: List[Dict] = []


@dataclass
class ComposioAction:
    """Represents a Composio action with embedding"""
    action_id: str
    action_name: str
    app_name: str
    description: str
    embedding: List[float] = field(default_factory=list)
    parameters: Dict = field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Format for inclusion in TaskWeaver prompt"""
        params = list(self.parameters.get('properties', {}).keys())[:5]  # Top 5 params
        required = self.parameters.get('required', [])
        required_str = f" (required: {', '.join(required[:3])})" if required else ""
        return f"- {self.action_id}: {self.description}{required_str}"


class ComposioActionSelector:
    """
    Selects the most relevant Composio actions for a user query.
    
    Uses TaskWeaver's PluginSelector pattern:
    1. Load action catalog with embeddings
    2. Compute similarity between query and actions
    3. Return top-k most relevant actions
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '.action_cache'
        )
        self._actions: List[ComposioAction] = []
        self._embeddings: Dict[str, List[float]] = {}
        self._loaded = False
    
    def load_actions_from_db(self) -> int:
        """Load all Composio actions from database"""
        global _action_catalog_cache
        
        if _action_catalog_cache:
            self._actions = [ComposioAction(**a) for a in _action_catalog_cache]
            return len(self._actions)
        
        try:
            import django
            if not django.apps.apps.ready:
                django.setup()
            
            from apps.integrations.models import ComposioActionSchema
            
            actions = ComposioActionSchema.objects.all().values(
                'action_id', 'action_name', 'app__name', 'description', 'parameters'
            )
            
            for a in actions:
                self._actions.append(ComposioAction(
                    action_id=a['action_id'],
                    action_name=a['action_name'],
                    app_name=a['app__name'] or '',
                    description=a['description'] or '',
                    parameters=a['parameters'] or {}
                ))
            
            # Cache for future use
            _action_catalog_cache = [
                {
                    'action_id': a.action_id,
                    'action_name': a.action_name,
                    'app_name': a.app_name,
                    'description': a.description,
                    'parameters': a.parameters
                }
                for a in self._actions
            ]
            
            logger.info(f"[ACTION_SELECTOR] Loaded {len(self._actions)} Composio actions")
            return len(self._actions)
            
        except Exception as e:
            logger.warning(f"[ACTION_SELECTOR] Could not load from DB: {e}")
            return 0
    
    def compute_embeddings(self) -> int:
        """Compute embeddings for all actions that don't have them"""
        global _action_embeddings_cache
        
        if _action_embeddings_cache:
            self._embeddings = _action_embeddings_cache
            return len(self._embeddings)
        
        actions_to_embed = []
        for action in self._actions:
            if action.action_id not in self._embeddings:
                # Combine action_id, name, and description for embedding
                text = f"{action.action_id}: {action.action_name}. {action.description}"
                actions_to_embed.append((action.action_id, text))
        
        if not actions_to_embed:
            return len(self._embeddings)
        
        try:
            # Try sentence transformers (fast, local)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            texts = [text for _, text in actions_to_embed]
            embeddings = model.encode(texts).tolist()
            
            for i, (action_id, _) in enumerate(actions_to_embed):
                self._embeddings[action_id] = embeddings[i]
            
            _action_embeddings_cache = self._embeddings
            logger.info(f"[ACTION_SELECTOR] Computed {len(actions_to_embed)} embeddings")
            
        except Exception as e:
            logger.warning(f"[ACTION_SELECTOR] Could not compute embeddings: {e}")
        
        return len(self._embeddings)
    
    def select_actions(
        self, 
        user_query: str, 
        app_filter: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[ComposioAction, float]]:
        """
        Select most relevant actions for user query.
        
        Args:
            user_query: The user's request
            app_filter: Optional app name to filter (e.g., "GOOGLESHEETS")
            top_k: Number of actions to return
        
        Returns:
            List of (action, similarity_score) tuples
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get query embedding
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(user_query)
        except Exception as e:
            logger.warning(f"[ACTION_SELECTOR] Could not get query embedding: {e}")
            return []
        
        # Filter actions
        candidate_actions = self._actions
        if app_filter:
            app_filter_upper = app_filter.upper()
            candidate_actions = [
                a for a in self._actions 
                if app_filter_upper in a.action_id.upper()
            ]
        
        if not candidate_actions:
            return []
        
        # Compute similarities
        similarities = []
        for action in candidate_actions:
            if action.action_id not in self._embeddings:
                continue
            
            action_embedding = np.array(self._embeddings[action.action_id])
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                action_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((action, float(sim)))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_actions_for_prompt(
        self, 
        user_query: str,
        app_hints: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Get formatted action list for TaskWeaver prompt.
        
        This injects the relevant action catalog into the LLM context,
        so it knows the exact action IDs to use.
        """
        if not self._loaded:
            self.load_actions_from_db()
            self.compute_embeddings()
            self._loaded = True
        
        all_actions = []
        
        # If app hints provided, get top actions from each app
        if app_hints:
            for app in app_hints:
                actions = self.select_actions(user_query, app_filter=app, top_k=3)
                all_actions.extend(actions)
        else:
            all_actions = self.select_actions(user_query, top_k=top_k)
        
        if not all_actions:
            return ""
        
        # Deduplicate and sort by similarity
        seen = set()
        unique_actions = []
        for action, sim in all_actions:
            if action.action_id not in seen:
                seen.add(action.action_id)
                unique_actions.append((action, sim))
        
        unique_actions.sort(key=lambda x: x[1], reverse=True)
        
        # Format for prompt
        lines = ["## Available Composio Actions (use exact action_id):"]
        for action, sim in unique_actions[:top_k]:
            lines.append(action.to_prompt())
        
        return "\n".join(lines)


# Singleton instance
_selector: Optional[ComposioActionSelector] = None


def get_action_selector() -> ComposioActionSelector:
    """Get or create the action selector"""
    global _selector
    if _selector is None:
        _selector = ComposioActionSelector()
    return _selector


def select_composio_actions(
    user_query: str,
    app_hints: Optional[List[str]] = None,
    top_k: int = 5
) -> str:
    """
    Convenience function to get relevant Composio actions for a query.
    
    Returns formatted string to inject into TaskWeaver prompt.
    """
    selector = get_action_selector()
    return selector.get_actions_for_prompt(user_query, app_hints, top_k)

