"""
Composio Batch Search API
==========================

Standalone wrapper for Composio's batch SEARCH_TOOLS API.
No Django dependencies - works directly within TaskWeaver.

Process:
1. Send full prompt to Composio (let Composio decompose internally)
2. Parse results (tools, validated plans, pitfalls)
3. Return structured result for TaskWeaver

Features:
- Direct prompt sending (no pre-decomposition)
- Batch API integration
- Session management
- Comprehensive result parsing
- Fallback to related tools when primary tools have low confidence

Usage in TaskWeaver plugins:
    from composio_batch_search import AIBatchSearch
    
    searcher = AIBatchSearch()
    result = await searcher.search_batch(
        user_prompt="Send email to user@example.com",
        entity_id="taskweaver_batch",
        session_id=None
    )
    
    for tool in result.all_tools:
        print(f"{tool['tool_id']} - {tool['app_name']}")
"""

import logging
import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


@dataclass
class SubTaskResult:
    """
    Result for a single sub-task from batch API.
    
    Contains tools, validated plan, and guidance from Composio.
    """
    index: int                      # Sub-task index (from batch API)
    use_case: str                   # Original use_case query
    tools: List[Dict[str, Any]]     # Tools for this sub-task
    validated_plan: List[str]       # Execution plan from Composio
    pitfalls: List[str]             # Common errors/warnings
    difficulty: str                 # Difficulty level
    toolkits: List[str]             # Required toolkits/apps
    cached_plan_id: Optional[str] = None  # Cached plan reference


@dataclass
class BatchSearchResult:
    """
    Complete result from batch SEARCH_TOOLS API.
    
    Contains:
    - Sub-task results (tools + guidance per sub-task)
    - All tools (flat list for code analysis)
    - Session info (for continuation)
    - Best practices and guidance
    """
    sub_task_results: List[SubTaskResult]
    all_tools: List[Dict[str, Any]]
    
    # Session management
    session_id: Optional[str]
    session_instructions: str
    
    # Guidance from Composio
    best_practices: str
    next_steps: Dict[str, Any]
    toolkit_statuses: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def total_tools(self) -> int:
        """Total number of unique tools."""
        return len(self.all_tools)
    
    @property
    def total_sub_tasks(self) -> int:
        """Total number of sub-tasks."""
        return len(self.sub_task_results)
    
    @property
    def total_pitfalls(self) -> int:
        """Total number of pitfalls/warnings."""
        return sum(len(r.pitfalls) for r in self.sub_task_results)
    
    def get_sub_task(self, index: int) -> Optional[SubTaskResult]:
        """Get sub-task by index."""
        return next((r for r in self.sub_task_results if r.index == index), None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_sub_tasks': self.total_sub_tasks,
            'total_tools': self.total_tools,
            'total_pitfalls': self.total_pitfalls,
            'session_id': self.session_id,
            'sub_tasks': [
                {
                    'index': r.index,
                    'use_case': r.use_case,
                    'tool_count': len(r.tools),
                    'difficulty': r.difficulty,
                    'has_plan': bool(r.validated_plan),
                    'has_pitfalls': bool(r.pitfalls),
                }
                for r in self.sub_task_results
            ]
        }


class AIBatchSearch:
    """
    Composio batch SEARCH_TOOLS API wrapper.
    
    Features:
    - Direct prompt sending (no decomposition)
    - Batch API calls
    - Result parsing and validation
    - Session management
    - Intelligent tool selection with fallback to related tools
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize batch search service.
        
        Args:
            api_key: Composio API key (defaults to COMPOSIO_API_KEY env var)
        """
        self.api_url = "https://backend.composio.dev/api/v3/tools/execute/COMPOSIO_SEARCH_TOOLS"
        self.api_key = api_key or os.environ.get('COMPOSIO_API_KEY')
        
        if not self.api_key:
            logger.warning("⚠️ [BATCH_SEARCH] COMPOSIO_API_KEY not configured")
    
    async def search_batch(
        self,
        user_prompt: str,
        entity_id: str,
        session_id: Optional[str] = None,
        timeout: int = 60
    ) -> BatchSearchResult:
        """
        Search for tools using batch API.
        
        Process:
        1. Send full prompt to Composio (let Composio handle decomposition)
        2. Parse results with intelligent fallback to related tools
        3. Return structured result
        
        Args:
            user_prompt: Full user request (Composio will decompose internally)
            entity_id: User/company ID for Composio
            session_id: Optional session ID for continuation
            timeout: Request timeout in seconds
            
        Returns:
            BatchSearchResult with tools and guidance
        """
        logger.info(f"🔍 [BATCH_SEARCH] Starting batch search...")
        logger.debug(f"📝 [BATCH_SEARCH] Prompt: {user_prompt[:200]}...")
        
        # ===== STEP 1: BUILD BATCH PAYLOAD =====
        logger.info("📤 [STEP 1] Sending full prompt to Composio (let Composio decompose)...")
        
        # Send the FULL user prompt directly to Composio
        # Let Composio's AI handle decomposition - it has better context!
        queries = [{
            'use_case': user_prompt,
            'known_fields': '',
            'index': 1
        }]
        
        payload = {
            'arguments': {
                'queries': queries,
                'session': {
                    'generate_id': session_id is None,
                    **({'id': session_id} if session_id else {})
                }
            },
            'entity_id': entity_id
        }
        
        # ===== STEP 2: CALL BATCH API =====
        logger.info(f"📤 [STEP 2] Calling Composio batch API with full prompt...")
        
        if not HAS_REQUESTS:
            raise ImportError("requests library not installed")
        
        if not self.api_key:
            raise ValueError("COMPOSIO_API_KEY not configured")
        
        # Call API (use asyncio.to_thread for async compatibility)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                self.api_url,
                json=payload,
                headers={
                    'X-API-Key': self.api_key,
                    'Content-Type': 'application/json'
                },
                timeout=timeout
            )
        )
        
        if response.status_code != 200:
            error_msg = f"Batch API returned {response.status_code}: {response.text[:500]}"
            logger.error(f"❌ [BATCH_SEARCH] {error_msg}")
            raise ValueError(error_msg)
        
        # ===== STEP 3: PARSE RESULTS =====
        logger.info("📥 [STEP 3] Parsing batch API response...")
        
        data = response.json()
        result = self._parse_batch_result(data, user_prompt)
        
        logger.info(
            f"✅ [BATCH_SEARCH] Success!\n"
            f"  - Sub-tasks: {result.total_sub_tasks}\n"
            f"  - Total tools: {result.total_tools}\n"
            f"  - Pitfalls: {result.total_pitfalls}\n"
            f"  - Session: {result.session_id}"
        )
        
        return result
    
    async def search_batch_multi_queries(
        self,
        queries: List[Dict[str, Any]],
        entity_id: str,
        session_id: Optional[str] = None,
        timeout: int = 60
    ) -> BatchSearchResult:
        """
        Search for tools using batch API with multiple pre-parsed queries.
        Each query gets tools independently (better than letting Composio decompose).
        
        Args:
            queries: List like [{'use_case': 'Search flights', 'known_fields': '', 'index': 2}, ...]
            entity_id: User/company ID for Composio
            session_id: Optional session ID
            timeout: Request timeout in seconds
            
        Returns:
            BatchSearchResult with tools for each query
        """
        logger.info(f"🔍 [BATCH_MULTI] Starting batch search with {len(queries)} queries...")
        
        # Build payload
        payload = {
            'arguments': {
                'queries': queries,
                'session': {
                    'generate_id': session_id is None,
                    **({'id': session_id} if session_id else {})
                }
            },
            'entity_id': entity_id
        }
        
        logger.info(f"📤 [BATCH_MULTI] Calling Composio API...")
        
        if not HAS_REQUESTS:
            raise ImportError("requests library not installed")
        
        if not self.api_key:
            raise ValueError("COMPOSIO_API_KEY not configured")
        
        # Call API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                self.api_url,
                json=payload,
                headers={
                    'X-API-Key': self.api_key,
                    'Content-Type': 'application/json'
                },
                timeout=timeout
            )
        )
        
        if response.status_code != 200:
            error_msg = f"Batch API returned {response.status_code}: {response.text[:500]}"
            logger.error(f"❌ [BATCH_MULTI] {error_msg}")
            raise ValueError(error_msg)
        
        # Parse results
        logger.info("📥 [BATCH_MULTI] Parsing response...")
        data = response.json()
        
        # Use first query as user_prompt for logging
        user_prompt = queries[0]['use_case'] if queries else ''
        result = self._parse_batch_result(data, user_prompt)
        
        logger.info(
            f"✅ [BATCH_MULTI] Success!\n"
            f"  - Queries sent: {len(queries)}\n"
            f"  - Sub-tasks returned: {result.total_sub_tasks}\n"
            f"  - Total tools: {result.total_tools}"
        )
        
        return result
    
    def _parse_batch_result(self, data: Dict[str, Any], user_prompt: str) -> BatchSearchResult:
        """
        Parse batch API response into structured result.
        
        Args:
            data: Raw response from Composio API
            user_prompt: Original user prompt (for intent filtering)
            
        Returns:
            BatchSearchResult with parsed data
        """
        data_section = data.get('data', {})
        
        results_data = data_section.get('results', [])
        tool_schemas = data_section.get('tool_schemas', {})
        session_info = data_section.get('session', {})
        toolkit_statuses = data_section.get('toolkit_connection_statuses', [])
        best_practices = data_section.get('best_practices', '')
        next_steps = data_section.get('next_steps', {})
        
        logger.info(f"📦 [PARSE] Processing {len(results_data)} sub-task results...")
        
        sub_task_results = []
        all_tools = []
        seen_tool_ids = set()
        
        for result_data in results_data:
            index = result_data.get('index', 0)
            use_case = result_data.get('use_case', '')
            
            # ✅ SIMPLIFIED: Use ONE variable name - tool_slugs
            # Composio returns tools in TWO fields: primary (main task) + related (complementary)
            primary_tools = result_data.get('primary_tool_slugs', []) or result_data.get('main_tool_slugs', []) or []
            related_tools = result_data.get('related_tool_slugs', []) or []
            
            # Merge both into single list
            tool_slugs = list(primary_tools) + list(related_tools)
            
            validated_plan = result_data.get('validated_plan', [])
            pitfalls = result_data.get('pitfalls', [])
            difficulty = result_data.get('difficulty', 'unknown')
            toolkits = result_data.get('toolkits', [])
            cached_plan_id = result_data.get('cached_plan_id')
            
            logger.info(
                f"  📋 Sub-task {index}: {use_case}\n"
                f"     Tools: {len(tool_slugs)}, "
                f"Difficulty: {difficulty}, "
                f"Pitfalls: {len(pitfalls)}"
            )
            
            # DEBUG: Log what tools Composio actually returned
            if tool_slugs:
                logger.info(f"  🔧 [DEBUG] Composio returned tools: {tool_slugs}")
            
            # Get tools for this sub-task
            # ✅ TRUST COMPOSIO'S ORDERING: Use ALL tools in the order Composio provides
            # Composio already validated the plan and determined the correct execution sequence!
            sub_task_tools = []
            
            # ✅ NEW APPROACH: Preserve ALL tools in Composio's validated order
            if tool_slugs:
                logger.info(f"  📦 [ORDER] Processing {len(tool_slugs)} tools in Composio's sequence...")
                
                for tool_index, tool_slug in enumerate(tool_slugs, 1):
                    if tool_slug in tool_schemas:
                        schema = tool_schemas[tool_slug]
                        
                        tool_obj = {
                            'tool_id': schema.get('tool_slug', tool_slug),
                            'app_name': schema.get('toolkit', '').lower(),
                            'description': schema.get('description', ''),
                            'parameters': schema.get('input_schema', {}),
                            'response_schema': schema.get('response_schema', {}) or schema.get('output_schema', {}),
                            'execution_order': tool_index,  # ✅ Preserve Composio's sequence!
                        }
                        
                        sub_task_tools.append(tool_obj)
                        
                        # Add to all_tools (deduplicate)
                        if tool_slug not in seen_tool_ids:
                            all_tools.append(tool_obj)
                            seen_tool_ids.add(tool_slug)
                            logger.info(f"     ✓ Tool {tool_index}: {tool_slug} ({tool_obj['app_name']})")
                    else:
                        logger.warning(f"⚠️ [PARSE] Tool schema not found for: {tool_slug}")
            else:
                logger.warning(f"⚠️ [PARSE] No tools returned for: {use_case}")
            
            sub_task_results.append(SubTaskResult(
                index=index,
                use_case=use_case,
                tools=sub_task_tools,
                validated_plan=validated_plan,
                pitfalls=pitfalls,
                difficulty=difficulty,
                toolkits=toolkits,
                cached_plan_id=cached_plan_id
            ))
        
        logger.info(
            f"✅ [PARSE] Extracted {len(all_tools)} unique tools from "
            f"{len(sub_task_results)} sub-tasks"
        )
        
        # ===== SCALABLE SOLUTION: Trust Composio's validated execution plan =====
        # We preserve ALL tools in the order Composio provided - it's already optimized!
        # No filtering/deduplication needed - Composio validated the entire workflow!
        relevant_tools = all_tools
        
        # Calculate tools per sub-task
        tools_per_subtask = len(relevant_tools) / len(sub_task_results) if sub_task_results else 0
        logger.info(f"✅ [RESULT] Using {len(relevant_tools)} tools ({tools_per_subtask:.1f} avg per sub-task)")
        for i, tool in enumerate(relevant_tools, 1):
            execution_order = tool.get('execution_order', '?')
            logger.info(f"   ✓ {i}. {tool['tool_id']} ({tool['app_name']}) [order: {execution_order}]")
        
        return BatchSearchResult(
            sub_task_results=sub_task_results,
            all_tools=relevant_tools,
            session_id=session_info.get('id'),
            session_instructions=session_info.get('instructions', ''),
            best_practices=best_practices,
            next_steps=next_steps,
            toolkit_statuses=toolkit_statuses
        )
    
    def _find_best_matching_tool(
        self, 
        use_case: str, 
        tool_slugs: List[str], 
        tool_schemas: Dict[str, Any]
    ) -> tuple:
        """
        Find the tool whose action best matches the use case (100% dynamic).
        
        Strategy: Extract action verbs from both use_case and tool_ids,
        then score based on verb overlap.
        
        Args:
            use_case: The user's requested action (e.g., "send email")
            tool_slugs: List of tool IDs from Composio
            tool_schemas: Tool schema dictionary
            
        Returns:
            Tuple of (best_tool_slug, score), or (first_tool, score) as fallback
        """
        use_case_words = set(use_case.lower().split())
        
        # Score each tool
        scored_tools = []
        
        for tool_slug in tool_slugs:
            if tool_slug not in tool_schemas:
                continue
                
            tool_id = tool_slug.lower()
            description = tool_schemas[tool_slug].get('description', '').lower()
            
            # Extract words from tool_id
            tool_words = set(tool_id.replace('_', ' ').split())
            
            # Calculate overlap score
            exact_matches = use_case_words & tool_words
            desc_matches = sum(1 for word in use_case_words if word in description)
            
            # Scoring:
            # - Exact word match in tool_id: +10 per word
            # - Word appears in description: +5 per word
            score = (len(exact_matches) * 10) + (desc_matches * 5)
            
            scored_tools.append((score, tool_slug))
            
            logger.debug(
                f"[MATCH] '{use_case}' vs '{tool_slug}': "
                f"score={score} (exact={exact_matches}, desc={desc_matches})"
            )
        
        # Sort by score (highest first)
        scored_tools.sort(reverse=True, key=lambda x: x[0])
        
        if scored_tools:
            best_score, best_tool = scored_tools[0]
            if best_score > 0:
                logger.info(
                    f"✅ [MATCH] Best tool for '{use_case}': {best_tool} (score={best_score})"
                )
                return best_tool, best_score
        
        # Fallback: return first tool with score=0
        logger.warning(
            f"⚠️ [MATCH] No good match for '{use_case}', using first tool: {tool_slugs[0]}"
        )
        return (tool_slugs[0], 0) if tool_slugs else (None, 0)

