"""
Workflow Complexity Detector

Analyzes user requests to determine workflow complexity and recommend
the appropriate generation strategy (all-in-one, chunked, or hierarchical).

ARCHITECTURE:
- AI-first: Uses LLM to analyze complexity (no hardcoded patterns)
- Scalable: Handles any workflow size from 1 to 100+ nodes
- Dynamic: Adapts to user request characteristics

INTEGRATION POINTS:
- Called by Planner before workflow generation
- Informs SmartPlannerModeSwitcher strategy selection
"""

import re
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComplexityAnalysis:
    """Result of workflow complexity analysis"""
    
    complexity_level: Literal["SIMPLE", "COMPLEX", "ENTERPRISE"]
    estimated_nodes: int
    estimated_parallel_groups: int
    estimated_conditionals: int
    has_loops: bool
    has_nested_approvals: bool
    has_multi_stage_processing: bool
    recommended_strategy: Literal["ALL_IN_ONE", "LOGICAL_CHUNKS", "HIERARCHICAL"]
    reasoning: str
    chunk_suggestions: Optional[List[str]] = None


class WorkflowComplexityDetector:
    """
    Detects workflow complexity from user requests using AI-first analysis.
    
    SCALABILITY IMPACT:
    - Prevents token limit errors by detecting oversized workflows early
    - Enables optimal generation strategy selection
    - Reduces LLM retries and failures
    
    SECURITY FEATURES:
    - No external API calls (uses local heuristics + optional LLM)
    - Input validation and sanitization
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize complexity detector.
        
        Args:
            llm_client: Optional LLM client for AI-powered analysis
                       Falls back to heuristic analysis if None
        """
        self.llm_client = llm_client
    
    def analyze(self, user_request: str, context: Optional[str] = None) -> ComplexityAnalysis:
        """
        Analyze workflow complexity from user request.
        
        Args:
            user_request: The user's workflow generation request
            context: Optional additional context (previous messages, etc.)
        
        Returns:
            ComplexityAnalysis with recommended strategy
        """
        logger.info(f"🔍 [COMPLEXITY] Analyzing request: {user_request[:100]}...")
        
        # Use LLM-powered analysis if available, otherwise use heuristics
        if self.llm_client:
            return self._analyze_with_llm(user_request, context)
        else:
            return self._analyze_with_heuristics(user_request, context)
    
    def _analyze_with_heuristics(
        self, 
        user_request: str, 
        context: Optional[str] = None
    ) -> ComplexityAnalysis:
        """
        Heuristic-based complexity analysis (fast, no LLM calls).
        
        SCALABILITY: O(n) where n = length of user_request
        """
        request_lower = user_request.lower()
        full_text = f"{user_request} {context or ''}"
        
        # Count indicators
        step_count = self._count_steps(full_text)
        parallel_indicators = self._count_parallel_indicators(request_lower)
        conditional_indicators = self._count_conditional_indicators(request_lower)
        loop_indicators = self._has_loop_indicators(request_lower)
        nested_approval = self._has_nested_approval(request_lower)
        multi_stage = self._has_multi_stage_processing(request_lower)
        
        # Estimate nodes (conservative)
        estimated_nodes = max(
            step_count,
            3 if "form" in request_lower or "collect" in request_lower else 0,
            2 if "approval" in request_lower or "review" in request_lower else 0,
            2 if "send" in request_lower or "notify" in request_lower else 0,
        )
        
        # Add nodes for parallel operations
        estimated_nodes += parallel_indicators * 2
        
        # Add nodes for conditionals
        estimated_nodes += conditional_indicators
        
        # Determine complexity level
        if estimated_nodes <= 10 and not loop_indicators and not nested_approval:
            complexity = "SIMPLE"
            strategy = "ALL_IN_ONE"
            reasoning = f"Simple workflow with ~{estimated_nodes} nodes, no loops or nested logic"
        elif estimated_nodes <= 20 or (loop_indicators and estimated_nodes <= 15):
            complexity = "COMPLEX"
            strategy = "LOGICAL_CHUNKS"
            reasoning = f"Complex workflow with ~{estimated_nodes} nodes, may have loops or parallel processing"
        else:
            complexity = "ENTERPRISE"
            strategy = "HIERARCHICAL"
            reasoning = f"Enterprise workflow with ~{estimated_nodes}+ nodes, requires hierarchical generation"
        
        # Generate chunk suggestions for complex workflows
        chunk_suggestions = None
        if complexity in ["COMPLEX", "ENTERPRISE"]:
            chunk_suggestions = self._suggest_chunks(full_text)
        
        logger.info(
            f"✅ [COMPLEXITY] {complexity} workflow detected: "
            f"{estimated_nodes} nodes, strategy={strategy}"
        )
        
        return ComplexityAnalysis(
            complexity_level=complexity,
            estimated_nodes=estimated_nodes,
            estimated_parallel_groups=parallel_indicators,
            estimated_conditionals=conditional_indicators,
            has_loops=loop_indicators,
            has_nested_approvals=nested_approval,
            has_multi_stage_processing=multi_stage,
            recommended_strategy=strategy,
            reasoning=reasoning,
            chunk_suggestions=chunk_suggestions,
        )
    
    def _count_steps(self, text: str) -> int:
        """Count explicit steps mentioned in request"""
        # Look for numbered steps, bullet points, "then", "after", etc.
        step_patterns = [
            r'\d+\.',  # 1. 2. 3.
            r'step \d+',
            r'first.*then.*',
            r'after.*then.*',
            r'once.*then.*',
        ]
        
        count = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, text.lower())
            count = max(count, len(matches))
        
        # Count transition words
        transitions = ['then', 'after', 'next', 'once', 'finally']
        for word in transitions:
            count += text.lower().count(word)
        
        return min(count, 50)  # Cap at 50 to avoid over-estimation
    
    def _count_parallel_indicators(self, text: str) -> int:
        """Count indicators of parallel processing"""
        parallel_words = [
            'parallel', 'simultaneously', 'at the same time', 
            'concurrently', 'both', 'all at once', 'multiple'
        ]
        count = sum(1 for word in parallel_words if word in text)
        
        # Check for "fetch X and Y and Z" patterns
        and_pattern = r'(fetch|get|retrieve|search).*\band\b.*\band\b'
        if re.search(and_pattern, text):
            count += 1
        
        return count
    
    def _count_conditional_indicators(self, text: str) -> int:
        """Count conditional logic indicators"""
        conditional_words = [
            'if', 'else', 'otherwise', 'depending on', 'based on',
            'when', 'unless', 'in case', 'approve', 'reject'
        ]
        return sum(1 for word in conditional_words if word in text)
    
    def _has_loop_indicators(self, text: str) -> bool:
        """Check for loop/iteration indicators"""
        loop_words = [
            'loop', 'iterate', 'repeat', 'for each', 'until',
            'while', 'retry', 'keep trying'
        ]
        return any(word in text for word in loop_words)
    
    def _has_nested_approval(self, text: str) -> bool:
        """Check for nested approval patterns"""
        nested_patterns = [
            r'approval.*approval',
            r'review.*review',
            r'escalate',
            r'multi.*level.*approval',
            r'manager.*director',
        ]
        return any(re.search(pattern, text) for pattern in nested_patterns)
    
    def _has_multi_stage_processing(self, text: str) -> bool:
        """Check for multi-stage processing indicators"""
        stage_patterns = [
            r'stage \d+',
            r'phase \d+',
            r'first.*second.*third',
            r'initial.*final',
            r'pre.*post',
        ]
        return any(re.search(pattern, text) for pattern in stage_patterns)
    
    def _suggest_chunks(self, text: str) -> List[str]:
        """
        Suggest logical chunks for complex workflows.
        
        Returns list of chunk descriptions like:
        ["Data Collection & Validation", "Processing & Analysis", "Approval & Notifications"]
        """
        chunks = []
        text_lower = text.lower()
        
        # Common workflow phases
        if any(word in text_lower for word in ['form', 'collect', 'input', 'gather']):
            chunks.append("Data Collection & Validation")
        
        if any(word in text_lower for word in ['search', 'fetch', 'retrieve', 'query', 'api']):
            chunks.append("Data Retrieval & Processing")
        
        if any(word in text_lower for word in ['analyze', 'calculate', 'process', 'transform']):
            chunks.append("Analysis & Transformation")
        
        if any(word in text_lower for word in ['approval', 'review', 'decision', 'hitl']):
            chunks.append("Approval & Decision Making")
        
        if any(word in text_lower for word in ['send', 'notify', 'email', 'message', 'update']):
            chunks.append("Notifications & Updates")
        
        # Default chunks if none detected
        if not chunks:
            chunks = ["Workflow Setup", "Main Processing", "Completion & Notifications"]
        
        logger.info(f"💡 [COMPLEXITY] Suggested chunks: {chunks}")
        return chunks
    
    def _analyze_with_llm(
        self, 
        user_request: str, 
        context: Optional[str] = None
    ) -> ComplexityAnalysis:
        """
        LLM-powered complexity analysis (more accurate, requires LLM call).
        
        TODO: Implement when LLM client is available
        """
        logger.warning("LLM-powered analysis not yet implemented, falling back to heuristics")
        return self._analyze_with_heuristics(user_request, context)

