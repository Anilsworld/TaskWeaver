"""
LangGraph-Native Workflow Generation Strategies

ARCHITECTURE:
- Leverages LangGraph's native sub-graph and composition features
- Extends existing WorkflowToLangGraphAdapter instead of reinventing
- Uses LangGraph checkpointing for resumable generation

SCALABILITY IMPACT:
- Unlimited workflow size through native sub-graphs
- Built-in parallel execution
- Native state management

INTEGRATION POINTS:
- Works with existing WorkflowToLangGraphAdapter
- Uses WorkflowComplexityDetector for analysis
- Coordinates with Planner for multi-stage generation
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass

from .workflow_complexity_detector import ComplexityAnalysis, WorkflowComplexityDetector

logger = logging.getLogger(__name__)


@dataclass
class LangGraphGenerationStrategy:
    """
    LangGraph-native generation strategy.
    
    Uses LangGraph's built-in features instead of custom implementations:
    - Sub-graphs for hierarchical workflows
    - Graph composition for merging
    - Checkpointing for resumable generation
    """
    
    mode: Literal["ALL_IN_ONE", "COMPOSE_GRAPHS", "SUBGRAPH_HIERARCHY"]
    max_nodes_per_graph: int
    use_subgraphs: bool
    requires_user_review: bool
    planner_instructions: str


class LangGraphNativeStrategies:
    """
    Factory for LangGraph-native workflow generation strategies.
    
    Instead of building custom merger/expander classes, this uses
    LangGraph's built-in capabilities:
    
    1. ALL_IN_ONE: Single StateGraph (< 10 nodes)
    2. COMPOSE_GRAPHS: Multiple StateGraphs composed with |
    3. SUBGRAPH_HIERARCHY: Main graph with sub-graph nodes
    """
    
    def __init__(self):
        self.complexity_detector = WorkflowComplexityDetector()
    
    def select_strategy(self, user_request: str, context: Optional[str] = None) -> LangGraphGenerationStrategy:
        """
        Analyze request and select LangGraph-native strategy.
        
        Args:
            user_request: User's workflow generation request
            context: Optional additional context
        
        Returns:
            LangGraphGenerationStrategy with mode and instructions
        """
        # Analyze complexity
        complexity = self.complexity_detector.analyze(user_request, context)
        
        logger.info(
            f"🎯 [LANGGRAPH_STRATEGY] Detected {complexity.complexity_level} workflow "
            f"({complexity.estimated_nodes} nodes)"
        )
        
        # Select LangGraph-native strategy
        if complexity.recommended_strategy == "ALL_IN_ONE":
            return self._create_all_in_one_strategy(complexity)
        elif complexity.recommended_strategy == "LOGICAL_CHUNKS":
            return self._create_compose_graphs_strategy(complexity)
        else:  # HIERARCHICAL
            return self._create_subgraph_strategy(complexity)
    
    def _create_all_in_one_strategy(self, complexity: ComplexityAnalysis) -> LangGraphGenerationStrategy:
        """
        Strategy 1: Single StateGraph (Simple workflows < 10 nodes)
        
        LANGGRAPH USAGE:
        - Standard StateGraph.add_node() for all nodes
        - Single graph compilation
        - No composition or sub-graphs needed
        """
        return LangGraphGenerationStrategy(
            mode="ALL_IN_ONE",
            max_nodes_per_graph=50,
            use_subgraphs=False,
            requires_user_review=False,
            planner_instructions="""
⚠️ ALL-IN-ONE GENERATION (LangGraph Single StateGraph):

Generate ONE complete WORKFLOW dict with ALL nodes.
The system will convert it to a single LangGraph StateGraph.

Your message to CodeInterpreter:
"[Describe complete workflow with all steps: forms → processing → approval → notifications]"

After CodeInterpreter returns WORKFLOW dict, mark stop="Completed".

LangGraph will handle:
- State management (WorkflowState)
- Checkpointing (PostgreSQL)
- Parallel execution (parallel_groups)
- HITL (interrupt())
""".strip()
        )
    
    def _create_compose_graphs_strategy(self, complexity: ComplexityAnalysis) -> LangGraphGenerationStrategy:
        """
        Strategy 2: Graph Composition (Complex workflows 10-20 nodes)
        
        LANGGRAPH USAGE:
        - Generate multiple WORKFLOW dicts (logical chunks)
        - Convert each to StateGraph
        - Compose using: graph_final = graph1 | graph2 | graph3
        - LangGraph handles state merging automatically!
        
        NO CUSTOM MERGER NEEDED - LangGraph does it natively!
        """
        chunks = complexity.chunk_suggestions or [
            "Data Collection",
            "Processing & Analysis",
            "Approval & Notifications"
        ]
        
        return LangGraphGenerationStrategy(
            mode="COMPOSE_GRAPHS",
            max_nodes_per_graph=10,
            use_subgraphs=False,
            requires_user_review=False,
            planner_instructions=f"""
⚠️ COMPOSE_GRAPHS MODE (LangGraph Native Composition):

This workflow will be generated in {len(chunks)} logical chunks.
Each chunk becomes a separate StateGraph, then composed with |.

CHUNKS:
{self._format_chunks(chunks)}

CURRENT CHUNK: (will be injected dynamically)

For this chunk:
1. Send message to CodeInterpreter for ONLY this chunk's nodes
2. System converts to StateGraph_{{chunk_number}}
3. Move to next chunk

After all chunks:
- System composes: final_graph = graph1 | graph2 | graph3
- LangGraph handles state merging automatically
- Single checkpointed execution

LangGraph composition benefits:
- Automatic state merging
- Preserved execution order
- Single checkpoint stream
- No manual edge connections needed!
""".strip()
        )
    
    def _create_subgraph_strategy(self, complexity: ComplexityAnalysis) -> LangGraphGenerationStrategy:
        """
        Strategy 3: Sub-graph Hierarchy (Enterprise workflows 20+ nodes)
        
        LANGGRAPH USAGE:
        - Generate high-level WORKFLOW dict (5-10 abstract nodes)
        - Each abstract node becomes a sub-graph placeholder
        - User selects nodes to expand
        - System generates detailed WORKFLOW for sub-graph
        - LangGraph: main_graph.add_node("processing", subgraph)
        
        NO CUSTOM EXPANDER NEEDED - LangGraph sub-graphs are native!
        """
        return LangGraphGenerationStrategy(
            mode="SUBGRAPH_HIERARCHY",
            max_nodes_per_graph=5,
            use_subgraphs=True,
            requires_user_review=True,
            planner_instructions=f"""
⚠️ SUBGRAPH_HIERARCHY MODE (LangGraph Native Sub-graphs):

PHASE 1 - HIGH-LEVEL STRUCTURE:
Generate abstract workflow with 5-10 main nodes.
Each node represents a major phase (e.g., "Data Processing", "Multi-Stage Approval").

System converts to:
```python
main_graph = StateGraph(State)
main_graph.add_node("data_processing", placeholder)  # Will expand later
main_graph.add_node("approval_flow", placeholder)
...
```

Mark stop="InProcess" for user review.

PHASE 2 - EXPANSION (user selects node):
User: "Expand 'data_processing' node"

Generate detailed WORKFLOW for that node only.
System creates sub-graph and replaces placeholder:
```python
processing_subgraph = StateGraph(State)
# ... add detailed nodes ...
main_graph.add_node("data_processing", processing_subgraph.compile())
```

LangGraph sub-graph benefits:
- State isolation (each sub-graph has own state scope)
- Checkpointing at both levels (main + sub)
- Parallel sub-graph execution
- Unlimited nesting depth
- No manual edge rewiring!

Repeat until all abstract nodes are expanded.
""".strip()
        )
    
    def _format_chunks(self, chunks: List[str]) -> str:
        """Format chunk list for prompt"""
        return "\n".join(f"  {i+1}. {chunk}" for i, chunk in enumerate(chunks))
    
    def get_langgraph_composition_guide(self) -> str:
        """
        Get guide for using LangGraph native composition.
        
        Returns:
            Technical guide for implementing composition
        """
        return """
# LangGraph Native Composition Guide

## Option 1: ALL_IN_ONE (Simple)
```python
from langgraph.graph import StateGraph

builder = StateGraph(WorkflowState)
builder.add_node("form", form_node)
builder.add_node("search", search_node)
builder.add_node("approval", approval_node)
builder.add_edge("form", "search")
builder.add_edge("search", "approval")
graph = builder.compile(checkpointer=postgres_checkpointer)
```

## Option 2: COMPOSE_GRAPHS (Complex)
```python
# Build chunk 1
builder1 = StateGraph(WorkflowState)
builder1.add_node("form", form_node)
builder1.add_node("validate", validate_node)
graph1 = builder1.compile()

# Build chunk 2
builder2 = StateGraph(WorkflowState)
builder2.add_node("search", search_node)
builder2.add_node("process", process_node)
graph2 = builder2.compile()

# Compose (LangGraph native!)
final_graph = graph1 | graph2  # Automatic state merging!
final_graph = final_graph.compile(checkpointer=postgres_checkpointer)
```

## Option 3: SUBGRAPH_HIERARCHY (Enterprise)
```python
# Create sub-graph for data processing
processing_builder = StateGraph(WorkflowState)
processing_builder.add_node("fetch_a", fetch_a_node)
processing_builder.add_node("fetch_b", fetch_b_node)
processing_subgraph = processing_builder.compile()

# Create main graph with sub-graph
main_builder = StateGraph(WorkflowState)
main_builder.add_node("collect", collect_node)
main_builder.add_node("processing", processing_subgraph)  # Sub-graph as node!
main_builder.add_node("approval", approval_node)
main_builder.add_edge("collect", "processing")
main_builder.add_edge("processing", "approval")

main_graph = main_builder.compile(checkpointer=postgres_checkpointer)
```

## Benefits:
- ✅ No custom merger/expander code
- ✅ Automatic state management
- ✅ Built-in checkpointing at all levels
- ✅ Native parallel execution
- ✅ Clean, maintainable code
""".strip()

