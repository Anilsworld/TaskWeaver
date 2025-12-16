"""
Workflow Intermediate Representation (IR)
==========================================
Structured, typed graph representation for workflows with automatic edge inference.

This module provides:
- Type-safe node and edge representations
- Automatic edge inference from depends_on and data flow references
- DAG validation (cycle detection, reachability)
- Conflict detection (parallel vs sequential, conditional branches)

Architecture:
    WORKFLOW dict (Pydantic) → WorkflowIR (Graph) → Validation → Execution Plan
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple
import re
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Node execution types."""
    AGENT_WITH_TOOLS = auto()  # External tool/API via Composio
    AGENT_ONLY = auto()         # Pure AI processing (no tools)
    HITL = auto()               # Human-in-the-loop approval
    FORM = auto()               # Data collection from user
    CODE_EXECUTION = auto()     # Python code execution
    LOOP = auto()               # Loop/iteration execution
    PARALLEL = auto()           # Parallel execution (deprecated - use dependencies)
    
    @classmethod
    def from_string(cls, type_str: str) -> NodeType:
        """Convert string to NodeType enum."""
        mapping = {
            "agent_with_tools": cls.AGENT_WITH_TOOLS,
            "agent_only": cls.AGENT_ONLY,
            "hitl": cls.HITL,
            "form": cls.FORM,
            "code_execution": cls.CODE_EXECUTION,
            "loop": cls.LOOP,
            "parallel": cls.PARALLEL
        }
        return mapping.get(type_str, cls.AGENT_ONLY)


class EdgeType(Enum):
    """Edge types in workflow graph."""
    SEQUENTIAL = auto()      # Explicit sequential execution
    DATA_FLOW = auto()       # Data flow from placeholders (deprecated - use dependencies)
    DEPENDENCY = auto()      # Inferred from depends_on (deprecated - use dependencies)
    CONDITIONAL_TRUE = auto() # Conditional branch (if_true)
    CONDITIONAL_FALSE = auto() # Conditional branch (if_false)
    NESTED = auto()          # Implicit parent-child relationship (loop_body, parallel groups)


@dataclass
class IRNode:
    """Typed workflow node in IR."""
    id: str
    type: NodeType
    tool_id: Optional[str] = None
    app_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Node IDs this node depends on (e.g., ['node_1', 'node_2'])
    code: Optional[str] = None  # For code_execution type nodes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, IRNode):
            return self.id == other.id
        return False


@dataclass
class IREdge:
    """Typed workflow edge in IR."""
    source: str
    target: str
    type: EdgeType
    condition: Optional[str] = None  # For conditional edges
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source, self.target, self.type))
    
    def __eq__(self, other):
        if isinstance(other, IREdge):
            return (self.source == other.source and 
                   self.target == other.target and 
                   self.type == other.type)
        return False


class WorkflowIR:
    """
    Intermediate Representation for workflows.
    
    Provides structured, validated graph representation with:
    - Automatic edge inference from depends_on and ${from_step:...}
    - DAG validation (cycle detection, disconnected components)
    - Conflict detection (parallel vs sequential constraints)
    """
    
    def __init__(self, workflow_dict: Dict[str, Any]):
        """
        Initialize WorkflowIR from WORKFLOW dict.
        
        Args:
            workflow_dict: Validated WORKFLOW dict from Pydantic
        """
        self.raw_dict = workflow_dict
        self.nodes: Dict[str, IRNode] = {}
        self.edges: List[IREdge] = []
        self.dag: nx.DiGraph = nx.DiGraph()
        self.conditional_branches: Dict[str, Dict[str, Any]] = {}
        
        # Build IR
        self._parse_nodes()
        self._parse_conditional_edges()
        self._build_dag()
        
        # ✅ AUTO-FIX: Detect and fix fork-join patterns (Sim.ai-style)
        fixed_count = self.infer_missing_fork_join_dependencies()
        if fixed_count > 0:
            logger.info(f"[FORK-JOIN] ✅ Auto-fixed {fixed_count} dependencies using NetworkX")
        
        self._validate_dag()
    
    def _parse_nodes(self):
        """Parse nodes from workflow dict."""
        for node_dict in self.raw_dict.get("nodes", []):
            node = IRNode(
                id=node_dict["id"],
                type=NodeType.from_string(node_dict["type"]),
                tool_id=node_dict.get("tool_id"),
                app_name=node_dict.get("app_name"),
                params=node_dict.get("params", {}),
                description=node_dict.get("description"),
                code=node_dict.get("code"),  # For code_execution nodes
                metadata={k: v for k, v in node_dict.items() 
                         if k not in ["id", "type", "tool_id", "app_name", "params", 
                                     "description", "code"]}
            )
            self.nodes[node.id] = node
            logger.debug(f"Parsed node: {node.id} (type: {node.type.name})")
    
    def _parse_conditional_edges(self):
        """Parse conditional_edges from workflow dict."""
        for cond_edge in self.raw_dict.get("conditional_edges", []):
            source = cond_edge.get("source")
            if source:
                self.conditional_branches[source] = {
                    "condition": cond_edge.get("condition"),
                    "if_true": cond_edge.get("if_true"),
                    "if_false": cond_edge.get("if_false")
                }
                logger.debug(f"Parsed conditional edge from: {source}")
    
    def _build_dag(self):
        """
        Build DAG from explicit edges array (PRIMARY SOURCE).
        
        Builds edges from:
        1. 'edges' array in workflow dict (PRIMARY - required)
        2. Data flow references ${...} in params (SECONDARY - auto-inferred)
        3. Conditional branches (if_true, if_false)
        4. Legacy sequential_edges (backward compatibility)
        """
        # Add all nodes to DAG
        for node_id in self.nodes:
            self.dag.add_node(node_id)
        
        # 1. Add edges from 'edges' array (PRIMARY SOURCE - REQUIRED)
        for edge_dict in self.raw_dict.get("edges", []):
            # Support both dict and tuple formats
            if isinstance(edge_dict, dict):
                # Support both 'source'/'target' AND 'from'/'to' keys
                source = edge_dict.get('source') or edge_dict.get('from')
                target = edge_dict.get('target') or edge_dict.get('to')
                edge_type_str = edge_dict.get('type', 'sequential')
                condition = edge_dict.get('condition')
            elif isinstance(edge_dict, (tuple, list)) and len(edge_dict) >= 2:
                source, target = edge_dict[0], edge_dict[1]
                edge_type_str = 'sequential'
                condition = None
            else:
                logger.warning(f"Skipping malformed edge: {edge_dict}")
                continue
            
            if not source or not target:
                logger.warning(f"Skipping edge with missing source/target: {edge_dict}")
                continue
            
            if source in self.nodes and target in self.nodes:
                # Map edge type string to EdgeType enum
                edge_type_map = {
                    'sequential': EdgeType.SEQUENTIAL,
                    'conditional': EdgeType.CONDITIONAL_TRUE,
                    'data_flow': EdgeType.DATA_FLOW
                }
                edge_type = edge_type_map.get(edge_type_str, EdgeType.SEQUENTIAL)
                
                edge = IREdge(source=source, target=target, type=edge_type, condition=condition)
                self.edges.append(edge)
                self.dag.add_edge(source, target, edge_type=edge_type)
                logger.debug(f"Added {edge_type.name} edge from edges array: {source} → {target}")
            else:
                if source not in self.nodes:
                    logger.warning(f"Edge references non-existent source node: {source}")
                if target not in self.nodes:
                    logger.warning(f"Edge references non-existent target node: {target}")
        
        # 2. Add conditional edges (if specified)
        for source, branches in self.conditional_branches.items():
            if source not in self.nodes:
                logger.warning(f"Conditional edge source doesn't exist: {source}")
                continue
            
            # Handle if_true branch
            if_true = branches.get("if_true")
            if if_true:
                targets = [if_true] if isinstance(if_true, str) else if_true
                for target in targets:
                    if target != "END" and target in self.nodes:
                        edge = IREdge(source=source, target=target, 
                                    type=EdgeType.CONDITIONAL_TRUE,
                                    condition=branches.get("condition"))
                        self.edges.append(edge)
                        self.dag.add_edge(source, target, edge_type=EdgeType.CONDITIONAL_TRUE)
                        logger.debug(f"Added CONDITIONAL_TRUE edge: {source} → {target}")
            
            # Handle if_false branch
            if_false = branches.get("if_false")
            if if_false:
                targets = [if_false] if isinstance(if_false, str) else if_false
                for target in targets:
                    if target != "END" and target in self.nodes:
                        edge = IREdge(source=source, target=target, 
                                    type=EdgeType.CONDITIONAL_FALSE,
                                    condition=branches.get("condition"))
                        self.edges.append(edge)
                        self.dag.add_edge(source, target, edge_type=EdgeType.CONDITIONAL_FALSE)
                        logger.debug(f"Added CONDITIONAL_FALSE edge: {source} → {target}")
        
        # 4. Add explicit sequential_edges (legacy backward compatibility)
        for seq_edge in self.raw_dict.get("sequential_edges", []):
            if isinstance(seq_edge, (tuple, list)) and len(seq_edge) >= 2:
                source, target = seq_edge[0], seq_edge[1]
                if source in self.nodes and target in self.nodes:
                    # Check if edge doesn't already exist
                    if not self.dag.has_edge(source, target):
                        edge = IREdge(source=source, target=target, type=EdgeType.SEQUENTIAL)
                        self.edges.append(edge)
                        self.dag.add_edge(source, target, edge_type=EdgeType.SEQUENTIAL)
                        logger.debug(f"Added SEQUENTIAL edge (legacy): {source} → {target}")
        
        # 5. Add implicit edges for nested nodes (loop_body, parallel groups)
        # This ensures nested nodes don't appear as "disconnected" in validation
        for node_id, node_dict in [(n["id"], n) for n in self.raw_dict.get("nodes", [])]:
            if node_id not in self.nodes:
                continue
            
            # Handle loop_body (nested nodes in loops)
            loop_body = node_dict.get("loop_body", [])
            if loop_body and isinstance(loop_body, list):
                for nested_id in loop_body:
                    if isinstance(nested_id, str) and nested_id in self.nodes:
                        # Create implicit NESTED edge: loop → nested_node
                        if not self.dag.has_edge(node_id, nested_id):
                            edge = IREdge(source=node_id, target=nested_id, type=EdgeType.NESTED)
                            self.edges.append(edge)
                            self.dag.add_edge(node_id, nested_id, edge_type=EdgeType.NESTED)
                            logger.debug(f"Added NESTED edge (loop body): {node_id} → {nested_id}")
            
            # Handle nodes array (alternative to loop_body)
            nested_nodes = node_dict.get("nodes", [])
            if nested_nodes and isinstance(nested_nodes, list):
                for nested_node in nested_nodes:
                    nested_id = nested_node.get("id") if isinstance(nested_node, dict) else nested_node
                    if nested_id and nested_id in self.nodes:
                        if not self.dag.has_edge(node_id, nested_id):
                            edge = IREdge(source=node_id, target=nested_id, type=EdgeType.NESTED)
                            self.edges.append(edge)
                            self.dag.add_edge(node_id, nested_id, edge_type=EdgeType.NESTED)
                            logger.debug(f"Added NESTED edge (nodes array): {node_id} → {nested_id}")
        
        logger.info(f"[WorkflowIR] Built DAG: {len(self.edges)} edges from explicit edges array")
    
    def _validate_dag(self):
        """
        Validate DAG structure.
        
        Checks:
        - No cycles (must be acyclic)
        - No disconnected components (all nodes reachable, excluding nested nodes)
        - Conditional nodes have both branches
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.dag):
            cycles = list(nx.simple_cycles(self.dag))
            raise ValueError(f"Workflow contains cycles: {cycles}")
        
        # Check for disconnected components (excluding nested nodes)
        if len(self.nodes) > 1:  # Only check if multiple nodes
            # Identify nested nodes (children of loop/parallel nodes)
            nested_node_ids = set()
            
            # DEBUG: Log raw_dict structure (using INFO so it's visible)
            logger.info(f"[NESTED_DEBUG] raw_dict has {len(self.raw_dict.get('nodes', []))} nodes")
            
            for node_dict in self.raw_dict.get("nodes", []):
                node_id = node_dict.get("id", "unknown")
                node_type = node_dict.get("type", "unknown")
                
                # DEBUG: Show ALL keys in this node
                logger.info(f"[NESTED_DEBUG] Node '{node_id}' (type={node_type}) has keys: {list(node_dict.keys())}")
                
                # Check loop_body in multiple locations
                loop_body = node_dict.get("loop_body", [])
                if not loop_body:
                    # Check if loop_body is in metadata (handle None case)
                    metadata = node_dict.get("metadata") or {}
                    if isinstance(metadata, dict) and metadata:
                        logger.info(f"[NESTED_DEBUG] Node '{node_id}' metadata keys: {list(metadata.keys())}")
                        loop_body = metadata.get("loop_body", [])
                        if loop_body:
                            logger.info(f"[NESTED_DEBUG] Found loop_body in metadata: {loop_body}")
                
                if loop_body:
                    logger.info(f"[NESTED_DEBUG] Node '{node_id}' (type={node_type}) has loop_body: {loop_body}")
                    nested_node_ids.update(
                        item for item in loop_body 
                        if isinstance(item, str)
                    )
                else:
                    logger.info(f"[NESTED_DEBUG] Node '{node_id}' has NO loop_body anywhere")
                # Check nodes array
                nested_nodes = node_dict.get("nodes", [])
                if nested_nodes:
                    logger.info(f"[NESTED_DEBUG] Node '{node_id}' (type={node_type}) has nodes array: {nested_nodes}")
                    nested_node_ids.update(
                        node.get("id") if isinstance(node, dict) else node 
                        for node in nested_nodes
                        if (isinstance(node, dict) and node.get("id")) or isinstance(node, str)
                    )
            
            logger.info(f"[NESTED_DEBUG] Found {len(nested_node_ids)} nested nodes: {nested_node_ids}")
            
            # Check connectivity only for non-nested nodes
            undirected = self.dag.to_undirected()
            components = list(nx.connected_components(undirected))
            
            if len(components) > 1:
                # Filter out components that only contain nested nodes
                non_nested_components = [
                    comp for comp in components 
                    if not all(node_id in nested_node_ids for node_id in comp)
                ]
                
                if len(non_nested_components) > 1:
                    logger.warning(
                        f"Workflow has {len(non_nested_components)} disconnected components "
                        f"(excluding {len(nested_node_ids)} nested nodes): {non_nested_components}"
                    )
                else:
                    logger.debug(
                        f"All components are connected when nested nodes are considered "
                        f"({len(nested_node_ids)} nested nodes in loop/parallel structures)"
                    )
        
        # Check conditional nodes have both branches
        for source, branches in self.conditional_branches.items():
            if branches.get("if_true") is None or branches.get("if_false") is None:
                logger.warning(f"Conditional node '{source}' missing branch: if_true={branches.get('if_true')}, if_false={branches.get('if_false')}")
        
        logger.info(f"✅ DAG validation passed: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get topological execution order with parallelism.
        
        Returns:
            List of layers, where each layer can execute in parallel
        """
        try:
            # Get topological generations (layers)
            return list(nx.topological_generations(self.dag))
        except nx.NetworkXError as e:
            logger.error(f"Failed to compute execution order: {e}")
            return [[node_id] for node_id in self.nodes.keys()]
    
    def get_node(self, node_id: str) -> Optional[IRNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all upstream dependencies for a node."""
        if node_id not in self.dag:
            return []
        return list(self.dag.predecessors(node_id))
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get all downstream dependents for a node."""
        if node_id not in self.dag:
            return []
        return list(self.dag.successors(node_id))
    
    def infer_missing_fork_join_dependencies(self) -> int:
        """
        Auto-detect and fix fork-join patterns using NetworkX (Sim.ai-style).
        
        Algorithm:
        1. Get parallel execution layers using nx.topological_generations()
        2. For each layer with 2+ parallel nodes (fork), check next layer
        3. If next layer node only depends on fork's INPUT, add ALL fork outputs
        
        Returns:
            Number of dependencies added
        
        Example:
            Before: [1] → [2, 3, 4] → [5]
            Dependencies: 5 depends on [1] ❌ (missing 2, 3, 4)
            After: 5 depends on [1, 2, 3, 4] ✅
        """
        added_count = 0
        
        # Get execution layers (parallel groups)
        layers = list(nx.topological_generations(self.dag))
        
        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            
            # Check if current layer has parallel execution (fork)
            if len(current_layer) <= 1:
                continue  # No fork, skip
            
            logger.info(f"[FORK-JOIN] Detected parallel layer: {current_layer}")
            
            # Find the common input to the parallel layer (fork source)
            fork_inputs = set()
            for node_id in current_layer:
                predecessors = list(self.dag.predecessors(node_id))
                fork_inputs.update(predecessors)
            
            # For each node in next layer, check if it's a join point
            for next_node_id in next_layer:
                current_deps = set(self.dag.predecessors(next_node_id))
                
                # Check if this node only depends on fork inputs (not fork outputs)
                depends_on_fork_input = bool(fork_inputs & current_deps)
                depends_on_fork_outputs = bool(set(current_layer) & current_deps)
                
                if depends_on_fork_input and not depends_on_fork_outputs:
                    # This is a join point missing fork output dependencies!
                    logger.warning(
                        f"[FORK-JOIN] Node {next_node_id} depends on fork input {fork_inputs} "
                        f"but missing fork outputs {current_layer}"
                    )
                    
                    # Add missing dependencies from ALL parallel nodes
                    for parallel_node_id in current_layer:
                        if not self.dag.has_edge(parallel_node_id, next_node_id):
                            self.dag.add_edge(parallel_node_id, next_node_id, edge_type=EdgeType.DATA_FLOW)
                            edge = IREdge(
                                source=parallel_node_id,
                                target=next_node_id,
                                type=EdgeType.DATA_FLOW
                            )
                            self.edges.append(edge)
                            added_count += 1
                            logger.info(
                                f"[FORK-JOIN] ✅ Auto-added edge: {parallel_node_id} → {next_node_id}"
                            )
        
        if added_count > 0:
            logger.info(f"[FORK-JOIN] Auto-fixed {added_count} missing fork-join dependencies")
        
        return added_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert IR back to dict representation."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type.name.lower(),
                    "tool_id": node.tool_id,
                    "app_name": node.app_name,
                    "params": node.params,
                    "description": node.description,
                    **node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type.name.lower(),
                    "condition": edge.condition
                }
                for edge in self.edges
            ],
            "execution_order": self.get_execution_order()
        }
    
    def visualize(self) -> str:
        """Generate ASCII art visualization of the workflow DAG."""
        lines = ["Workflow DAG:", "=" * 50]
        
        for i, layer in enumerate(self.get_execution_order()):
            lines.append(f"Layer {i}: {', '.join(layer)}")
            for node_id in layer:
                node = self.nodes[node_id]
                lines.append(f"  └─ {node_id} ({node.type.name})")
                deps = self.get_dependencies(node_id)
                if deps:
                    lines.append(f"     Dependencies: {', '.join(deps)}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def create_workflow_ir(workflow_dict: Dict[str, Any]) -> WorkflowIR:
    """
    Factory function to create WorkflowIR from dict.
    
    Args:
        workflow_dict: Validated WORKFLOW dict
    
    Returns:
        WorkflowIR instance
    
    Raises:
        ValueError: If workflow is invalid
    """
    try:
        ir = WorkflowIR(workflow_dict)
        logger.info(f"✅ Created WorkflowIR: {len(ir.nodes)} nodes, {len(ir.edges)} edges")
        return ir
    except Exception as e:
        logger.error(f"❌ Failed to create WorkflowIR: {e}")
        raise
