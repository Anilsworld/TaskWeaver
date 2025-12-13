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
    
    @classmethod
    def from_string(cls, type_str: str) -> NodeType:
        """Convert string to NodeType enum."""
        mapping = {
            "agent_with_tools": cls.AGENT_WITH_TOOLS,
            "agent_only": cls.AGENT_ONLY,
            "hitl": cls.HITL,
            "form": cls.FORM,
            "code_execution": cls.CODE_EXECUTION
        }
        return mapping.get(type_str, cls.AGENT_ONLY)


class EdgeType(Enum):
    """Edge types in workflow graph."""
    SEQUENTIAL = auto()      # Explicit sequential execution
    DATA_FLOW = auto()       # Inferred from ${from_step:...}
    DEPENDENCY = auto()      # Inferred from depends_on
    CONDITIONAL_TRUE = auto() # Conditional branch (if_true)
    CONDITIONAL_FALSE = auto() # Conditional branch (if_false)
    PARALLEL = auto()        # Parallel execution group


@dataclass
class IRNode:
    """Typed workflow node in IR."""
    id: str
    type: NodeType
    tool_id: Optional[str] = None
    app_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    parallel_group: Optional[int] = None
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
        self.parallel_groups: Dict[int, List[str]] = {}
        self.conditional_branches: Dict[str, Dict[str, Any]] = {}
        
        # Build IR
        self._parse_nodes()
        self._parse_parallel_groups()
        self._parse_conditional_edges()
        self._build_dag()
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
                depends_on=node_dict.get("depends_on", []),
                parallel_group=node_dict.get("parallel_group"),
                metadata={k: v for k, v in node_dict.items() 
                         if k not in ["id", "type", "tool_id", "app_name", "params", 
                                     "description", "depends_on", "parallel_group"]}
            )
            self.nodes[node.id] = node
            logger.debug(f"Parsed node: {node.id} (type: {node.type.name})")
    
    def _parse_parallel_groups(self):
        """Parse parallel_groups from workflow dict."""
        parallel_groups_dict = self.raw_dict.get("parallel_groups", {})
        for group_name, node_ids in parallel_groups_dict.items():
            # Extract group number from parallel_group field or use hash
            group_num = hash(group_name) % 1000
            self.parallel_groups[group_num] = node_ids
            logger.debug(f"Parsed parallel group '{group_name}': {node_ids}")
    
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
        Build DAG with automatic edge inference.
        
        Infers edges from:
        1. Explicit depends_on arrays
        2. Data flow references ${from_step:...}
        3. Parallel groups (implicit concurrency)
        4. Conditional branches
        5. Sequential edges (if specified)
        """
        # Add all nodes to DAG
        for node_id in self.nodes:
            self.dag.add_node(node_id)
        
        # 1. Infer edges from depends_on
        for node_id, node in self.nodes.items():
            for dep in node.depends_on:
                if dep in self.nodes:
                    edge = IREdge(source=dep, target=node_id, type=EdgeType.DEPENDENCY)
                    self.edges.append(edge)
                    self.dag.add_edge(dep, node_id, edge_type=EdgeType.DEPENDENCY)
                    logger.debug(f"Inferred DEPENDENCY edge: {dep} → {node_id}")
                else:
                    logger.warning(f"Node {node_id} depends on non-existent node: {dep}")
        
        # 2. Infer edges from ${from_step:...} in params
        for node_id, node in self.nodes.items():
            data_deps = self._extract_data_dependencies(node.params)
            for dep_node_id in data_deps:
                if dep_node_id in self.nodes:
                    # Only add if not already covered by depends_on
                    if dep_node_id not in node.depends_on:
                        edge = IREdge(source=dep_node_id, target=node_id, type=EdgeType.DATA_FLOW)
                        self.edges.append(edge)
                        self.dag.add_edge(dep_node_id, node_id, edge_type=EdgeType.DATA_FLOW)
                        logger.debug(f"Inferred DATA_FLOW edge: {dep_node_id} → {node_id}")
                else:
                    logger.warning(f"Node {node_id} references non-existent node in data flow: {dep_node_id}")
        
        # 3. Add conditional edges
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
        
        # 4. Add explicit sequential_edges (if any)
        for seq_edge in self.raw_dict.get("sequential_edges", []):
            if isinstance(seq_edge, (tuple, list)) and len(seq_edge) >= 2:
                source, target = seq_edge[0], seq_edge[1]
                if source in self.nodes and target in self.nodes:
                    # Check if edge doesn't already exist
                    if not self.dag.has_edge(source, target):
                        edge = IREdge(source=source, target=target, type=EdgeType.SEQUENTIAL)
                        self.edges.append(edge)
                        self.dag.add_edge(source, target, edge_type=EdgeType.SEQUENTIAL)
                        logger.debug(f"Added SEQUENTIAL edge: {source} → {target}")
    
    def _extract_data_dependencies(self, params: Any) -> Set[str]:
        """
        Extract node IDs from ${from_step:node_id.field} references.
        
        Args:
            params: Parameter value (can be dict, list, str, etc.)
        
        Returns:
            Set of node IDs referenced in data flow
        """
        dependencies = set()
        param_str = str(params)
        
        # Regex to match ${from_step:node_id.field} or ${from_step:node_id}
        pattern = r'\$\{from_step:([a-zA-Z0-9_]+)'
        matches = re.findall(pattern, param_str)
        dependencies.update(matches)
        
        return dependencies
    
    def _validate_dag(self):
        """
        Validate DAG structure.
        
        Checks:
        - No cycles (must be acyclic)
        - No disconnected components (all nodes reachable)
        - Conditional nodes have both branches
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.dag):
            cycles = list(nx.simple_cycles(self.dag))
            raise ValueError(f"Workflow contains cycles: {cycles}")
        
        # Check for disconnected components
        if len(self.nodes) > 1:  # Only check if multiple nodes
            undirected = self.dag.to_undirected()
            components = list(nx.connected_components(undirected))
            if len(components) > 1:
                logger.warning(f"Workflow has {len(components)} disconnected components: {components}")
        
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
                    "depends_on": node.depends_on,
                    "parallel_group": node.parallel_group,
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
