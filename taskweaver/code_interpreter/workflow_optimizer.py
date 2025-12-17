"""
Schema-based Workflow Dependency Optimizer

Analyzes tool schemas to identify true data dependencies and optimizes
workflow structure for maximum parallelization.

Inspired by Autogen's separation of concerns - dedicated service for
post-generation optimization.
"""
import logging
import asyncio
from typing import Dict, List, Tuple, Set, Any, Optional

logger = logging.getLogger(__name__)


class WorkflowOptimizer:
    """
    Optimizes workflow dependencies based on schema analysis.
    
    Takes a generated workflow and tool schemas, analyzes true data dependencies,
    and rewrites node dependencies to enable parallel execution where possible.
    
    Architecture:
        1. Analyze tool schemas to find true dependencies
        2. Identify nodes that can run in parallel
        3. Rewrite node.dependencies array
        4. Rebuild edges for sequential vs parallel execution
        5. Return optimized workflow + metadata
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def optimize_dependencies(
        self,
        workflow_dict: Dict[str, Any],
        tool_schemas: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize workflow dependencies based on schema analysis.
        
        Args:
            workflow_dict: Generated workflow with nodes and edges
            tool_schemas: List of tool schema dicts from _BATCH_CACHE
            session_id: Optional session ID for logging
            
        Returns:
            Tuple of (optimized_workflow, optimization_metadata)
        """
        self.logger.info(f"[OPTIMIZER] Starting optimization for {len(workflow_dict.get('nodes', []))} nodes")
        
        # Initialize metadata
        metadata = {
            'original_edges': len(workflow_dict.get('edges', [])),
            'optimized_edges': 0,
            'parallel_groups': 0,
            'parallelizable_nodes': [],
            'issues_fixed': [],
            'analysis_duration_ms': 0
        }
        
        try:
            # Phase 1: Run schema-based dependency analysis
            analysis_result = await self._run_dependency_analysis(tool_schemas, session_id)
            
            if not analysis_result:
                self.logger.warning("[OPTIMIZER] Analysis failed, returning original workflow")
                return workflow_dict, metadata
            
            # Phase 2: Build node-to-tool mapping
            node_tool_map = self._build_node_tool_mapping(workflow_dict, tool_schemas)
            
            # Phase 3: Identify structural constraints (form, hitl, etc.)
            constraints = self._identify_constraints(workflow_dict)
            
            # Phase 4: Rewrite dependencies based on schema analysis
            optimized_workflow = self._rewrite_dependencies(
                workflow_dict,
                analysis_result,
                node_tool_map,
                constraints,
                metadata
            )
            
            # Phase 5: Rebuild edges from optimized dependencies
            optimized_workflow = self._rebuild_edges(optimized_workflow, metadata)
            
            self.logger.info(
                f"[OPTIMIZER] ✅ Optimization complete: "
                f"{metadata['parallel_groups']} parallel groups, "
                f"{len(metadata['parallelizable_nodes'])} parallelizable nodes"
            )
            
            return optimized_workflow, metadata
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] ❌ Optimization failed: {e}", exc_info=True)
            # Return original workflow on failure (fail-safe)
            return workflow_dict, metadata
    
    async def _run_dependency_analysis(
        self,
        tool_schemas: List[Dict[str, Any]],
        session_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Run AIDependencyAnalyzer on tool schemas (async)."""
        try:
            from apps.py_workflows.generation.analysis.dependency_analyzer import AIDependencyAnalyzer
            
            analyzer = AIDependencyAnalyzer()
            
            # Run async analysis (await directly since we're in async context)
            analysis_result = await analyzer.analyze_tools(tool_schemas)
            
            self.logger.info(
                f"[OPTIMIZER] Analysis complete: "
                f"{analysis_result.total_dependencies} dependencies, "
                f"{len(analysis_result.parallel_groups)} parallel groups"
            )
            
            return {
                'dependencies': analysis_result.dependency_graph.dependencies,
                'parallel_groups': analysis_result.parallel_groups,
                'execution_order': analysis_result.optimal_order,
                'has_cycles': analysis_result.has_cycles,
                'is_valid': analysis_result.is_valid
            }
            
        except Exception as e:
            self.logger.error(f"[OPTIMIZER] Analysis failed: {e}", exc_info=True)
            return None
    
    def _build_node_tool_mapping(
        self,
        workflow_dict: Dict[str, Any],
        tool_schemas: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Build mapping of node_id → tool action_id.
        
        Returns:
            Dict like {'node_2': 'FIRECRAWL_EXTRACT', 'node_5': 'COMPOSIO_SEARCH_WEB'}
        """
        node_tool_map = {}
        
        for node in workflow_dict.get('nodes', []):
            node_id = node.get('id')
            tool_id = node.get('tool_id')
            node_type = node.get('type')
            
            if tool_id:
                node_tool_map[node_id] = tool_id
            elif node_type in ['agent_only', 'form', 'hitl']:
                # These don't have tool_ids but are important for structure
                node_tool_map[node_id] = f"__{node_type}__"
        
        self.logger.info(f"[OPTIMIZER] Built mapping for {len(node_tool_map)} nodes")
        return node_tool_map
    
    def _identify_constraints(self, workflow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify structural constraints that must be preserved.
        
        Returns:
            Dict with constraint info:
            - entry_nodes: List of nodes that must come first (forms)
            - exit_nodes: List of nodes that must come last
            - sequential_nodes: Nodes that must remain sequential
        """
        constraints = {
            'entry_nodes': [],
            'exit_nodes': [],
            'sequential_nodes': [],
            'hitl_nodes': []
        }
        
        for node in workflow_dict.get('nodes', []):
            node_id = node.get('id')
            node_type = node.get('type')
            
            # Form nodes typically are entry points
            if node_type == 'form':
                constraints['entry_nodes'].append(node_id)
            
            # HITL nodes need special handling - they wait for human input
            if node_type == 'hitl':
                constraints['hitl_nodes'].append(node_id)
        
        self.logger.info(
            f"[OPTIMIZER] Constraints: "
            f"{len(constraints['entry_nodes'])} entry, "
            f"{len(constraints['hitl_nodes'])} HITL"
        )
        
        return constraints
    
    def _rewrite_dependencies(
        self,
        workflow_dict: Dict[str, Any],
        analysis_result: Dict[str, Any],
        node_tool_map: Dict[str, str],
        constraints: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rewrite node dependencies based on schema analysis.
        
        Strategy:
        1. For nodes with tools: Use schema-based dependencies
        2. For entry nodes (forms): Keep dependencies=[]
        3. For agent_only nodes: Analyze if they can parallelize
        4. For HITL nodes: Preserve their dependencies
        """
        optimized_workflow = workflow_dict.copy()
        nodes = optimized_workflow.get('nodes', [])
        
        # Build reverse mapping: tool_id → node_id
        tool_node_map = {v: k for k, v in node_tool_map.items() if not v.startswith('__')}
        
        # Get analyzed dependencies (tool_id → tool_id)
        analyzed_deps = analysis_result.get('dependencies', [])
        parallel_groups = analysis_result.get('parallel_groups', [])
        
        # Track which nodes can be parallelized
        parallelizable = set()
        for group in parallel_groups:
            parallelizable.update(group)
        
        for node in nodes:
            node_id = node.get('id')
            node_type = node.get('type')
            tool_id = node.get('tool_id')
            original_deps = node.get('dependencies', [])
            
            # Entry nodes (forms) keep empty dependencies
            if node_id in constraints['entry_nodes']:
                node['dependencies'] = []
                self.logger.info(f"[OPTIMIZER] {node_id}: Kept as entry node (form)")
                continue
            
            # If node has a tool, use schema-based dependencies
            if tool_id and tool_id in parallelizable:
                # This tool can run in parallel - find its true dependencies
                true_deps = []
                
                for dep in analyzed_deps:
                    if dep.get('to_action') == tool_id:
                        from_action = dep.get('from_action')
                        if from_action in tool_node_map:
                            true_deps.append(tool_node_map[from_action])
                
                # If no schema dependencies, depend on entry nodes
                if not true_deps:
                    true_deps = constraints['entry_nodes']
                
                if set(true_deps) != set(original_deps):
                    self.logger.info(
                        f"[OPTIMIZER] {node_id}: "
                        f"Rewriting deps from {original_deps} → {true_deps}"
                    )
                    node['dependencies'] = true_deps
                    metadata['issues_fixed'].append({
                        'node_id': node_id,
                        'original': original_deps,
                        'optimized': true_deps,
                        'reason': 'schema_analysis'
                    })
                    metadata['parallelizable_nodes'].append(node_id)
                else:
                    node['dependencies'] = original_deps
            
            # agent_only nodes: Check if they reference previous node outputs
            elif node_type == 'agent_only':
                # For now, preserve their dependencies (they might need context)
                # Future: Analyze their descriptions/code to determine true deps
                node['dependencies'] = original_deps
                self.logger.info(f"[OPTIMIZER] {node_id}: Preserved agent_only dependencies")
            
            # HITL nodes: Preserve dependencies (they need human approval)
            elif node_type == 'hitl':
                node['dependencies'] = original_deps
                self.logger.info(f"[OPTIMIZER] {node_id}: Preserved HITL dependencies")
            
            else:
                # Other node types: Keep original
                node['dependencies'] = original_deps
        
        # Count parallel groups in optimized workflow
        metadata['parallel_groups'] = self._count_parallel_groups(nodes)
        
        return optimized_workflow
    
    def _count_parallel_groups(self, nodes: List[Dict[str, Any]]) -> int:
        """
        Count how many nodes can run in parallel (share same dependencies).
        """
        # Group nodes by their dependencies signature
        dep_groups = {}
        for node in nodes:
            deps_key = tuple(sorted(node.get('dependencies', [])))
            if deps_key not in dep_groups:
                dep_groups[deps_key] = []
            dep_groups[deps_key].append(node['id'])
        
        # Count groups with 2+ nodes (those are parallel groups)
        parallel_groups = sum(1 for group in dep_groups.values() if len(group) >= 2)
        
        return parallel_groups
    
    def _rebuild_edges(
        self,
        workflow_dict: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rebuild edges array from node dependencies.
        
        Detects parallel vs sequential edges based on shared dependencies.
        """
        nodes = workflow_dict.get('nodes', [])
        edges = []
        sequential_edges = []
        parallel_edges = []
        
        # Group nodes by their dependencies to find parallel patterns
        dep_groups = {}
        for node in nodes:
            node_id = node.get('id')
            deps = tuple(sorted(node.get('dependencies', [])))
            
            if deps not in dep_groups:
                dep_groups[deps] = []
            dep_groups[deps].append(node_id)
        
        # Build edges from dependencies
        for node in nodes:
            node_id = node.get('id')
            deps = node.get('dependencies', [])
            
            for dep_id in deps:
                # ✅ Create edges with BOTH formats for compatibility
                edge = {
                    'from': dep_id,
                    'to': node_id,
                    'source': dep_id,   # ← Frontend expects this!
                    'target': node_id,  # ← Frontend expects this!
                    'type': 'sequential',  # Default to sequential
                    'from_step': dep_id,  # ← Backward compatibility
                    'to_step': node_id,   # ← Backward compatibility
                    'condition': None,
                    'metadata': {}
                }
                edges.append(edge)
                sequential_edges.append((dep_id, node_id))
        
        # Detect parallel edges: nodes with same dependencies
        for deps, node_ids in dep_groups.items():
            if len(node_ids) >= 2 and deps:  # 2+ nodes with same non-empty deps
                # These nodes can run in parallel after their shared dependency
                for dep_id in deps:
                    for node_id in node_ids:
                        # Mark as parallel edge in metadata (but keep sequential in edges)
                        # Frontend will handle parallel visualization
                        parallel_edges.append((dep_id, node_ids))
                break  # Only record once per group
        
        workflow_dict['edges'] = edges
        workflow_dict['sequential_edges'] = sequential_edges
        workflow_dict['parallel_edges'] = parallel_edges if parallel_edges else []
        
        metadata['optimized_edges'] = len(edges)
        
        self.logger.info(
            f"[OPTIMIZER] Rebuilt {len(edges)} edges "
            f"({len(sequential_edges)} sequential, {len(parallel_edges)} parallel groups)"
        )
        
        return workflow_dict
