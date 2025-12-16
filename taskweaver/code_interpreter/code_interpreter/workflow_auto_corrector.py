"""
Workflow Auto-Corrector - Phase 3 Corrections
==============================================

Deterministic post-processing to fix common LLM mistakes in workflow generation.

This module applies rule-based corrections to workflows after LLM generation:
1. Forms: Remove auth fields, remove fields for values in prompt
2. HITL: Remove trivial approvals, ensure decision fields
3. agent_only: Ensure mode matches code presence, warn about combined logic

Cross-reference: code_generator.py calls this after validation
Cross-reference: workflow_schema_builder.py defines node schemas
"""

import re
import logging
from typing import Dict, List, Set, Any


class WorkflowAutoCorrector:
    """
    Phase 3: Deterministic auto-correction for workflow IR.
    
    Fixes common issues:
    - Unnecessary form fields (auth, already-specified values)
    - Missing HITL decision fields
    - agent_only mode mismatches
    - Broken edges after node removal
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Auth/credential keywords to detect and remove from forms
        self.auth_keywords = [
            'user_id', 'token', 'api_key', 'credentials', 'password', 'auth',
            'access_token', 'refresh_token', 'client_id', 'client_secret',
            'oauth', 'permission', 'authorization'
        ]
        
        # Read-only tool actions (HITL approval not needed)
        self.read_only_actions = [
            'FETCH', 'GET', 'LIST', 'SEARCH', 'READ', 'FIND',
            'RETRIEVE', 'VIEW', 'SHOW', 'QUERY'
        ]
    
    def auto_correct(
        self,
        workflow_json: Dict[str, Any],
        user_prompt: str,
        init_plan: str = ""
    ) -> Dict[str, Any]:
        """
        Apply all auto-corrections to workflow.
        
        Args:
            workflow_json: Workflow dict from LLM
            user_prompt: Original user query
            init_plan: Planner's init_plan with markers
            
        Returns:
            Corrected workflow dict
        """
        nodes = workflow_json.get('nodes', [])
        edges = workflow_json.get('edges', [])
        
        self.logger.info("[AUTO_CORRECT] Phase 3: Starting auto-correction...")
        self.logger.info(f"[AUTO_CORRECT] Input: {len(nodes)} nodes, {len(edges)} edges")
        
        corrected_nodes = []
        removed_node_ids: Set[str] = set()
        
        for node in nodes:
            node_type = node.get('type')
            node_id = node.get('id')
            
            # Apply type-specific corrections
            if node_type == 'form':
                node, should_remove = self._correct_form_node(node, user_prompt)
                if should_remove:
                    removed_node_ids.add(node_id)
                    continue
            
            elif node_type == 'hitl':
                node, should_remove = self._correct_hitl_node(node, nodes, init_plan)
                if should_remove:
                    removed_node_ids.add(node_id)
                    continue
            
            elif node_type == 'agent_only':
                node = self._correct_agent_only_node(node)
            
            corrected_nodes.append(node)
        
        # Update edges (remove edges to/from removed nodes)
        if removed_node_ids:
            corrected_edges = self._correct_edges(edges, removed_node_ids)
            workflow_json['edges'] = corrected_edges
        
        workflow_json['nodes'] = corrected_nodes
        
        # Summary
        self.logger.info(
            f"[AUTO_CORRECT] Complete: {len(nodes)} → {len(corrected_nodes)} nodes, "
            f"{len(removed_node_ids)} removed"
        )
        
        return workflow_json
    
    def _correct_form_node(
        self,
        node: Dict[str, Any],
        user_prompt: str
    ) -> tuple[Dict[str, Any], bool]:
        """
        Correct form node issues.
        
        Returns:
            (corrected_node, should_remove)
        """
        node_id = node.get('id')
        fields = node.get('fields', [])
        necessary_fields = []
        
        prompt_lower = user_prompt.lower()
        
        for field in fields:
            field_name = field.get('name', '').lower()
            field_label = field.get('label', '').lower()
            
            # Rule 1: Remove auth-related fields
            if any(auth_kw in field_name for auth_kw in self.auth_keywords):
                self.logger.info(
                    f"[AUTO_CORRECT] Form {node_id}: Removing auth field '{field['name']}'"
                )
                continue
            
            # Rule 2: Remove if value explicitly in prompt
            if self._is_field_value_in_prompt(field_name, field_label, prompt_lower):
                self.logger.info(
                    f"[AUTO_CORRECT] Form {node_id}: Removing field '{field['name']}' "
                    f"(value in prompt)"
                )
                continue
            
            # Keep this field
            necessary_fields.append(field)
        
        # Decide if entire form should be removed
        should_remove = len(necessary_fields) == 0
        
        if should_remove:
            self.logger.info(f"[AUTO_CORRECT] Form {node_id}: No necessary fields - removing form")
        else:
            node['fields'] = necessary_fields
            removed_count = len(fields) - len(necessary_fields)
            if removed_count > 0:
                self.logger.info(
                    f"[AUTO_CORRECT] Form {node_id}: Kept {len(necessary_fields)}/{len(fields)} fields"
                )
        
        return node, should_remove
    
    def _is_field_value_in_prompt(
        self,
        field_name: str,
        field_label: str,
        prompt_lower: str
    ) -> bool:
        """Check if field value is already specified in user prompt."""
        
        # Check for count/limit fields
        if any(kw in field_name or kw in field_label for kw in ['limit', 'count', 'number', 'max']):
            # Check if user specified a number
            count_match = re.search(r'\b(last|first|recent|top)\s+(\d+)\b', prompt_lower)
            if count_match:
                return True
        
        # Check for channel/recipient when user says "my"
        if any(kw in field_name for kw in ['channel', 'recipient', 'to', 'target']):
            if 'my messages' in prompt_lower or 'my account' in prompt_lower or 'my inbox' in prompt_lower:
                return True
        
        # Check for date/time when user specifies relative time
        if any(kw in field_name for kw in ['date', 'time', 'when', 'start', 'end']):
            if any(rel in prompt_lower for rel in ['today', 'yesterday', 'tomorrow', 'this week', 'last week']):
                return True
        
        return False
    
    def _correct_hitl_node(
        self,
        node: Dict[str, Any],
        all_nodes: List[Dict[str, Any]],
        init_plan: str
    ) -> tuple[Dict[str, Any], bool]:
        """
        Correct HITL node issues.
        
        Returns:
            (corrected_node, should_remove)
        """
        node_id = node.get('id')
        fields = node.get('fields', [])
        
        # Rule 1: Ensure decision field exists
        has_decision = any(
            f.get('name') in ['decision', 'approval', 'approved', 'action']
            for f in fields
        )
        
        if not has_decision:
            self.logger.warning(
                f"[AUTO_CORRECT] HITL {node_id}: Missing decision field - adding it"
            )
            fields.insert(0, {
                'name': 'decision',
                'type': 'select',
                'label': 'Decision',
                'options': ['Approve', 'Reject'],
                'required': True
            })
            node['fields'] = fields
        
        # Rule 2: Check if HITL is trivial (approving read-only operation)
        should_remove = False
        depends_on = node.get('depends_on', [])
        
        if len(depends_on) == 1:
            # Find predecessor node
            prev_node = next((n for n in all_nodes if n['id'] == depends_on[0]), None)
            
            if prev_node and prev_node.get('type') == 'agent_with_tools':
                tool_id = prev_node.get('tool_id', '').upper()
                
                # Check if it's a read-only operation
                is_read_only = any(action in tool_id for action in self.read_only_actions)
                
                # Check if user explicitly requested approval
                explicit_approval = any(
                    kw in init_plan.lower()
                    for kw in ['approv', 'review', 'authorize', 'confirm']
                )
                
                if is_read_only and not explicit_approval:
                    self.logger.info(
                        f"[AUTO_CORRECT] HITL {node_id}: Removing trivial approval "
                        f"for read-only operation ({tool_id})"
                    )
                    should_remove = True
        
        return node, should_remove
    
    def _correct_agent_only_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correct agent_only node issues.
        
        Returns:
            Corrected node
        """
        node_id = node.get('id')
        agent_mode = node.get('agent_mode')
        has_code = bool(node.get('code', '').strip())
        
        # Rule 1: If mode='reasoning' but has code, remove code
        if agent_mode == 'reasoning' and has_code:
            self.logger.warning(
                f"[AUTO_CORRECT] agent_only {node_id}: mode='reasoning' but has code - removing code"
            )
            del node['code']
            has_code = False
        
        # Rule 2: If mode='code' but no code, change to reasoning
        elif agent_mode == 'code' and not has_code:
            self.logger.warning(
                f"[AUTO_CORRECT] agent_only {node_id}: mode='code' but no code - changing to reasoning"
            )
            node['agent_mode'] = 'reasoning'
        
        # Rule 3: If no agent_mode, infer from code presence
        elif not agent_mode:
            inferred_mode = 'code' if has_code else 'reasoning'
            self.logger.info(
                f"[AUTO_CORRECT] agent_only {node_id}: no agent_mode - inferring '{inferred_mode}'"
            )
            node['agent_mode'] = inferred_mode
        
        # Rule 4: Warn if code contains subjective decision logic
        if has_code and node.get('agent_mode') == 'code':
            code = node.get('code', '')
            
            # Check for subjective decision keywords
            decision_patterns = [
                r'if.*\b(wordy|concise|better|worse|best|worst)\b',
                r'\b(recommend|suggest|opinion|prefer|should)\b',
                r'if.*\b(good|bad|appropriate|suitable)\b'
            ]
            
            has_subjective = any(re.search(pattern, code.lower()) for pattern in decision_patterns)
            
            if has_subjective:
                self.logger.warning(
                    f"[AUTO_CORRECT] agent_only {node_id}: Code contains subjective logic "
                    f"(keywords: wordy/better/recommend) - should split into code + reasoning nodes. "
                    f"Code should only calculate/transform, reasoning should judge/recommend."
                )
                # Note: Can't automatically split without re-generation
                # This warning helps identify the issue for future improvements
        
        return node
    
    def _correct_edges(
        self,
        edges: List[Dict[str, Any]],
        removed_node_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Update edges after node removal.
        
        Strategy: Bridge gaps by connecting predecessor to successor
        """
        corrected_edges = []
        bridge_edges = []
        
        for edge in edges:
            from_node = edge.get('from')
            to_node = edge.get('to')
            
            # Skip edges involving removed nodes
            if from_node in removed_node_ids or to_node in removed_node_ids:
                self.logger.info(
                    f"[AUTO_CORRECT] Removing edge: {from_node} → {to_node}"
                )
                
                # Try to bridge the gap
                if from_node in removed_node_ids and to_node not in removed_node_ids:
                    # Find predecessors of the removed node
                    predecessors = [e['from'] for e in edges if e['to'] == from_node]
                    for pred in predecessors:
                        if pred not in removed_node_ids:
                            bridge_edge = {
                                'type': edge.get('type', 'sequential'),
                                'from': pred,
                                'to': to_node
                            }
                            bridge_edges.append(bridge_edge)
                            self.logger.info(
                                f"[AUTO_CORRECT] Bridged edge: {pred} → {to_node}"
                            )
                
                continue
            
            corrected_edges.append(edge)
        
        # Add bridge edges
        corrected_edges.extend(bridge_edges)
        
        return corrected_edges


# Singleton instance
_corrector_instance = None


def get_auto_corrector(logger: logging.Logger = None) -> WorkflowAutoCorrector:
    """Get or create singleton auto-corrector instance."""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = WorkflowAutoCorrector(logger=logger)
    return _corrector_instance

