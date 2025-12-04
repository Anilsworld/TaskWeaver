"""
Composio Schema Validator - FULLY DYNAMIC (No Hardcoding!)

This validates composio_action() calls against ACTUAL Composio schemas
fetched from the database. It auto-learns parameter mappings from
successful executions.

Key Principles:
1. NO HARDCODED ALIASES - All mappings are learned or fetched from schemas
2. DYNAMIC SCHEMA LOOKUP - Fetches actual schema from Composio DB
3. AUTO-LEARNING - Saves successful mappings for future use
4. USES TASKWEAVER EXPERIENCE - Follows TW's pattern for persistence
"""
import ast
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ComposioCallInfo:
    """Extracted info from a composio_action() call"""
    action_name: str
    params: Dict[str, Any]
    line_number: int
    is_variable_action: bool = False


@dataclass
class ValidationError:
    """A single validation error with auto-fix suggestion"""
    line: int
    action: str
    param_used: str  # What TaskWeaver used
    param_expected: Optional[str] = None  # What Composio expects
    message: str = ""
    is_missing_required: bool = False
    is_wrong_name: bool = False
    
    def format(self) -> str:
        if self.is_missing_required:
            return f"Line {self.line}: {self.action} requires '{self.param_used}'"
        elif self.is_wrong_name and self.param_expected:
            return f"Line {self.line}: {self.action} - use '{self.param_expected}' instead of '{self.param_used}'"
        return f"Line {self.line}: {self.action} - {self.message}"


@dataclass
class ValidationResult:
    """Result of validation with learning data"""
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    discovered_mappings: List[Tuple[str, str, str]] = field(default_factory=list)  # (action, tw_name, composio_name)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def format_correction_message(self) -> str:
        """Format for TaskWeaver's retry mechanism"""
        if self.is_valid:
            return ""
        
        lines = [
            "⚠️ PARAMETER VALIDATION ERRORS:",
            ""
        ]
        
        for err in self.errors:
            lines.append(f"  ❌ {err.format()}")
        
        lines.extend([
            "",
            "Please regenerate the code with the CORRECT parameter names.",
            "Use the exact parameter names from the Composio action schema.",
        ])
        
        return "\n".join(lines)


class DynamicAliasStore:
    """
    Stores learned parameter mappings using TaskWeaver's Experience pattern.
    
    Structure:
    {
        "GMAIL_SEND_EMAIL": {
            "to": "recipient_email",
            "msg": "body"
        },
        "GOOGLESHEETS_CREATE_GOOGLE_SHEET1": {
            "name": "title"
        }
    }
    """
    
    def __init__(self, experience_dir: Optional[str] = None):
        self.experience_dir = experience_dir
        self._mappings: Dict[str, Dict[str, str]] = {}  # {action_id: {tw_name: composio_name}}
        self._reverse: Dict[str, Dict[str, str]] = {}   # {action_id: {composio_name: tw_name}}
        self._load()
    
    def _get_file_path(self) -> Optional[str]:
        if not self.experience_dir:
            return None
        return os.path.join(self.experience_dir, "learned_param_mappings.json")
    
    def _load(self):
        """Load learned mappings from experience file"""
        file_path = self._get_file_path()
        if not file_path or not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self._mappings = data.get('mappings', {})
                # Build reverse index
                for action, mapping in self._mappings.items():
                    self._reverse[action] = {v: k for k, v in mapping.items()}
                logger.info(f"[ALIAS_STORE] Loaded mappings for {len(self._mappings)} actions")
        except Exception as e:
            logger.warning(f"[ALIAS_STORE] Could not load: {e}")
    
    def save(self):
        """Persist mappings to experience file"""
        file_path = self._get_file_path()
        if not file_path:
            return
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump({
                    'mappings': self._mappings,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"[ALIAS_STORE] Saved mappings for {len(self._mappings)} actions")
        except Exception as e:
            logger.warning(f"[ALIAS_STORE] Could not save: {e}")
    
    def learn(self, action_id: str, tw_name: str, composio_name: str):
        """Learn a new mapping from successful execution"""
        if action_id not in self._mappings:
            self._mappings[action_id] = {}
            self._reverse[action_id] = {}
        
        if tw_name != composio_name:  # Only store if different
            self._mappings[action_id][tw_name] = composio_name
            self._reverse[action_id][composio_name] = tw_name
            logger.info(f"[ALIAS_STORE] Learned: {action_id}.{tw_name} -> {composio_name}")
            self.save()
    
    def get_composio_name(self, action_id: str, tw_name: str) -> Optional[str]:
        """Get Composio parameter name for a TaskWeaver name"""
        if action_id in self._mappings:
            return self._mappings[action_id].get(tw_name)
        return None
    
    def get_tw_name(self, action_id: str, composio_name: str) -> Optional[str]:
        """Get TaskWeaver parameter name for a Composio name"""
        if action_id in self._reverse:
            return self._reverse[action_id].get(composio_name)
        return None


class ComposioSchemaFetcher:
    """
    Fetches actual Composio schemas from database.
    No hardcoding - all schema info comes from the source.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def get_schema(self, action_id: str) -> Optional[Dict]:
        """Fetch schema from DB, with caching"""
        if action_id in self._cache:
            return self._cache[action_id]
        
        try:
            # Import Django models
            from apps.integrations.models import ComposioActionSchema
            
            action = ComposioActionSchema.objects.filter(action_id=action_id).first()
            if action and action.parameters_schema:
                schema = action.parameters_schema
                self._cache[action_id] = schema
                return schema
        except Exception as e:
            logger.debug(f"[SCHEMA_FETCHER] Could not fetch {action_id}: {e}")
        
        return None
    
    def get_schema_properties(self, action_id: str) -> Dict[str, Any]:
        """Get just the properties dict from schema"""
        schema = self.get_schema(action_id)
        if schema:
            return schema.get('properties', {})
        return {}
    
    def get_required_params(self, action_id: str) -> Set[str]:
        """Get set of required parameter names"""
        schema = self.get_schema(action_id)
        if schema:
            return set(schema.get('required', []))
        return set()


class ComposioActionExtractor(ast.NodeVisitor):
    """Extract composio_action() calls from code"""
    
    def __init__(self):
        self.calls: List[ComposioCallInfo] = []
        self.function_aliases: Dict[str, str] = {}
    
    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Name) and node.value.id == 'composio_action':
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.function_aliases[target.id] = 'composio_action'
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        func_name = self._get_func_name(node)
        if func_name == 'composio_action' or func_name in self.function_aliases:
            call_info = self._extract_call(node)
            if call_info:
                self.calls.append(call_info)
        self.generic_visit(node)
    
    def _get_func_name(self, node: ast.Call) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
    
    def _extract_call(self, node: ast.Call) -> Optional[ComposioCallInfo]:
        if len(node.args) < 1:
            return None
        
        action_name = None
        is_variable = False
        
        if isinstance(node.args[0], ast.Constant):
            action_name = str(node.args[0].value)
        elif isinstance(node.args[0], ast.Name):
            action_name = f"${node.args[0].id}"
            is_variable = True
        else:
            return None
        
        params = {}
        if len(node.args) >= 2:
            params = self._extract_dict(node.args[1])
        
        for kw in node.keywords:
            if kw.arg == 'params':
                params.update(self._extract_dict(kw.value))
        
        return ComposioCallInfo(
            action_name=action_name,
            params=params,
            line_number=node.lineno,
            is_variable_action=is_variable
        )
    
    def _extract_dict(self, node: ast.AST, depth: int = 0) -> Dict[str, Any]:
        if not isinstance(node, ast.Dict) or depth > 5:
            return {}
        
        result = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant):
                key_str = str(key.value)
            elif isinstance(key, ast.Name):
                key_str = key.id
            else:
                continue
            
            if isinstance(value, ast.Constant):
                result[key_str] = value.value
            elif isinstance(value, ast.Name):
                result[key_str] = f"${value.id}"
            elif isinstance(value, ast.Dict):
                result[key_str] = self._extract_dict(value, depth + 1)
            elif isinstance(value, ast.List):
                result[key_str] = [self._extract_value(el) for el in value.elts]
            else:
                result[key_str] = "<expression>"
        
        return result
    
    def _extract_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return f"${node.id}"
        elif isinstance(node, ast.Dict):
            return self._extract_dict(node)
        return "<expression>"


class DynamicComposioValidator:
    """
    FULLY DYNAMIC validator - no hardcoded aliases!
    
    Workflow:
    1. Extract composio_action() calls from code
    2. For each call, fetch ACTUAL schema from Composio DB
    3. Compare provided params against schema
    4. Use learned mappings to suggest fixes
    5. Learn new mappings from successful executions
    """
    
    def __init__(self, experience_dir: Optional[str] = None):
        self.alias_store = DynamicAliasStore(experience_dir)
        self.schema_fetcher = ComposioSchemaFetcher()
    
    def validate(self, code: str) -> ValidationResult:
        """Validate code against actual Composio schemas"""
        result = ValidationResult()
        
        try:
            tree = ast.parse(code)
            extractor = ComposioActionExtractor()
            extractor.visit(tree)
            
            if not extractor.calls:
                return result
            
            logger.info(f"[DYNAMIC_VALIDATOR] Validating {len(extractor.calls)} composio_action() calls")
            
            for call in extractor.calls:
                if call.is_variable_action:
                    result.warnings.append(f"Line {call.line_number}: Variable action name, skipping validation")
                    continue
                
                # DYNAMIC: Fetch actual schema from DB
                schema_props = self.schema_fetcher.get_schema_properties(call.action_name)
                required_params = self.schema_fetcher.get_required_params(call.action_name)
                
                if not schema_props:
                    result.warnings.append(f"Line {call.line_number}: {call.action_name} - schema not found in DB")
                    continue
                
                # Validate this call
                call_errors = self._validate_call(call, schema_props, required_params)
                result.errors.extend(call_errors)
            
        except SyntaxError as e:
            result.errors.append(ValidationError(
                line=getattr(e, 'lineno', 0) or 0,
                action="<syntax>",
                param_used="",
                message=f"Syntax error: {e}"
            ))
        except Exception as e:
            logger.error(f"[DYNAMIC_VALIDATOR] Error: {e}")
        
        return result
    
    def _validate_call(
        self,
        call: ComposioCallInfo,
        schema_props: Dict[str, Any],
        required_params: Set[str]
    ) -> List[ValidationError]:
        """Validate a single composio_action() call"""
        errors = []
        provided = set(call.params.keys())
        schema_params = set(schema_props.keys())
        
        # 1. Check for missing REQUIRED params
        for req_param in required_params:
            if req_param not in provided:
                # Check if TaskWeaver used a different name that we've learned maps to this
                learned_tw_name = self.alias_store.get_tw_name(call.action_name, req_param)
                
                if learned_tw_name and learned_tw_name in provided:
                    # TaskWeaver used a name we've seen before - suggest the correct one
                    errors.append(ValidationError(
                        line=call.line_number,
                        action=call.action_name,
                        param_used=learned_tw_name,
                        param_expected=req_param,
                        is_wrong_name=True
                    ))
                else:
                    # Check if any provided param looks similar (fuzzy match)
                    similar = self._find_similar_param(req_param, provided)
                    if similar:
                        errors.append(ValidationError(
                            line=call.line_number,
                            action=call.action_name,
                            param_used=similar,
                            param_expected=req_param,
                            is_wrong_name=True
                        ))
                        # Learn this mapping for next time
                        result.discovered_mappings.append((call.action_name, similar, req_param))
                    else:
                        errors.append(ValidationError(
                            line=call.line_number,
                            action=call.action_name,
                            param_used=req_param,
                            is_missing_required=True
                        ))
        
        # 2. Check for unknown params (might be wrong names)
        for param in provided:
            if param not in schema_params:
                # Check if we've learned a mapping for this
                learned_composio = self.alias_store.get_composio_name(call.action_name, param)
                
                if learned_composio and learned_composio in schema_params:
                    errors.append(ValidationError(
                        line=call.line_number,
                        action=call.action_name,
                        param_used=param,
                        param_expected=learned_composio,
                        is_wrong_name=True
                    ))
                else:
                    # Try fuzzy match to find what they might have meant
                    similar = self._find_similar_param(param, schema_params)
                    if similar:
                        errors.append(ValidationError(
                            line=call.line_number,
                            action=call.action_name,
                            param_used=param,
                            param_expected=similar,
                            is_wrong_name=True
                        ))
        
        return errors
    
    def _find_similar_param(self, param: str, candidates: Set[str]) -> Optional[str]:
        """Find a similar parameter name using simple heuristics"""
        param_lower = param.lower().replace('_', '')
        
        for candidate in candidates:
            cand_lower = candidate.lower().replace('_', '')
            
            # Exact match (case insensitive)
            if param_lower == cand_lower:
                return candidate
            
            # One contains the other
            if param_lower in cand_lower or cand_lower in param_lower:
                return candidate
            
            # Common suffix/prefix patterns
            if (param_lower.endswith('email') and cand_lower.endswith('email')) or \
               (param_lower.endswith('id') and cand_lower.endswith('id')) or \
               (param_lower.endswith('name') and cand_lower.endswith('name')):
                return candidate
        
        return None
    
    def learn_from_success(self, action_id: str, tw_params: Dict[str, Any], composio_params: Dict[str, Any]):
        """
        Learn parameter mappings from a successful execution.
        
        Called after a workflow executes successfully to learn which
        TaskWeaver parameter names map to which Composio names.
        """
        for tw_name, value in tw_params.items():
            # Find the Composio param with the same value
            for c_name, c_value in composio_params.items():
                if value == c_value and tw_name != c_name:
                    self.alias_store.learn(action_id, tw_name, c_name)


# ============================================================================
# Integration Functions (for TaskWeaver's code_verification.py)
# ============================================================================

_validator: Optional[DynamicComposioValidator] = None


def get_validator(experience_dir: Optional[str] = None) -> DynamicComposioValidator:
    """Get or create the global validator instance"""
    global _validator
    if _validator is None:
        _validator = DynamicComposioValidator(experience_dir)
    return _validator


def validate_composio_code(
    code: str,
    experience_dir: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate code against Composio schemas.
    
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    
    Integration example for code_verification.py:
    
    ```python
    from composio_validator import validate_composio_code
    
    def code_snippet_verification(...):
        # ... existing validation ...
        
        # Add Composio validation
        composio_valid, composio_errors = validate_composio_code(code_snippet)
        if not composio_valid:
            errors.extend(composio_errors)
        
        return errors
    ```
    """
    validator = get_validator(experience_dir)
    result = validator.validate(code)
    
    if result.is_valid:
        return True, []
    
    return False, [err.format() for err in result.errors]


def learn_from_execution(
    action_id: str,
    tw_params: Dict[str, Any],
    composio_params: Dict[str, Any],
    experience_dir: Optional[str] = None
):
    """
    Learn parameter mappings from successful execution.
    
    Call this after a workflow executes successfully:
    
    ```python
    from composio_validator import learn_from_execution
    
    # After successful Composio API call
    learn_from_execution(
        action_id="GMAIL_SEND_EMAIL",
        tw_params={"to": "user@example.com", "msg": "Hello"},
        composio_params={"recipient_email": "user@example.com", "body": "Hello"}
    )
    ```
    """
    validator = get_validator(experience_dir)
    validator.learn_from_success(action_id, tw_params, composio_params)
