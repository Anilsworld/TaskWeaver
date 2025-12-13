import ast
import json
import re
from typing import List, Optional, Tuple

from injector import inject


class FunctionCallValidator(ast.NodeVisitor):
    @inject
    def __init__(
        self,
        lines: List[str],
        allowed_modules: Optional[List[str]] = None,
        blocked_modules: Optional[List[str]] = None,
        allowed_functions: Optional[List[str]] = None,
        blocked_functions: Optional[List[str]] = None,
        allowed_variables: Optional[List[str]] = None,
    ):
        self.lines = lines
        self.errors = []
        self.allowed_modules = allowed_modules
        self.blocked_modules = blocked_modules
        assert (
            allowed_modules is None or blocked_modules is None
        ), "Only one of allowed_modules or blocked_modules can be set."
        self.blocked_functions = blocked_functions
        self.allowed_functions = allowed_functions
        assert (
            allowed_functions is None or blocked_functions is None
        ), "Only one of allowed_functions or blocked_functions can be set."
        self.allowed_variables = allowed_variables

    def _is_allowed_function_call(self, func_name: str) -> bool:
        if self.allowed_functions is not None:
            if len(self.allowed_functions) > 0:
                return func_name in self.allowed_functions
            return False
        if self.blocked_functions is not None:
            if len(self.blocked_functions) > 0:
                return func_name not in self.blocked_functions
            return True
        return True

    def visit_Call(self, node):
        if self.allowed_functions is None and self.blocked_functions is None:
            return

        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            raise ValueError(f"Unsupported function call: {node.func}")

        if not self._is_allowed_function_call(function_name):
            self.errors.append(
                f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                f"=> Function '{function_name}' is not allowed.",
            )

    def _is_allowed_module_import(self, mod_name: str) -> bool:
        if self.allowed_modules is not None:
            if len(self.allowed_modules) > 0:
                return mod_name in self.allowed_modules
            return False
        if self.blocked_modules is not None:
            if len(self.blocked_modules) > 0:
                return mod_name not in self.blocked_modules
            return True
        return True

    def visit_Import(self, node):
        if self.allowed_modules is None and self.blocked_modules is None:
            return

        for alias in node.names:
            if "." in alias.name:
                module_name = alias.name.split(".")[0]
            else:
                module_name = alias.name

            if not self._is_allowed_module_import(module_name):
                self.errors.append(
                    f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                    f"=> Importing module '{module_name}' is not allowed. ",
                )

    def visit_ImportFrom(self, node):
        if self.allowed_modules is None and self.blocked_modules is None:
            return

        if "." in node.module:
            module_name = node.module.split(".")[0]
        else:
            module_name = node.module

        if not self._is_allowed_module_import(module_name):
            self.errors.append(
                f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                f"=>  Importing from module '{node.module}' is not allowed.",
            )

    def _is_allowed_variable(self, var_name: str) -> bool:
        if self.allowed_variables is not None:
            if len(self.allowed_variables) > 0:
                return var_name in self.allowed_variables
            return False
        return True

    def visit_Assign(self, node: ast.Assign):
        if self.allowed_variables is None:
            return

        for target in node.targets:
            variable_names = []
            if isinstance(target, ast.Name):
                variable_names.append(target.id)
            else:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        variable_names.append(name.id)
            for variable_name in variable_names:
                if not self._is_allowed_variable(variable_name):
                    self.errors.append(
                        f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                        f"=> Assigning to {variable_name} is not allowed.",
                    )

    def generic_visit(self, node):
        super().generic_visit(node)


def format_code_correction_message(code: str = "", error_details: str = "") -> str:
    """
    Format error message for LLM self-correction.
    
    Parso/Black already provides detailed context, so we just wrap it nicely.
    """
    base_message = (
        "The generated code has been verified and some errors are found. "
        "If you think you can fix the problem by rewriting the code, "
        "please do it and try again.\n"
        "Otherwise, please explain the problem to me."
    )
    
    if error_details:
        return f"{base_message}\n\n{error_details}"
    
    return base_message


def separate_magics_and_code(input_code: str) -> Tuple[List[str], str, List[str]]:
    line_magic_pattern = re.compile(r"^\s*%\s*[a-zA-Z_]\w*")
    cell_magic_pattern = re.compile(r"^\s*%%\s*[a-zA-Z_]\w*")
    shell_command_pattern = re.compile(r"^\s*!")

    magics = []
    python_code = []
    package_install_commands = []

    lines = input_code.splitlines()
    inside_cell_magic = False

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue

        if inside_cell_magic:
            magics.append(line)
            if not line.strip():
                inside_cell_magic = False
            continue
        if line_magic_pattern.match(line) or shell_command_pattern.match(line):
            # Check if the line magic or shell command is a package installation command
            if "pip install" in line or "conda install" in line:
                package_install_commands.append(line)
            else:
                magics.append(line)
        elif cell_magic_pattern.match(line):
            inside_cell_magic = True
            magics.append(line)
        else:
            python_code.append(line)
    python_code_str = "\n".join(python_code)
    return magics, python_code_str, package_install_commands


def code_snippet_verification(
    code_snippet: str,
    code_verification_on: bool = False,
    allowed_modules: Optional[List[str]] = None,
    blocked_modules: Optional[List[str]] = None,
    allowed_functions: Optional[List[str]] = None,
    blocked_functions: Optional[List[str]] = None,
    allowed_variables: Optional[List[str]] = None,
    session_variables: Optional[dict] = None,  # ✅ NEW: Follow plugin pattern for constraint enforcement
) -> Optional[List[str]]:
    if not code_verification_on:
        print("[CODE_VERIFICATION] ⚠️ Code verification is DISABLED")
        return None
    
    print(f"[CODE_VERIFICATION] 🔍 Code verification ENABLED - checking {len(code_snippet)} chars...")
    errors = []
    try:
        magics, python_code, _ = separate_magics_and_code(code_snippet)
        if len(magics) > 0:
            errors.append(f"Magic commands except package install are not allowed. Details: {magics}")
        
        # ========================================================================
        # CRITICAL FIX: Run constraint check BEFORE ast.parse()
        # ========================================================================
        # ast.parse() throws SyntaxError on truncated JSON, which skips WORKFLOW validation.
        # We MUST check constraints BEFORE parsing to catch anti-patterns even in truncated code.
        # ========================================================================
        if "WORKFLOW" in python_code:
            print("[CODE_VERIFICATION] 🔍 PRE-PARSE: Detected WORKFLOW, running constraint check...")
            workflow_errors = validate_workflow_structure(python_code, session_variables=session_variables)
            if workflow_errors:
                print(f"[CODE_VERIFICATION] ❌ PRE-PARSE: WORKFLOW validation failed with {len(workflow_errors)} errors")
                return workflow_errors  # Return immediately - don't continue parsing
            else:
                print("[CODE_VERIFICATION] ✅ PRE-PARSE: WORKFLOW constraint check passed!")
        
        tree = ast.parse(python_code)

        processed_lines = []
        for line in python_code.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            processed_lines.append(line)
        validator = FunctionCallValidator(
            lines=processed_lines,
            allowed_modules=allowed_modules,
            blocked_modules=blocked_modules,
            allowed_functions=allowed_functions,
            blocked_functions=blocked_functions,
            allowed_variables=allowed_variables,
        )
        validator.visit(tree)
        errors.extend(validator.errors)
        
        # ✅ COMPOSIO SCHEMA VALIDATION (Dynamic, No Hardcoding!)
        # This hooks into TaskWeaver's built-in retry mechanism
        # If composio_action() has wrong params, retry with correct ones
        composio_errors = validate_composio_actions(python_code)
        if composio_errors:
            errors.extend(composio_errors)
        
        # Note: WORKFLOW validation moved to pre-parse check (before ast.parse())
        # This ensures constraint enforcement even on truncated/syntax-error code
        
        return errors
    except SyntaxError as e:
        # ✅ ENHANCED SYNTAX CHECKING: Parso + Black + Pydantic
        return enhanced_syntax_validation(python_code, e)


def enhanced_syntax_validation(python_code: str, original_error: SyntaxError) -> List[str]:
    """
    Enhanced syntax checking with fault-tolerant parsing.
    
    Uses proven open-source tools:
    1. Black - Auto-fix formatting issues
    2. Parso - Fault-tolerant parsing with exact error positions
    3. Pydantic - Structure validation (if syntax is valid)
    
    Args:
        python_code: The code to validate
        original_error: The original SyntaxError from ast.parse
    
    Returns:
        List of error messages
    """
    errors = []
    
    # ============================================
    # STEP 1: Try Black for Auto-Fixing
    # ============================================
    try:
        import black
        from black import FileMode
        
        # Try to auto-fix with Black
        fixed_code = black.format_str(python_code, mode=FileMode())
        
        # Black succeeded! Try parsing the fixed code
        try:
            ast.parse(fixed_code)
            # Success! But don't modify original code, just note it could be fixed
            errors.append(
                "[!] Code has formatting issues but can be auto-fixed.\n"
                "**Hint:** Use consistent bracket placement and avoid deep nesting."
            )
        except SyntaxError:
            pass  # Black fixed formatting but syntax still broken, continue to Parso
    except Exception:
        pass  # Black not available or failed, continue
    
    # ============================================
    # STEP 2: Parso for Detailed Error Messages
    # ============================================
    try:
        import parso
        
        # Load grammar for current Python version (3.x)
        grammar = parso.load_grammar(version="3.10")
        module = grammar.parse(python_code)
        
        # Use grammar.iter_errors() to get errors (correct Parso API)
        parso_errors = list(grammar.iter_errors(module))
        
        if parso_errors:
            for err in parso_errors:
                line = err.start_pos[0]
                col = err.start_pos[1]
                lines = python_code.split('\n')
                
                if line <= len(lines):
                    code_line = lines[line - 1]
                    
                    # Show context with exact column position
                    start = max(0, col - 20)
                    end = min(len(code_line), col + 20)
                    
                    before = code_line[start:col]
                    after = code_line[col:end]
                    context = f"...{before} >>HERE<< {after}..."
                    
                    errors.append(
                        f"[!] Syntax Error at Line {line}, Column {col}:\n"
                        f"   {err.message}\n"
                        f"   {context}\n"
                    )
            
            return errors
    
    except ImportError:
        # Parso not installed, fall through to basic error
        pass
    except Exception as parso_ex:
        # Parso failed, fall through to basic error
        import logging
        logging.getLogger(__name__).debug(f"Parso parsing failed: {parso_ex}")
        pass
    
    # ============================================
    # STEP 3: Pydantic Structure Validation
    # ============================================
    # Only try Pydantic if we have a WORKFLOW dict that might be structurally wrong
    if "WORKFLOW" in python_code and not errors:
        try:
            import re
            workflow_match = re.search(r'WORKFLOW\s*=\s*(\{.*\})\s*(?:result\s*=|$)', python_code, re.DOTALL)
            
            print(f"[CODE_VERIFICATION] WORKFLOW pattern match: {bool(workflow_match)}")
            
            if workflow_match:
                print("[CODE_VERIFICATION] 📋 WORKFLOW dict detected! Running Pydantic + WorkflowIR validation...")
                from taskweaver.code_interpreter.workflow_schema import (
                    validate_workflow_dict,
                    format_workflow_validation_error
                )
                
                workflow_str = workflow_match.group(1)
                try:
                    # CRITICAL FIX: Clean the extracted string before JSON parsing
                    # Remove any trailing "result = WORKFLOW" if present
                    import re
                    workflow_str_clean = re.sub(r'\s*\nresult\s*=.*$', '', workflow_str, flags=re.DOTALL)
                    
                    # Convert Python syntax to JSON
                    json_str = workflow_str_clean.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                    # Convert Python tuples to JSON arrays: ("a", "b") → ["a", "b"]
                    json_str = re.sub(r'\(([^)]+)\)', r'[\1]', json_str)
                    workflow_dict = json.loads(json_str)
                    
                    # Validate structure with Pydantic + WorkflowIR
                    print(f"[CODE_VERIFICATION] 🔍 Calling validate_workflow_dict() with {len(workflow_dict.get('nodes', []))} nodes...")
                    is_valid, workflow_obj, pydantic_errors = validate_workflow_dict(workflow_dict)
                    print(f"[CODE_VERIFICATION] ✅ validate_workflow_dict() returned: is_valid={is_valid}, errors={len(pydantic_errors)}")
                    
                    if not is_valid:
                        print(f"[CODE_VERIFICATION] ❌ Validation FAILED: {pydantic_errors}")
                        return [format_workflow_validation_error(pydantic_errors, python_code)]
                    else:
                        print(f"[CODE_VERIFICATION] ✅ Validation PASSED!")
                
                except (json.JSONDecodeError, ValueError):
                    pass  # Can't extract dict, fall through to basic error
        except Exception:
            pass  # Pydantic validation failed
    
    # ============================================
    # STEP 4: Fallback to Basic Error Message
    # ============================================
    if not errors:
        error_msg = f"Syntax error: {original_error}"
        if hasattr(original_error, 'lineno') and original_error.lineno:
            lines = python_code.split('\n')
            if 0 < original_error.lineno <= len(lines):
                error_line = lines[original_error.lineno - 1]
                error_msg += f"\n  -> Line {original_error.lineno}: {error_line.strip()}"
        errors.append(error_msg)
    
    return errors


def validate_workflow_structure(python_code: str, session_variables: Optional[dict] = None) -> List[str]:
    """
    Validate WORKFLOW dict structure using Pydantic + WorkflowIR.
    
    This runs for ALL WORKFLOW dicts, not just syntax errors.
    
    ARCHITECTURAL FIX (following battle-tested plugin pattern):
    - Checks session variables for constraint enforcement (like composio_action, form_collect)
    - BLOCKS generation if constraints violated (hard stop, not advice)
    - Scalable: Can add more constraint checks (_max_nodes, _force_compact, etc.)
    """
    errors = []
    session_vars = session_variables or {}
    
    try:
        import re
        # CRITICAL FIX: Extract ONLY the WORKFLOW dict, stop before "result ="
        # Match: WORKFLOW = {...everything...} but STOP at line starting with "result ="
        # This prevents json.loads() from choking on "result = WORKFLOW" line
        # Use lazy match with lookahead to stop before \nresult
        workflow_match = re.search(r'WORKFLOW\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\})', python_code, re.DOTALL)
        if not workflow_match:
            # Fallback: try simpler pattern for truncated JSON
            workflow_match = re.search(r'WORKFLOW\s*=\s*(\{.*)', python_code, re.DOTALL)
        
        print(f"[CODE_VERIFICATION] WORKFLOW pattern match: {bool(workflow_match)}")
        
        if workflow_match:
            print("[CODE_VERIFICATION] 📋 WORKFLOW dict detected! Running Pydantic + WorkflowIR validation...")
            from taskweaver.code_interpreter.workflow_schema import (
                validate_workflow_dict,
                format_workflow_validation_error
            )
            
            workflow_str = workflow_match.group(1)
            
            # ========================================================================
            # ARCHITECTURAL FIX: PRE-PARSE CONSTRAINT CHECK (SCALABLE FIX)
            # ========================================================================
            # Check constraints BEFORE parsing dict, so truncation doesn't block enforcement.
            # Count nodes in raw string using regex (works even on truncated JSON).
            # ========================================================================
            use_loops = session_vars.get('_use_loops', 'false')
            if use_loops == 'true' or use_loops is True:
                print(f"[CODE_VERIFICATION] 🚫 PRE-PARSE CHECK: _use_loops=True, counting nodes in raw string...")
                
                # Count node IDs in raw string (works on truncated JSON!)
                node_id_pattern = r'"id"\s*:\s*"([^"]+)"'
                node_ids = re.findall(node_id_pattern, workflow_str)
                node_count_estimate = len(node_ids)
                
                print(f"[CODE_VERIFICATION] 📊 Detected {node_count_estimate} node IDs: {node_ids[:10]}...")
                
                # Check if pattern looks like parallel nodes (fetch_X, send_X patterns)
                parallel_patterns = {}
                for node_id in node_ids:
                    # Extract base pattern (e.g., "fetch_gmail" → "fetch", "send_outlook" → "send")
                    if '_' in node_id:
                        base_pattern = node_id.split('_')[0]
                        parallel_patterns[base_pattern] = parallel_patterns.get(base_pattern, 0) + 1
                
                max_repetition = max(parallel_patterns.values()) if parallel_patterns else 0
                print(f"[CODE_VERIFICATION] 📊 Pattern analysis: {parallel_patterns} (max repetition: {max_repetition})")
                
                # Check if workflow explicitly declares parallel_groups (the smoking gun!)
                has_parallel_groups = '"parallel_groups"' in workflow_str or "'parallel_groups'" in workflow_str
                if has_parallel_groups:
                    print(f"[CODE_VERIFICATION] 🚨 Detected explicit parallel_groups declaration!")
                
                # ENFORCEMENT: REJECT if clear loop anti-pattern detected
                # Scenario 1: 4+ nodes with same prefix (fetch_X, send_X) = should be a loop!
                # Scenario 2: Explicit parallel_groups with 3+ parallel nodes
                # Scenario 3: Large workflow (8+ nodes) with repetition
                should_reject = (
                    (max_repetition >= 4) or  # 4+ parallel nodes with same pattern
                    (has_parallel_groups and max_repetition >= 3) or  # Explicit parallel declaration
                    (max_repetition >= 3 and node_count_estimate >= 8)  # Original threshold
                )
                
                if should_reject:
                    complexity_guidance = session_vars.get('_complexity_guidance', '')
                    detected_pattern = session_vars.get('_detected_pattern', 'unknown')
                    
                    print(f"[CODE_VERIFICATION] ❌ CONSTRAINT VIOLATION!")
                    print(f"   - Node count: {node_count_estimate}")
                    print(f"   - Max repetition: {max_repetition}")
                    print(f"   - Has parallel_groups: {has_parallel_groups}")
                    print(f"   - Pattern: {parallel_patterns}")
                    
                    # Identify the specific anti-pattern for clear feedback
                    violation_reason = ""
                    if max_repetition >= 4:
                        violation_reason = f"4+ parallel nodes with same pattern ('{list(parallel_patterns.keys())}': {max_repetition}x)"
                    elif has_parallel_groups and max_repetition >= 3:
                        violation_reason = f"Explicit parallel_groups declaration with {max_repetition}x repetition"
                    else:
                        violation_reason = f"{node_count_estimate} nodes with {max_repetition}x repetition"
                    
                    # ========================================================================
                    # LANGCHAIN/LANGGRAPH PATTERN: Show concrete BEFORE/AFTER code example
                    # ========================================================================
                    # LangGraph Send API: ONE node executed N times, not N separate nodes
                    # ========================================================================
                    error_msg = (
                        f"🚫 WORKFLOW REJECTED: Loop constraint violated!\n\n"
                        f"**ANTI-PATTERN DETECTED:**\n"
                        f"  - {violation_reason}\n"
                        f"  - Node IDs: {', '.join(node_ids[:8])}{'...' if len(node_ids) > 8 else ''}\n"
                        f"  - Pattern analysis: {parallel_patterns}\n"
                        f"  - Complexity guidance: {detected_pattern}\n\n"
                        f"**SYSTEM ANALYSIS:**{complexity_guidance}\n\n"
                        f"**WHY THIS FAILED:**\n"
                        f"  - Token limit: {node_count_estimate} nodes × ~300 tokens = ~{node_count_estimate * 300} tokens → TRUNCATION ❌\n\n"
                        f"**❌ YOUR CODE (WRONG):**\n"
                        f"```python\n"
                        f"WORKFLOW = {{\n"
                        f"  \"nodes\": [\n"
                        f"    {{\"id\": \"fetch_gmail\", \"type\": \"agent_with_tools\", ...}},    # Node 1\n"
                        f"    {{\"id\": \"fetch_outlook\", \"type\": \"agent_with_tools\", ...}},  # Node 2\n"
                        f"    {{\"id\": \"fetch_instagram\", \"type\": \"agent_with_tools\", ...}}, # Node 3\n"
                        f"    {{\"id\": \"fetch_facebook\", \"type\": \"agent_with_tools\", ...}},  # Node 4\n"
                        f"    # ❌ 4 separate node definitions = TOO MANY TOKENS!\n"
                        f"  ]\n"
                        f"}}\n"
                        f"```\n\n"
                        f"**✅ CORRECT CODE (LANGGRAPH Send PATTERN):**\n"
                        f"```python\n"
                        f"WORKFLOW = {{\n"
                        f"  \"nodes\": [\n"
                        f"    # Step 1: Define list to iterate over\n"
                        f"    {{\"id\": \"platforms\", \"type\": \"agent_only\",\n"
                        f"     \"params\": {{\"items\": [\"gmail\", \"outlook\", \"instagram\", \"facebook\"]}}}},\n\n"
                        f"    # Step 2: Loop node with ONE nested node (not 4!)\n"
                        f"    {{\"id\": \"fetch_loop\", \"type\": \"loop\",\n"
                        f"     \"loop_over\": \"${{from_step:platforms.items}}\",\n"
                        f"     \"nodes\": [\n"
                        f"       {{\"id\": \"fetch_message\", \"type\": \"agent_with_tools\",  # ✅ ONE node!\n"
                        f"        \"tool_id\": \"FETCH\", \"params\": {{\"platform\": \"${{loop_item}}\"}}}}\n"
                        f"     ]}},\n\n"
                        f"    # Step 3: Aggregate results\n"
                        f"    {{\"id\": \"prepare_drafts\", \"type\": \"agent_only\",\n"
                        f"     \"params\": {{\"all_messages\": \"${{from_loop:fetch_loop.results}}\"}},\n"
                        f"     \"depends_on\": [\"fetch_loop\"]}}\n"
                        f"  ],\n"
                        f"  \"sequential_edges\": [(\"platforms\", \"fetch_loop\"), (\"fetch_loop\", \"prepare_drafts\")]\n"
                        f"}}\n"
                        f"```\n\n"
                        f"**THE KEY DIFFERENCE (LangGraph Send API):**\n"
                        f"  - ❌ WRONG: 4 separate nodes (fetch_gmail, fetch_outlook...)\n"
                        f"  - ✅ CORRECT: 1 node inside loop (fetch_message)\n"
                        f"  - Runtime: Loop executes fetch_message 4x with different platform states\n"
                        f"  - Token count: ~500 tokens (loop) vs ~{node_count_estimate * 300} tokens (parallel)\n\n"
                        f"**This is a HARD CONSTRAINT. Follow the CORRECT CODE example above.**"
                    )
                    return [error_msg]
                else:
                    print(f"[CODE_VERIFICATION] ✅ PRE-PARSE CHECK PASSED:")
                    print(f"   - {node_count_estimate} nodes, {max_repetition}x max repetition")
                    print(f"   - Has parallel_groups: {has_parallel_groups}")
                    print(f"   - Thresholds: max_rep>=4 OR (parallel_groups AND max_rep>=3) OR (8+ nodes AND max_rep>=3)")
            
            try:
                # CRITICAL FIX: Clean the extracted string before JSON parsing
                # Remove any trailing "result = WORKFLOW" if present
                workflow_str_clean = re.sub(r'\s*\nresult\s*=.*$', '', workflow_str, flags=re.DOTALL)
                
                # Convert Python syntax to JSON
                json_str = workflow_str_clean.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                # Convert Python tuples to JSON arrays: ("a", "b") → ["a", "b"]
                json_str = re.sub(r'\(([^)]+)\)', r'[\1]', json_str)
                workflow_dict = json.loads(json_str)
                node_count = len(workflow_dict.get('nodes', []))
                
                # ========================================================================
                # POST-PARSE VALIDATION: Verify the dict is structurally sound
                # ========================================================================
                # Pre-parse check already enforced constraints, this just confirms the parse succeeded
                print(f"[CODE_VERIFICATION] ✅ Successfully parsed workflow dict with {node_count} nodes")
                
                # ========================================================================
                # STANDARD VALIDATION: Pydantic + WorkflowIR
                # ========================================================================
                print(f"[CODE_VERIFICATION] 🔍 Calling validate_workflow_dict() with {node_count} nodes...")
                is_valid, workflow_obj, pydantic_errors = validate_workflow_dict(workflow_dict)
                print(f"[CODE_VERIFICATION] ✅ validate_workflow_dict() returned: is_valid={is_valid}, errors={len(pydantic_errors)}")
                
                if not is_valid:
                    print(f"[CODE_VERIFICATION] ❌ Validation FAILED: {pydantic_errors}")
                    return [format_workflow_validation_error(pydantic_errors, python_code)]
                else:
                    print(f"[CODE_VERIFICATION] ✅ Validation PASSED!")
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[CODE_VERIFICATION] ⚠️ Could not parse WORKFLOW dict: {e}")
                errors.append(f"WORKFLOW dict parse error: {e}")
    except Exception as e:
        print(f"[CODE_VERIFICATION] ⚠️ Workflow validation error: {e}")
        import logging
        logging.getLogger(__name__).debug(f"Workflow validation failed: {e}")
    
    return errors


def validate_composio_actions(code: str) -> List[str]:
    """
    Validate composio_action() calls against actual Composio schemas.
    
    ✅ FULLY DYNAMIC - No hardcoding!
    - Fetches schemas from Composio DB
    - Uses learned mappings from Experience system
    - Returns errors that trigger TaskWeaver's built-in retry
    """
    errors = []
    try:
        # Import dynamically to avoid circular imports
        from TaskWeaver.project.plugins.composio_validator import (
            DynamicComposioValidator, 
            validate_composio_code
        )
        
        is_valid, error_list = validate_composio_code(code)
        if not is_valid:
            errors.extend(error_list)
            
    except ImportError:
        # Composio validator not available, skip
        pass
    except Exception as e:
        # Don't fail on validation errors, just log
        import logging
        logging.getLogger(__name__).debug(f"Composio validation skipped: {e}")
    
    return errors
