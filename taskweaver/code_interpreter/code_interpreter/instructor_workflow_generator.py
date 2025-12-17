"""
Instructor-based Workflow Generation
====================================
Uses Instructor library to generate validated workflow IR from LLMs.

Replaces:
- workflow_schema_builder.py (manual schema building)
- dynamic_model_validator.py (manual validation)
- Custom retry logic in code_generator.py

With:
- Automatic schema generation from Pydantic models
- Built-in validation with auto-retry
- Clean, maintainable code
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

import instructor
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Union

from taskweaver.code_interpreter.workflow_schema import WorkflowDefinition, WorkflowNode

logger = logging.getLogger(__name__)


class RetryDecision(BaseModel):
    """
    Autogen-inspired retry decision model.
    
    LLM decides whether a validation error is fixable and provides reasoning.
    This prevents wasting retry attempts on unfixable errors (e.g., missing user data).
    """
    retry: bool = Field(
        ...,
        description="Whether to retry generation (true if error is fixable, false otherwise)"
    )
    reason: str = Field(
        ...,
        description="Detailed explanation: what caused the error, why retry/not, and proposed fix if retrying"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the proposed fix (0.0 = not confident, 1.0 = very confident)"
    )


class WorkflowGenerationContext(BaseModel):
    """Context information for workflow generation."""
    available_tools: List[str] = Field(
        ...,
        description="List of available Composio tool IDs"
    )
    user_request: str = Field(
        ...,
        description="The original user request describing what the workflow should accomplish"
    )
    plan: Optional[str] = Field(
        None,
        description="Optional plan or steps to follow"
    )
    current_year: int = Field(
        default_factory=lambda: datetime.now().year,
        description="Current year for date context (e.g., 'Dec 25' should be interpreted as current year)"
    )


class EnhancedWorkflowNode(WorkflowNode):
    """
    Enhanced WorkflowNode with additional validation guidance for LLMs.
    
    ✅ MOVED FROM code_generator.py:
    - Dependencies validation (lines 896-936)
    - App name extraction (auto-correct)
    - Params population validation
    - Tool ID validation
    
    This extends the base WorkflowNode with better descriptions and examples
    derived from the tool schemas.
    """
    
    @field_validator('dependencies', mode='after')
    @classmethod
    def validate_dependencies(cls, v, info):
        """
        ✅ MOVED FROM code_generator.py lines 903-936
        Validates:
        1. Dependencies is a list (was line 903)
        2. Each dependency is a string (was line 926)
        
        Note: dependencies field has default_factory=list in base class,
        so None check is not needed (it will always be a list)
        """
        if not isinstance(v, list):
            raise ValueError(f"dependencies must be a list, got {type(v)}")
        
        for dep in v:
            if not isinstance(dep, str):
                raise ValueError(f"Dependency must be a string node ID, got {type(dep)}: {dep}")
        
        return v
    
    @model_validator(mode='after')
    def extract_app_name_from_tool_id(self):
        """
        ✅ MOVED FROM code_generator.py auto-correction
        Auto-extracts app_name from tool_id if missing.
        
        Example: GMAIL_SEND_EMAIL → app_name: "gmail"
        """
        if self.app_name is None and self.tool_id:
            if '_' in self.tool_id:
                self.app_name = self.tool_id.split('_')[0].lower()
                logger.debug(f"[VALIDATOR] Auto-extracted app_name='{self.app_name}' from tool_id='{self.tool_id}'")
        
        return self
    
    @field_validator('params', mode='before')
    @classmethod
    def ensure_params_populated(cls, v, info):
        """
        ✅ ALREADY EXISTS (line 58-75)
        Enhanced to fail fast on empty params for agent_with_tools.
        
        Validates that agent_with_tools nodes have non-empty params.
        """
        if not v or (isinstance(v, dict) and not v):
            node_type = info.data.get('type')
            tool_id = info.data.get('tool_id')
            
            if node_type == 'agent_with_tools' and tool_id:
                raise ValueError(
                    f"agent_with_tools node with tool_id={tool_id} must have "
                    f"populated 'params' field. Extract values from user request."
                )
        
        return v if v is not None else {}


class EnhancedWorkflowDefinition(WorkflowDefinition):
    """
    Enhanced WorkflowDefinition that uses EnhancedWorkflowNode.
    
    ✅ MOVED FROM code_generator.py:
    - Cross-node dependency validation (lines 933-936)
    - HITL decision field auto-add (auto-correct)
    - Min nodes validation
    """
    nodes: List[EnhancedWorkflowNode] = Field(
        ...,
        min_items=1,
        description=(
            "List of workflow nodes. Each node represents a step in the workflow. "
            "IMPORTANT: For agent_with_tools nodes, extract ALL required parameter values "
            "from the user request and populate the 'params' field. "
            "Example: If user says 'send message to user@example.com with content Hello', "
            "extract: params={'to': 'user@example.com', 'content': 'Hello'}"
        )
    )
    
    @model_validator(mode='after')
    def validate_all_dependencies_exist(self):
        """
        ✅ MOVED FROM code_generator.py line 933-936
        Validates that all node dependencies reference existing nodes.
        """
        all_node_ids = {node.id for node in self.nodes}
        
        for node in self.nodes:
            for dep_id in node.dependencies:
                if dep_id not in all_node_ids:
                    raise ValueError(
                        f"Node '{node.id}' depends on non-existent node '{dep_id}'. "
                        f"Available nodes: {sorted(all_node_ids)}"
                    )
        
        return self
    
    @model_validator(mode='after')
    def add_hitl_decision_field(self):
        """
        ✅ MOVED FROM code_generator.py auto-correction
        Auto-adds decision field to HITL nodes if missing.
        """
        for node in self.nodes:
            if node.type == 'hitl':
                # Check if form_schema exists
                if not hasattr(node, 'form_schema') or not node.form_schema:
                    node.form_schema = {'fields': []}
                
                # Ensure form_schema is a dict with 'fields' key
                if not isinstance(node.form_schema, dict):
                    logger.warning(f"[VALIDATOR] Node '{node.id}' has invalid form_schema type: {type(node.form_schema)}")
                    node.form_schema = {'fields': []}
                
                if 'fields' not in node.form_schema:
                    node.form_schema['fields'] = []
                
                # Check if decision field already exists
                has_decision = any(
                    isinstance(field, dict) and field.get('name') == 'decision' 
                    for field in node.form_schema.get('fields', [])
                )
                
                if not has_decision and not node.decision:
                    # Auto-add decision field
                    node.form_schema['fields'].append({
                        'name': 'decision',
                        'type': 'select',
                        'label': 'Decision',
                        'options': ['Approve', 'Reject'],
                        'required': True
                    })
                    logger.debug(f"[VALIDATOR] Auto-added decision field to HITL node '{node.id}'")
        
        return self


class InstructorWorkflowGenerator:
    """
    Generates workflow IR using Instructor library.
    
    Features:
    - Automatic schema generation from Pydantic models
    - Built-in validation with auto-retry on errors
    - Simplified error handling
    - No manual schema building required
    - Works with both OpenAI and Azure OpenAI clients
    """
    
    def __init__(self, openai_client: Union[OpenAI, AzureOpenAI], max_retries: int = 3):
        """
        Initialize the Instructor workflow generator.
        
        Args:
            openai_client: OpenAI or AzureOpenAI client instance
            max_retries: Maximum number of retry attempts on validation failure
        """
        # Patch the OpenAI/Azure OpenAI client with Instructor
        # Instructor's from_openai works with both OpenAI and AzureOpenAI clients
        self.client = instructor.from_openai(openai_client)
        self.max_retries = max_retries
        self.client_type = type(openai_client).__name__
        logger.info(f"[INSTRUCTOR] Initialized with {self.client_type}, max_retries={max_retries}")
    
    def generate_workflow(
        self,
        user_request: str,
        available_tools: List[str],
        plan: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        composio_actions_text: Optional[str] = None,
    ) -> tuple[Optional[EnhancedWorkflowDefinition], Optional[str]]:
        """
        Generate a validated workflow IR using Instructor.
        
        Args:
            user_request: The user's request describing what the workflow should do
            available_tools: List of available Composio tool IDs
            plan: Optional plan or steps from the planner
            model: OpenAI model to use
            temperature: Temperature for generation
            composio_actions_text: Full text of Composio actions + step guidance (appended in code_generator.py)
        
        Returns:
            Tuple of (workflow_definition, error_message)
            - workflow_definition: Validated WorkflowDefinition object if successful, None if failed
            - error_message: Error message if generation failed, None if successful
        """
        try:
            # Build context-aware system prompt
            # Note: composio_actions_text includes step_guidance, but we'll extract it for user prompt
            system_prompt = self._build_system_prompt(
                available_tools=available_tools,
                composio_actions_text=composio_actions_text
            )
            
            # Build user prompt with plan and request
            # Pass composio_actions_text so we can extract step_guidance for user prompt
            user_prompt = self._build_user_prompt(
                user_request=user_request,
                plan=plan,
                step_guidance_from_system=composio_actions_text
            )
            
            logger.info(
                f"[INSTRUCTOR] Generating workflow for request: {user_request[:100]}..."
            )
            logger.info(f"[INSTRUCTOR] Available tools: {len(available_tools)} tools")
            
            # Use Instructor to generate and validate workflow
            # Instructor automatically:
            # 1. Converts Pydantic model to OpenAI function calling schema
            # 2. Calls the LLM with the schema
            # 3. Validates the response against the Pydantic model
            # 4. Retries on validation errors with detailed error messages
            workflow = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=EnhancedWorkflowDefinition,
                max_retries=self.max_retries,
            )
            
            logger.info(
                f"[INSTRUCTOR] ✅ Successfully generated workflow: "
                f"{len(workflow.nodes)} nodes, "
                f"{len(workflow.sequential_edges or [])} sequential edges, "
                f"{len(workflow.parallel_edges or [])} parallel edges"
            )
            
            # Post-generation validation and auto-correction
            workflow = self._post_process_workflow(workflow)
            
            return workflow, None
            
        except ValidationError as e:
            # Pydantic validation failed after all retries
            error_msg = (
                f"❌ Workflow generation failed after {self.max_retries} attempts.\n"
                f"Validation errors:\n{str(e)}\n\n"
                f"💡 Suggestions:\n"
                f"1. Ensure all required parameters are extracted from the user request\n"
                f"2. Use only tool IDs from the available tools list\n"
                f"3. Ensure dates include the current year (e.g., 'Dec 25' → '{datetime.now().year}-12-25')\n"
                f"4. Check that all node IDs in depends_on and edges actually exist"
            )
            logger.error(f"[INSTRUCTOR] {error_msg}")
            return None, error_msg
            
        except Exception as e:
            # Unexpected error
            error_msg = f"❌ Unexpected error during workflow generation: {str(e)}"
            logger.error(f"[INSTRUCTOR] {error_msg}", exc_info=True)
            return None, error_msg
    
    def _build_system_prompt(
        self,
        available_tools: List[str],
        composio_actions_text: Optional[str] = None
    ) -> str:
        """Build the system prompt with tool context."""
        current_year = datetime.now().year
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        base_prompt = f"""You are a workflow compiler that generates structured workflows from user requests.

**CURRENT CONTEXT:**
- Today's date: {current_date}
- Current year: {current_year}

**AVAILABLE TOOLS:**
You have access to the following Composio tool IDs:
{chr(10).join(f"  - {tool}" for tool in available_tools[:20])}
{"  ... and more" if len(available_tools) > 20 else ""}

**CRITICAL RULES:**

1. **PARAMETER EXTRACTION IS MANDATORY**
   - READ the user request carefully
   - EXTRACT all mentioned values (emails, names, dates, subjects, locations, etc.)
   - POPULATE the params field for agent_with_tools nodes
   - Required parameters MUST have values (never null/empty)

2. **DATE HANDLING**
   - If user mentions dates without year (e.g., "Dec 25"), use current year: {current_year}
   - Format: YYYY-MM-DD (e.g., "{current_year}-12-25")
   - For relative dates: calculate based on {current_date}

3. **TOOL SELECTION**
   - Use ONLY tool IDs from the available tools list above
   - Each agent_with_tools node MUST have a valid tool_id
   - Match tools to user intent based on action description

4. **PARALLEL EXECUTION (Automatic via Dependencies)**
   - ⚠️ DEPRECATED: Do NOT use type="parallel" anymore
   - ✅ NEW APPROACH: List independent operations as separate nodes
   - Parallel execution is AUTOMATIC when nodes have no dependencies
   
   **HOW PARALLEL EXECUTION WORKS:**
   ```python
   # If plan says: "1. Fetch A", "2. Fetch B", "3. Analyze $1 and $2"
   # Generate THREE separate nodes (no parallel wrapper):
   
   {{"id": "fetch_a", "type": "agent_with_tools", "tool_id": "API_A_SEARCH", ...}}
   {{"id": "fetch_b", "type": "agent_with_tools", "tool_id": "API_B_SEARCH", ...}}
   {{"id": "analyze", "type": "agent_only", "description": "Analyze $1 and $2", ...}}
   
   # fetch_a and fetch_b will execute in PARALLEL automatically (no dependencies)
   # analyze will wait for both (dependencies inferred from "$1 and $2")
   ```
   
   **KEY RULES:**
   - ✅ If plan step references "$1" or "$2" → Node depends on those steps
   - ✅ If plan step has NO "$id" references → Node is independent (parallel)
   - ✅ Use dependencies field to control execution order
   - ✅ Nodes with same dependencies execute in parallel automatically

5. **EDGES - DO NOT GENERATE THEM!**
   - ⚠️ CRITICAL: Leave 'edges' field EMPTY ([])
   - ⚠️ CRITICAL: Leave 'sequential_edges' field EMPTY ([])
   - ⚠️ CRITICAL: Leave 'parallel_edges' field EMPTY ([])
   - ✅ Edges will be automatically inferred from:
     * $id references in plan descriptions (e.g., "Analyze $1 and $2")
     * Data flow in params (e.g., params={{"data": "${{step1.result}}"}})
   - This ensures parallel execution works correctly
   
   **WHY NOT GENERATE EDGES:**
   - Manual edges force sequential execution
   - Automatic inference enables parallel execution
   - Dependencies are clearer from $id notation

6. **APP NAME**
   - For agent_with_tools nodes, extract app_name from tool schema
   - Extract from tool ID prefix (e.g., PLATFORM_ACTION → app_name: "platform")

6. **CODE EXECUTION NODES**
   - Use code_execution type for data transformation/formatting
   - **CRITICAL**: ALL code MUST assign final output to variable named 'result'
   - Access tool node outputs: '${{node_id.data.field}}' or '${{node_id.field}}'
   - Access code_execution outputs: '${{node_id.execution_result}}' (if result is primitive)
   - Or: '${{node_id.execution_result.data}}' (if result is dict with 'data' key)
   
   **CODE EXECUTION EXAMPLES:**
   
   Example 1 - Simple String Result:
   ```python
   # Get data from previous tool node
   data = '${{previous_node.data.field}}'
   # Format it
   formatted = f"Processed: {{data}}"
   # ✅ MUST assign to 'result'
   result = formatted
   ```
   
   Example 2 - Dict Result:
   ```python
   # Process multiple fields
   field_a = '${{node_a.field1}}'
   field_b = '${{node_b.field2}}'
   # Create structured result
   result = {{
       'combined_a': field_a,
       'combined_b': field_b
   }}
   ```
   
   Example 3 - Calculations:
   ```python
   # Get numeric data
   value = float('${{node_x.data.numeric_field}}')
   multiplier = 2
   # Calculate
   calculated = value * multiplier
   # ✅ Assign to 'result'
   result = calculated
   ```
   
   **REFERENCING CODE_EXECUTION OUTPUTS:**
   - If code assigns string/number: '${{node_id.execution_result}}'
   - If code assigns dict: '${{node_id.execution_result.key_name}}'
   - Example: code node 'prepare_data' creates result="Hello"
     → Next node references: '${{prepare_data.execution_result}}'

7. **PLACEHOLDER SYNTAX**
   - User inputs: '${{user_input.field_name}}'
   - Tool node outputs: Use ACTUAL response fields from tool schemas
     * Use '${{node_id.data.field}}' or '${{node_id.field}}' based on schema
   - Code execution node outputs:
     * Primitive result (string/number): '${{node_id.execution_result}}'
     * Dict result: '${{node_id.execution_result.field_name}}'
   - Never use generic '.response_field' - always use the actual field name
   - Environment vars: '${{env.VAR_NAME}}'
"""
        
        # Add full tool descriptions if available
        # Note: composio_actions_text includes step_guidance appended in code_generator.py
        if composio_actions_text:
            base_prompt += f"\n\n**DETAILED TOOL DESCRIPTIONS:**\n{composio_actions_text}\n"
            # Check if step_guidance is present
            if "**GENERATE ALL" in composio_actions_text:
                logger.info("[INSTRUCTOR] ✅ Step guidance is present in composio_actions_text")
                # Extract preview
                idx = composio_actions_text.find("**GENERATE ALL")
                preview = composio_actions_text[idx:idx+500] if idx >= 0 else ""
                logger.info(f"[INSTRUCTOR] Step guidance preview: {preview}...")
            else:
                logger.warning("[INSTRUCTOR] ⚠️ Step guidance NOT found in composio_actions_text!")
        
        return base_prompt
    
    def _build_user_prompt(
        self,
        user_request: str,
        plan: Optional[str] = None,
        step_guidance_from_system: Optional[str] = None
    ) -> str:
        """Build the user prompt with request, plan, and step guidance.
        
        NOTE: step_guidance MUST be in the USER prompt (not system prompt) for LLM to follow it!
        The old working code had it in the user message at line 819.
        """
        prompt_parts = []
        
        if plan:
            prompt_parts.append(f"**PLAN:**\n{plan}\n")
        
        # ✅ CRITICAL FIX: Extract step_guidance from composio_actions_text if present
        # In the old code, step_guidance was in the USER message, not system!
        if step_guidance_from_system and "**GENERATE ALL" in step_guidance_from_system:
            idx = step_guidance_from_system.find("**GENERATE ALL")
            step_guidance_section = step_guidance_from_system[idx:]
            prompt_parts.append(step_guidance_section)
            logger.info(f"[INSTRUCTOR] ✅ Moved step_guidance to USER prompt ({len(step_guidance_section)} chars)")
        else:
            logger.warning("[INSTRUCTOR] ⚠️ NO step_guidance found to move to USER prompt!")
        
        prompt_parts.append(f"\n**USER REQUEST:**\n{user_request}\n")
        
        prompt_parts.append(
            "\n**TASK:**\n"
            "Generate a complete workflow that accomplishes the user's request. "
            "IMPORTANT:\n"
            "1. Extract ALL parameter values from the user request\n"
            "2. **🚨 CRITICAL - READ CAREFULLY 🚨**: For EACH node's 'dependencies' field:\n"
            "   - Find the matching step number in \"GENERATE ALL N NODES\" section above\n"
            "   - Copy the EXACT dependencies array from that step's guidance\n"
            "   - Example: If step says 'dependencies=[\"node_1\"]', you MUST write: \"dependencies\": [\"node_1\"]\n"
            "   - DO NOT use empty [] unless guidance explicitly shows dependencies=[]\n"
            "   - This is NOT optional - the schema default is WRONG, follow the guidance!\n"
            "3. **CRITICAL**: Leave 'edges', 'sequential_edges', and 'parallel_edges' EMPTY ([])\n"
            "4. Extract app_name from tool ID prefix (first part before underscore)\n"
            "5. Use actual response field names from tool schemas, not '.response_field'\n"
            "6. **CRITICAL**: For code_execution nodes, ALL code MUST assign to 'result' variable\n"
            "7. **CRITICAL**: Reference code_execution outputs as ${node_id.execution_result}, not ${node_id}\n"
            "8. Edges will be automatically generated from the 'dependencies' arrays you set"
        )
        
        return "\n".join(prompt_parts)
    
    def _post_process_workflow(self, workflow: EnhancedWorkflowDefinition) -> EnhancedWorkflowDefinition:
        """
        ✅ SIMPLIFIED POST-PROCESSING (Validation moved to Pydantic!)
        
        Now that Pydantic validators handle most validation DURING generation:
        - Code execution 'result =' checks → Pydantic @model_validator
        - Placeholder reference validation → Pydantic @model_validator
        - App name extraction → Pydantic @model_validator
        - Generic .response_field checks → Pydantic @model_validator
        
        Post-processing only handles:
        1. Generate sequential_edges from depends_on if empty (convenience)
        
        This eliminates the never-ending post-processing accumulation problem!
        """
        issues_fixed = []
        
        # Fix 1: Generate sequential_edges from dependencies if empty
        # This is a convenience fix, not critical validation
        if not workflow.sequential_edges and any(node.dependencies for node in workflow.nodes):
            sequential_edges = []
            for node in workflow.nodes:
                for dep in node.dependencies:
                    sequential_edges.append((dep, node.id))
            
            workflow.sequential_edges = sequential_edges
            issues_fixed.append(f"Generated {len(sequential_edges)} edges from dependencies")
        
        # Log results
        if issues_fixed:
            logger.info(f"[POST_PROCESS] Auto-fixed {len(issues_fixed)} convenience issue(s):")
            for issue in issues_fixed:
                logger.info(f"  ✓ {issue}")
        
        return workflow


class IntelligentRetryWrapper:
    """
    Wraps Instructor with Autogen-inspired intelligent retry decision.
    
    Instead of blindly retrying on validation errors, asks LLM:
    "Is this error fixable?" before each retry attempt.
    
    Benefits:
    - Stops wasting retries on unfixable errors (e.g., missing user data)
    - Provides transparency (logs LLM's reasoning)
    - Better error messages (LLM explains why it failed)
    
    Cost:
    - +1 LLM call per retry decision
    - Worth it to avoid 2-3 wasted retry attempts
    """
    
    def __init__(self, instructor_gen: InstructorWorkflowGenerator):
        """
        Initialize wrapper with Instructor generator.
        
        Args:
            instructor_gen: InstructorWorkflowGenerator instance to wrap
        """
        self.instructor_gen = instructor_gen
        self.logger = logger
    
    def generate_with_intelligent_retry(
        self,
        user_request: str,
        available_tools: List[str],
        plan: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        composio_actions_text: Optional[str] = None,
        max_retries: int = 3
    ) -> tuple[Optional[EnhancedWorkflowDefinition], Optional[str]]:
        """
        Generate workflow with intelligent retry decision.
        
        On validation error, asks LLM: "Should we retry?"
        If LLM says not fixable, stops immediately instead of wasting retries.
        
        Args:
            user_request: User's workflow request
            available_tools: List of available tool IDs
            plan: Optional plan from planner
            model: Model name for generation
            temperature: Generation temperature
            composio_actions_text: Full tool descriptions + step guidance (appended in code_generator.py)
            max_retries: Maximum retry attempts
        
        Returns:
            Tuple of (workflow_definition, error_message)
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Try Instructor generation (with max_retries=1 to disable its internal retry)
                # Note: step_guidance is now included in composio_actions_text, not as a separate parameter
                workflow, error = self.instructor_gen.generate_workflow(
                    user_request=user_request,
                    available_tools=available_tools,
                    plan=plan,
                    model=model,
                    temperature=temperature,
                    composio_actions_text=composio_actions_text
                )
                
                if not error:
                    # Success!
                    if attempt > 0:
                        self.logger.info(
                            f"[INTELLIGENT_RETRY] ✅ Success on attempt {attempt + 1}/{max_retries}"
                        )
                    return workflow, None
                
                # Instructor failed, store error
                last_error = error
                
                # If this is not the last attempt, ask LLM if retry is worth it
                if attempt < max_retries - 1:
                    retry_decision = self._should_retry(
                        error=error,
                        user_request=user_request,
                        plan=plan,
                        attempt=attempt,
                        max_retries=max_retries,
                        model=model
                    )
                    
                    if not retry_decision.retry:
                        # LLM says not fixable, stop wasting retries
                        self.logger.warning(
                            f"[INTELLIGENT_RETRY] ❌ LLM says unfixable after attempt {attempt + 1}: "
                            f"{retry_decision.reason}"
                        )
                        return None, f"Unfixable error: {retry_decision.reason}\n\nOriginal error: {error}"
                    
                    # LLM says fixable, log reasoning and retry
                    self.logger.info(
                        f"[INTELLIGENT_RETRY] 🔄 Retry {attempt + 1}/{max_retries} "
                        f"(confidence: {retry_decision.confidence:.0%}): {retry_decision.reason}"
                    )
                    continue
                
                # Last attempt failed
                self.logger.error(
                    f"[INTELLIGENT_RETRY] ❌ Failed after {max_retries} attempts"
                )
                return None, last_error
                
            except Exception as e:
                self.logger.error(
                    f"[INTELLIGENT_RETRY] Unexpected error on attempt {attempt + 1}: {e}",
                    exc_info=True
                )
                last_error = str(e)
                
                if attempt == max_retries - 1:
                    return None, last_error
        
        return None, last_error or f"Failed after {max_retries} intelligent retries"
    
    def _should_retry(
        self,
        error: str,
        user_request: str,
        plan: Optional[str],
        attempt: int,
        max_retries: int,
        model: str
    ) -> RetryDecision:
        """
        Ask LLM if retry is worth it (Autogen-inspired).
        
        Analyzes the validation error and decides:
        1. Is this error fixable with another generation attempt?
        2. Is missing data from the user request? (If yes, not fixable)
        3. What specific changes would fix this?
        
        Args:
            error: Validation error message
            user_request: Original user request
            plan: Plan from planner
            attempt: Current attempt number (0-indexed)
            max_retries: Maximum retries allowed
            model: Model name for retry decision
        
        Returns:
            RetryDecision with retry flag, reason, and confidence
        """
        retry_prompt = f"""
You are analyzing a workflow generation validation error to decide if retrying is worthwhile.

ERROR:
{error}

USER REQUEST:
{user_request}

PLAN:
{plan or "No plan provided"}

ATTEMPT:
{attempt + 1} of {max_retries}

Analyze this error carefully:

1. **Is this error fixable with another generation attempt?**
   - YES if: LLM made a mistake in structure, dependencies, or placeholder format
   - NO if: Missing required data from user (e.g., user said "send email" but no email address)

2. **What caused this error?**
   - Identify the root cause (missing field, wrong format, invalid reference, etc.)

3. **If fixable, what specific changes would fix it?**
   - Be precise: "Add email parameter extracted from..." not just "fix the error"

4. **Confidence level?**
   - 0.0-0.3: Low confidence (might not work)
   - 0.4-0.7: Medium confidence (likely to work)
   - 0.8-1.0: High confidence (very likely to work)

Examples of UNFIXABLE errors (return retry=false):
- "User said 'send email to John' but no email address provided" → Missing data
- "User said 'create workflow' but no tools specified" → Ambiguous request

Examples of FIXABLE errors (return retry=true):
- "Node 'node_2' depends on non-existent node 'node_999'" → Fix node ID reference
- "agent_with_tools node missing tool_id field" → Add tool_id from available tools
- "params field is empty for agent_with_tools" → Extract params from user request

Respond with:
- retry: true/false (whether to retry)
- reason: Explanation of decision and proposed fix
- confidence: 0.0-1.0 (how confident you are in the fix)
"""
        
        try:
            # Call LLM with structured output (using same client as Instructor)
            response = self.instructor_gen.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a workflow generation expert. "
                            "Analyze validation errors and decide if they're fixable. "
                            "Be conservative: if missing user data, say not fixable."
                        )
                    },
                    {"role": "user", "content": retry_prompt}
                ],
                response_model=RetryDecision,
                max_retries=1  # Don't retry the retry decision itself!
            )
            
            return response
            
        except Exception as e:
            # If retry decision fails, default to retrying
            # (conservative: give it a chance)
            self.logger.warning(
                f"[INTELLIGENT_RETRY] Failed to get retry decision: {e}. "
                f"Defaulting to retry=True"
            )
            return RetryDecision(
                retry=True,
                reason=f"Failed to analyze error, will retry. Error: {str(e)}",
                confidence=0.5
            )


def create_instructor_client(openai_client: Union[OpenAI, AzureOpenAI], max_retries: int = 3) -> InstructorWorkflowGenerator:
    """
    Factory function to create an Instructor workflow generator.
    
    Args:
        openai_client: OpenAI or AzureOpenAI client instance
        max_retries: Maximum number of retry attempts
    
    Returns:
        InstructorWorkflowGenerator instance
    """
    return InstructorWorkflowGenerator(openai_client, max_retries)


def create_intelligent_instructor_client(
    openai_client: Union[OpenAI, AzureOpenAI],
    max_retries: int = 3
) -> IntelligentRetryWrapper:
    """
    Factory function to create an Instructor workflow generator with intelligent retry.
    
    Args:
        openai_client: OpenAI or AzureOpenAI client instance
        max_retries: Maximum number of retry attempts
    
    Returns:
        IntelligentRetryWrapper instance (recommended over plain Instructor)
    """
    instructor_gen = InstructorWorkflowGenerator(openai_client, max_retries=1)
    return IntelligentRetryWrapper(instructor_gen)
