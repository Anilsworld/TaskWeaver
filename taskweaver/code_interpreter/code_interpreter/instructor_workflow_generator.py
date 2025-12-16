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
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Union

from taskweaver.code_interpreter.workflow_schema import WorkflowDefinition, WorkflowNode

logger = logging.getLogger(__name__)


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
    
    This extends the base WorkflowNode with better descriptions and examples
    derived from the tool schemas.
    """
    
    @field_validator('params', mode='before')
    @classmethod
    def ensure_params_populated(cls, v, info):
        """
        Ensure params are populated for agent_with_tools nodes.
        This validator runs before Pydantic validation to provide better error messages.
        """
        if not v or (isinstance(v, dict) and not v):
            node_type = info.data.get('type')
            tool_id = info.data.get('tool_id')
            
            if node_type == 'agent_with_tools' and tool_id:
                logger.warning(
                    f"⚠️ Node has empty params but is agent_with_tools with tool_id={tool_id}. "
                    f"LLM should extract parameter values from user request."
                )
        
        return v if v is not None else {}


class EnhancedWorkflowDefinition(WorkflowDefinition):
    """
    Enhanced WorkflowDefinition that uses EnhancedWorkflowNode.
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
            composio_actions_text: Full text of available Composio actions with descriptions
        
        Returns:
            Tuple of (workflow_definition, error_message)
            - workflow_definition: Validated WorkflowDefinition object if successful, None if failed
            - error_message: Error message if generation failed, None if successful
        """
        try:
            # Build context-aware system prompt
            system_prompt = self._build_system_prompt(
                available_tools=available_tools,
                composio_actions_text=composio_actions_text
            )
            
            # Build user prompt with plan and request
            user_prompt = self._build_user_prompt(
                user_request=user_request,
                plan=plan
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
        if composio_actions_text:
            base_prompt += f"\n\n**DETAILED TOOL DESCRIPTIONS:**\n{composio_actions_text}\n"
        
        return base_prompt
    
    def _build_user_prompt(
        self,
        user_request: str,
        plan: Optional[str] = None
    ) -> str:
        """Build the user prompt with request and plan."""
        prompt_parts = []
        
        if plan:
            prompt_parts.append(f"**PLAN:**\n{plan}\n")
        
        prompt_parts.append(f"**USER REQUEST:**\n{user_request}\n")
        
        prompt_parts.append(
            "\n**TASK:**\n"
            "Generate a complete workflow that accomplishes the user's request. "
            "IMPORTANT:\n"
            "1. Extract ALL parameter values from the user request\n"
            "2. **CRITICAL**: Leave 'edges', 'sequential_edges', and 'parallel_edges' EMPTY ([])\n"
            "3. Extract app_name from tool ID prefix (first part before underscore)\n"
            "4. Use actual response field names from tool schemas, not '.response_field'\n"
            "5. **CRITICAL**: For code_execution nodes, ALL code MUST assign to 'result' variable\n"
            "6. **CRITICAL**: Reference code_execution outputs as ${node_id.execution_result}, not ${node_id}\n"
            "7. Edges will be automatically inferred from $id references in plan (e.g., 'Analyze $1 and $2')"
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
        
        # Fix 1: Generate sequential_edges from depends_on if empty
        # This is a convenience fix, not critical validation
        if not workflow.sequential_edges and any(node.depends_on for node in workflow.nodes):
            sequential_edges = []
            for node in workflow.nodes:
                for dep in node.depends_on:
                    sequential_edges.append((dep, node.id))
            
            workflow.sequential_edges = sequential_edges
            issues_fixed.append(f"Generated {len(sequential_edges)} edges from depends_on")
        
        # Log results
        if issues_fixed:
            logger.info(f"[POST_PROCESS] Auto-fixed {len(issues_fixed)} convenience issue(s):")
            for issue in issues_fixed:
                logger.info(f"  ✓ {issue}")
        
        return workflow


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
