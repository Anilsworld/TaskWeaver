"""
OpenAI Function Calling Schema Builder with Dynamic Tool Parameters
====================================================================
Generates OpenAI-compatible function calling schemas with dynamic injection
of tool-specific parameter schemas from Composio.

This eliminates 280+ lines of manual schema building and ensures the LLM
knows exactly what parameters to extract from user requests.

✅ INTEGRATES WITH: dynamic_model_validator.py
   - Shares the same ToolSchemaCache singleton
   - Single source of truth for schema loading
   - No code duplication
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkflowSchemaBuilder:
    """
    Builds OpenAI function calling schemas with dynamic tool parameter injection.
    
    Architecture:
    1. Reuse ToolSchemaCache from dynamic_model_validator (singleton)
    2. Filter to only the selected tools (5-50 per workflow)
    3. Dynamically inject their parameter schemas into the workflow schema
    4. LLM sees exact params for each tool → better extraction
    
    ✅ DRY PRINCIPLE: Shares schema cache with dynamic_model_validator.py
    """
    
    def __init__(self):
        """Initialize builder - simplified without dynamic schemas."""
        # Skip dynamic tool schema loading for now
        self.schema_cache = None
        logger.info("[SCHEMA_BUILDER] Initialized (dynamic schemas disabled)")
    
    def _get_tool_schemas(self, tool_ids: List[str]) -> Dict[str, Dict]:
        """
        Get schemas for specific tools using shared cache.
        
        Args:
            tool_ids: List of tool IDs to get schemas for
        
        Returns:
            Dict mapping tool_id → tool schema
        """
        if not self.schema_cache:
            return {}  # Dynamic schemas disabled
        
        schemas = {}
        for tool_id in tool_ids:
            schema = self.schema_cache.get_schema(tool_id)
            if schema:
                schemas[tool_id] = schema
        return schemas
    
    def build_function_schema(self, tool_ids: List[str]) -> Dict[str, Any]:
        """
        Build OpenAI function calling schema with dynamic tool parameter injection.
        
        Args:
            tool_ids: List of Composio action IDs (e.g., ["GMAIL_SEND_EMAIL", "COMPOSIO_SEARCH_FLIGHTS"])
        
        Returns:
            OpenAI function schema dict with tool-specific parameter schemas embedded
        
        Architecture:
        - Base structure from manual schema (triggers, edges, node types)
        - Tool parameter schemas dynamically injected from shared ToolSchemaCache
        - LLM sees: "For COMPOSIO_SEARCH_FLIGHTS, params MUST include departure_id, arrival_id, outbound_date"
        """
        # Get schemas for selected tools using shared cache
        tool_schemas = self._get_tool_schemas(tool_ids)
        
        # Build tool-specific parameter descriptions
        tool_param_hints = self._build_tool_param_hints(tool_ids, tool_schemas)
        
        # ============================================================
        # OpenAI Function Calling Schema
        # ============================================================
        schema = {
            "name": "create_workflow_ir",
            "description": (
                "🎯 Generate a complete, executable workflow from user request.\n\n"
                f"{tool_param_hints}\n\n"
                "=" * 70 + "\n"
                "🚨 CRITICAL: PARAMETER EXTRACTION IS MANDATORY!\n"
                "=" * 70 + "\n"
                "For EVERY agent_with_tools node, you MUST:\n"
                "1. READ the user's request word-by-word\n"
                "2. MATCH words/phrases to tool parameter names/descriptions (see schemas above)\n"
                "3. EXTRACT those values using the patterns shown in param descriptions\n"
                "4. POPULATE the 'params' field with extracted values\n"
                "5. REQUIRED parameters MUST have values (never null/empty!)\n\n"
                "=" * 70 + "\n"
                "🚨 CRITICAL: FORM vs HITL NODE TYPES!\n"
                "=" * 70 + "\n"
                "⚠️ NEVER USE 'form' FOR APPROVALS! Follow these rules:\n"
                "✅ USE 'hitl' (NOT 'form') when:\n"
                "   - Step involves 'approve', 'review', 'authorize', 'validate', 'confirm'\n"
                "   - Human makes a decision based on processed data\n"
                "   - User is reviewing/approving results (not providing initial input)\n"
                "   Examples: approve_booking, review_draft, authorize_payment\n\n"
                "✅ USE 'form' (NOT 'hitl') when:\n"
                "   - Collecting INITIAL data from user to START workflow\n"
                "   - User is PROVIDING information (not reviewing results)\n"
                "   - Gathering search criteria, contact details, preferences\n"
                "   Examples: collect_search_criteria, get_user_details\n\n"
                "⚠️ SELF-CHECK Before Returning Workflow:\n"
                "Ask yourself:\n"
                "  ✅ Did I read the user's request and extract ALL relevant values?\n"
                "  ✅ Does every agent_with_tools node have a populated params field?\n"
                "  ✅ Did I check each tool's REQUIRED params and provide values?\n"
                "  ✅ Are dates in YYYY-MM-DD format with current year?\n"
                "  ✅ Did I use parameter examples from schemas as formatting guides?\n"
                "  ✅ Did I generate sensible defaults for missing but required fields?\n"
                "  ✅ Did I use 'hitl' (NOT 'form') for ALL approval/review steps?\n\n"
                "If ANY answer is NO, you MUST fix it before returning!\n\n"
                "🔗 WORKFLOW CONNECTIVITY:\n"
                "- Edges will be auto-generated based on step order from the plan\n"
                "- You only need to generate the NODES (one per step)\n"
                "- Sequential connections are handled automatically\n\n"
                "📤 CODE NODES:\n"
                "- Use ONLY response fields from tool schemas (see 'Returns:' above)\n"
                "- Access: node_id.field_name\n"
                "- Never invent field names!"
            ),
            "parameters": {
                "type": "object",
                "description": (
                    "Workflow Intermediate Representation (IR)\n\n"
                    "⚠️ BEFORE RETURNING THIS WORKFLOW, VERIFY:\n"
                    "1. Every agent_with_tools node HAS a 'params' field with extracted values\n"
                    "2. All REQUIRED parameters (see tool schemas above) have non-null values\n"
                    "3. Dates are in YYYY-MM-DD format with current year\n"
                    "4. If user says 'my messages/emails/account', DO NOT create form nodes - use authenticated context\n\n"
                    "DO NOT return empty params! Extract values from user request!"
                ),
                "properties": {
                    # ============================================================
                    # 1️⃣ TRIGGERS
                    # ============================================================
                    "triggers": {
                                "type": "array",
                                "description": "Workflow entry points (time-based or event-based)",
                                "items": {
                                    "oneOf": [
                                        {
                                            "type": "object",
                                            "description": "Time-based trigger (use 'schedule', NOT 'time')",
                                            "properties": {
                                                "type": {"type": "string", "enum": ["schedule"]},
                                                "cron": {"type": "string", "description": "Cron expression"},
                                                "start_node": {"type": "string"}
                                            },
                                            "required": ["type", "cron", "start_node"],
                                            "additionalProperties": False
                                        },
                                        {
                                            "type": "object",
                                            "description": "Event-based trigger (use 'event', NOT 'webhook')",
                                            "properties": {
                                                "type": {"type": "string", "enum": ["event"]},
                                                "source": {"type": "string"},
                                                "start_node": {"type": "string"}
                                            },
                                            "required": ["type", "source", "start_node"],
                                            "additionalProperties": False
                                        }
                                    ]
                                }
                            },
                            # ============================================================
                            # 2️⃣ NODES with DYNAMIC TOOL PARAMS
                            # ============================================================
                            "nodes": {
                                "type": "array",
                                "description": (
                                    "Workflow nodes. CRITICAL: For agent_with_tools nodes, the 'params' field MUST contain "
                                    "parameter values extracted from the user request. DO NOT leave params empty!"
                                ),
                                "items": {
                                    "oneOf": self._build_node_schemas(tool_ids, tool_schemas)
                                }
                            },
                            # ============================================================
                            # 3️⃣ EDGES - DEPRECATED (Auto-generated from dependencies)
                            # ============================================================
                            # ⚠️ REMOVED: Edges field no longer exposed to LLM
                            # Edges are automatically inferred from:
                            # - $id references in plan descriptions (e.g., "Analyze $1 and $2")
                            # - Data flow in params (e.g., ${{from_step:node1.result}})
                            # This ensures parallel execution works correctly without manual edge generation
                    },
                "required": ["nodes", "triggers"],
                "additionalProperties": False
            }
        }
        
        return schema
    
    def _build_tool_param_hints(self, tool_ids: List[str], tool_schemas: Dict[str, Dict]) -> str:
        """
        Build schema-driven hints using ONLY metadata from tool schemas.
        NO HARDCODING - everything derived from schema descriptions, examples, constraints.
        
        Returns:
            String with tool descriptions, params, response fields - all from schemas
        """
        if not tool_ids:
            return ""
        
        # Get current date/time context dynamically
        now = datetime.now()
        context_lines = [
            f"📅 EXECUTION CONTEXT:",
            f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Use this context when extracting dates/times from user requests.",
            ""
        ]
        
        # Build tool-specific guidance from schemas
        if tool_schemas:
            tool_lines = ["📋 AVAILABLE TOOLS (schemas from Composio):"]
        else:
            tool_lines = [f"📋 AVAILABLE TOOLS: {', '.join(tool_ids[:15])} (dynamic schemas disabled)"]
            return "\n".join(context_lines + tool_lines)
        
        for tool_id in tool_ids[:15]:  # Up to 15 tools
            tool_schema = tool_schemas.get(tool_id)
            if not tool_schema:
                continue
            
            tool_lines.append(f"\n▶ {tool_id}")
            
            # Tool description (business rules from schema)
            tool_desc = tool_schema.get('description', '')
            if tool_desc:
                # Truncate long descriptions
                tool_desc_short = tool_desc[:200] + '...' if len(tool_desc) > 200 else tool_desc
                tool_lines.append(f"  📖 {tool_desc_short}")
            
            # Input parameters
            params_schema = tool_schema.get('parameters_schema', {})
            required = params_schema.get('required', [])
            properties = params_schema.get('properties', {})
            
            # Required params with their schema metadata
            if required:
                req_params = []
                for param in required[:6]:
                    param_info = properties.get(param, {})
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    examples = param_info.get('examples', [])
                    
                    # Build rich param description from schema
                    param_str = f"{param}: {param_type}"
                    if examples:
                        param_str += f" (e.g., {examples[0]})"
                    elif param_desc and len(param_desc) < 60:
                        param_str += f" ({param_desc})"
                    
                    req_params.append(param_str)
                
                tool_lines.append(f"  ✅ Required: {', '.join(req_params)}")
            
            # Key optional params (prioritized by schema metadata)
            optional_params = self._get_important_optional_params(properties, required)
            if optional_params:
                tool_lines.append(f"  ⚙️ Optional: {', '.join(optional_params[:4])}")
            
            # Response fields (what this tool returns)
            response_schema = tool_schema.get('response_schema', {})
            response_fields = self._extract_response_fields(response_schema)
            if response_fields:
                field_preview = ', '.join(response_fields[:8])
                tool_lines.append(f"  📤 Returns: {field_preview}")
        
        return "\n".join(context_lines + tool_lines)
    
    def _get_important_optional_params(self, properties: Dict[str, Any], required: List[str]) -> List[str]:
        """
        Identify important optional params using schema metadata (NO hardcoding).
        
        Uses signals from schema:
        - Has examples → provider documented it
        - Has default → important enough for default
        - Has validation constraints → important
        - Mentioned in description → cross-referenced
        """
        scored_params = []
        
        for param_name, param_schema in properties.items():
            if param_name in required:
                continue
            
            score = 0
            
            # Signal 1: Has examples (provider documented it)
            if param_schema.get('examples'):
                score += 10
            
            # Signal 2: Has non-null default
            if 'default' in param_schema and param_schema['default'] is not None:
                score += 8
            
            # Signal 3: Not nullable (more constrained)
            if param_schema.get('nullable') is False:
                score += 5
            
            # Signal 4: Has validation constraints
            if param_schema.get('enum') or param_schema.get('pattern'):
                score += 6
            
            # Signal 5: Long description (well-documented)
            desc_len = len(param_schema.get('description', ''))
            if desc_len > 100:
                score += 4
            elif desc_len > 50:
                score += 2
            
            if score > 0:
                param_type = param_schema.get('type', 'any')
                default = param_schema.get('default')
                examples = param_schema.get('examples', [])
                
                param_str = f"{param_name}: {param_type}"
                if default is not None:
                    param_str += f" (default: {default})"
                elif examples:
                    param_str += f" (e.g., {examples[0]})"
                
                scored_params.append((score, param_str))
        
        # Sort by score descending
        scored_params.sort(key=lambda x: x[0], reverse=True)
        return [p[1] for p in scored_params]
    
    def _extract_response_fields(self, response_schema: Dict[str, Any], prefix: str = "") -> List[str]:
        """
        Extract all available field paths from a response schema.
        
        Args:
            response_schema: JSON schema for tool response
            prefix: Current path prefix (for nested fields)
        
        Returns:
            List of field paths like ['data', 'data.results', 'error', 'successful']
        """
        fields = []
        
        if not isinstance(response_schema, dict):
            return fields
        
        properties = response_schema.get('properties', {})
        
        for field_name, field_schema in properties.items():
            full_path = f"{prefix}.{field_name}" if prefix else field_name
            fields.append(full_path)
            
            # Recurse for nested objects (but only 1 level deep to avoid bloat)
            if not prefix and field_schema.get('type') == 'object':
                nested_fields = self._extract_response_fields(field_schema, full_path)
                fields.extend(nested_fields[:5])  # Limit nested fields
        
        return fields
    
    def _build_node_schemas(self, tool_ids: List[str], tool_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        Build node schema definitions with tool-specific parameter schemas.
        
        For agent_with_tools nodes, the 'params' field gets dynamically injected
        with the actual tool parameter schema from Composio.
        
        Returns:
            List of node schema objects for the 'oneOf' array
        """
        node_schemas = []
        
        # ===================================================================
        # agent_with_tools - WITH DYNAMIC TOOL PARAMETER SCHEMAS
        # ===================================================================
        # Cross-reference: planner_prompt.yaml defines <agent_with_tools> marker
        # Cross-reference: code_generator_prompt.yaml explains tool execution patterns
        # Cross-reference: langgraph_adapter.py._build_composio_tool_node() handles execution
        # Build a description that includes ALL tool parameter schemas
        tool_param_descriptions = []
        for tool_id in tool_ids[:20]:  # Limit to 20 to avoid token explosion
            tool_schema = tool_schemas.get(tool_id)
            if not tool_schema:
                continue
            
            params_schema = tool_schema.get('parameters_schema', {})
            required = params_schema.get('required', [])
            
            if required:
                tool_param_descriptions.append(
                    f"- If tool_id='{tool_id}', params MUST include: {', '.join(required[:5])}"
                )
        
        # Get current date for extraction examples
        now = datetime.now()
        current_year = now.year
        today = now.strftime("%Y-%m-%d")
        
        params_description = (
            "🎯 CRITICAL: Extract parameter values from user request and populate this params object.\n\n"
            f"📅 CURRENT CONTEXT: Today is {today}. Use {current_year} for ambiguous dates.\n\n"
            "✅ UNIVERSAL EXTRACTION PATTERNS (learn the logic):\n\n"
            "Pattern 1 - Direct Value Extraction:\n"
            "  User mentions: 'send to user@example.com'\n"
            "  Tool has param: recipient_email (string)\n"
            "  → Extract: {{ recipient_email: 'user@example.com' }}\n\n"
            "Pattern 2 - Location/ID Extraction:\n"
            "  User mentions: 'from LocationA to LocationB'\n"
            "  Tool has params: source_id, destination_id\n"
            "  → Extract: {{ source_id: 'LocationA', destination_id: 'LocationB' }}\n\n"
            "Pattern 3 - Date Extraction:\n"
            "  User mentions: 'on Dec 25' or 'next Monday'\n"
            "  Tool has param: date (string, format: YYYY-MM-DD)\n"
            f"  → Extract: {{ date: '{current_year}-12-25' }}  # Always use current year!\n\n"
            "Pattern 4 - Quantity Extraction:\n"
            "  User mentions: '3 items' or 'for 5 people'\n"
            "  Tool has param: quantity (integer)\n"
            "  → Extract: {{ quantity: 3 }} or {{ count: 5 }}\n\n"
            "Pattern 5 - Search/Query Extraction:\n"
            "  User mentions: 'find X in Y'\n"
            "  Tool has params: query, location\n"
            "  → Extract: {{ query: 'X', location: 'Y' }}\n\n"
            "Pattern 6 - Reference Previous Node Output:\n"
            "  Tool node output: Use '${{node_id.response_field}}'\n"
            "    Example: {{ body: '${{search_flights.data}}' }}\n"
            "  Code node output: Use '${{node_id.variable_name}}'\n"
            "    Example: {{ body: '${{prepare_content.email_body}}' }}\n"
            "    (where 'email_body' is a variable created in the code)\n\n"
            "Pattern 7 - Generate Missing Values:\n"
            "  Tool requires 'subject' but user didn't provide\n"
            "  → Generate contextual value: {{ subject: 'Results from [task_name]' }}\n\n"
            "🔑 EXTRACTION RULES:\n"
            "1. Match user's words to parameter names/descriptions (semantic matching)\n"
            "2. Use parameter 'examples' from schema to understand expected format\n"
            "3. For dates: use YYYY-MM-DD format with current year if ambiguous\n"
            "4. For enums: pick the closest matching enum value from schema\n"
            "5. REQUIRED params (see tool requirements below): MUST have a value\n"
            "6. Optional params: include if user provides related information\n"
            "7. Previous node outputs: use '${{node_id.field}}' syntax\n"
            "8. Numbers: extract as integers or floats based on param type\n"
            "9. Booleans: extract from 'yes/no', 'true/false', 'enabled/disabled'\n"
            "10. NEVER leave required params null/empty/missing!\n\n"
        )
        
        if tool_param_descriptions:
            params_description += "Tool-specific requirements:\n" + "\n".join(tool_param_descriptions[:15])
        
        node_schemas.append({
            "type": "object",
            "description": "Tool execution node",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["agent_with_tools"]},
                "tool_id": {
                    "type": "string",
                    "enum": tool_ids,
                    "description": "MUST be from Available Composio Actions list"
                },
                "params": {
                    "type": "object",
                    "description": (
                        f"{params_description}\n\n"
                        "⚠️ STRUCTURE: This is a FLAT object of key-value pairs.\n"
                        "✅ Correct: {{ param1: 'value1', param2: 'value2' }}\n"
                        "❌ Wrong: {{ parameters: {{ param1: 'value1' }} }}  # Don't nest!\n"
                    ),
                    "additionalProperties": True  # Allow any params (tool-specific)
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "🎯 EXPLICIT DEPENDENCIES (Node IDs - HIGHEST PRIORITY!):\n"
                        "List of node IDs this node depends on.\n\n"
                        "✅ WHEN TO USE:\n"
                        "- node_2 processes data from node_1 → dependencies: ['node_1']\n"
                        "- node_4 needs results from node_2 & node_3 → dependencies: ['node_2', 'node_3']\n"
                        "- node_2_1 (nested) depends on loop node_2 → dependencies: ['node_2']\n"
                        "- node_2_2 depends on loop node_2 AND node_2_1 → dependencies: ['node_2', 'node_2_1']\n"
                        "- node_1 is independent (fetches new data) → dependencies: []\n\n"
                        "📊 WHY NODE IDs:\n"
                        "- Can reference ANY node (nested, parallel, etc.)\n"
                        "- Works with LangGraph/NetworkX out-of-the-box\n"
                        "- 100% deterministic execution order\n\n"
                        "⚠️ CRITICAL: Use exact node IDs like 'node_1', 'node_2', 'node_2_1'"
                    )
                },
                "description": {"type": "string", "description": "Clear description of what this node does (taken from plan step)"}
            },
            "required": ["id", "type", "tool_id", "description"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # code_execution - DEPRECATED: Use agent_only with agent_mode='code' instead
        # ===================================================================
        # This node type is redundant with agent_only (agent_mode='code')
        # Kept for backward compatibility only
        # Build code generation guidance from actual response schemas
        code_guidance = (
            "Python code to execute. Access previous nodes AND create variables for downstream use.\n\n"
            "🔑 RULE 1: Use ACTUAL Node IDs (not 'node_id' placeholder!):\n"
            "✅ CORRECT - If previous node ID is 'search_flights':\n"
            "  results = search_flights.data.results\n"
            "  email_body = f\"Found: {search_flights.data}\"\n\n"
            "❌ WRONG - Don't use generic 'node_id':\n"
            "  results = node_id.data.results         # Who is 'node_id'??\n"
            "  email_body = f\"{previous_node.data}\"  # Be specific!\n\n"
            "🔑 RULE 2: Create Descriptive Variables for Downstream Nodes:\n"
            "✅ CORRECT:\n"
            "  email_body = f\"Results: {search_flights.data}\"  # Clear variable name\n"
            "  formatted_text = process(fetch_api.data)         # Descriptive\n"
            "  error_message = validate.error                   # Explicit\n\n"
            "Downstream nodes can then use:\n"
            "  - '${{this_node_id.email_body}}'\n"
            "  - '${{this_node_id.formatted_text}}'\n"
            "  - '${{this_node_id.error_message}}'\n\n"
            "🔑 RULE 3: Check Response Schemas (see '📤 Returns:' above):\n"
            "- Tool nodes return: data, error, successful (from schema)\n"
            "- Code nodes return: variables you create in the code\n"
            "- NEVER invent field names not in schema!\n\n"
            "✅ COMPLETE EXAMPLE:\n"
            "Suppose previous node ID is 'search_flights' which returns {data, error, successful}:\n"
            "  flight_data = search_flights.data           # Access using ACTUAL ID\n"
            "  best_price = min(flight_data.results)       # Process the data\n"
            "  email_body = f\"Best price: ${best_price}\"  # Create variable for next node"
        )
        
        # SCHEMA REMOVED - use agent_only with agent_mode='code' instead
        # (code_execution type is no longer exposed to LLMs)
        
        # ===================================================================
        # agent_only - Pure AI processing (UNIFIED: reasoning + code)
        # ===================================================================
        # Cross-reference: planner_prompt.yaml defines <agent_only> marker
        # Cross-reference: code_generator_prompt.yaml explains when to use
        # Cross-reference: langgraph_adapter.py._build_agent_node() handles execution
        node_schemas.append({
            "type": "object",
            "description": (
                "Agent-only node for AI processing WITHOUT external tools. "
                "🎯 TWO MODES (pick ONE):\n"
                "1. REASONING (agent_mode='reasoning'): Pure LLM analysis - NO code field\n"
                "   - Use for: 'Analyze sentiment', 'Which is better?', 'Recommend option'\n"
                "   - LLM provides analysis at runtime\n"
                "   - ❌ DO NOT include 'code' field\n"
                "2. CODE (agent_mode='code'): Execute Python code\n"
                "   - Use for: 'Calculate average', 'Transform JSON', 'Parse data'\n"
                "   - ✅ MUST provide 'code' field with Python code\n"
                "   - For deterministic operations (math, data manipulation)\n\n"
                "⚠️ IMPORTANT: If agent_mode='reasoning', omit the 'code' field entirely!"
            ),
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["agent_only"]},
                "agent_mode": {
                    "type": "string",
                    "enum": ["reasoning", "code"],
                    "description": (
                        "⚠️ CRITICAL CHOICE:\n"
                        "'reasoning' = Pure LLM analysis (omit 'code' field)\n"
                        "'code' = Python execution (include 'code' field)"
                    )
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute.\n"
                        "⚠️ ONLY include if agent_mode='code'\n"
                        "❌ OMIT if agent_mode='reasoning'"
                    )
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "🎯 EXPLICIT DEPENDENCIES (Node IDs - HIGHEST PRIORITY!):\n"
                        "List of node IDs this node depends on.\n\n"
                        "✅ EXAMPLES:\n"
                        "- node_2 'Calculate from node_1' → dependencies: ['node_1']\n"
                        "- node_3 'Analyze node_2' → dependencies: ['node_2']\n"
                        "- node_2_1 (nested in loop) → dependencies: ['node_2']\n"
                        "- Independent analysis → dependencies: []\n\n"
                        "⚠️ CRITICAL FOR REASONING NODES:\n"
                        "Even if agent_mode='reasoning', if the analysis uses data from previous steps,\n"
                        "you MUST specify dependencies! This prevents disconnected workflow components.\n\n"
                        "📊 DETERMINISTIC > INFERENCE:\n"
                        "Explicit dependencies = 100% accuracy. Don't rely on text inference alone!"
                    )
                },
                "description": {"type": "string", "description": "Clear description of what this node does (taken from plan step)"}
            },
            "required": ["id", "type", "description"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # form - User input collection ONLY (NOT for approvals!)
        # ===================================================================
        # Build form guidance with downstream tool requirements
        form_field_requirements = []
        for tool_id in tool_ids[:10]:  # Check up to 10 tools
            tool_schema = tool_schemas.get(tool_id) if tool_schemas else None
            if not tool_schema:
                continue
            
            params_schema = tool_schema.get('parameters_schema', {})
            required_params = params_schema.get('required', [])
            
            if required_params:
                # Extract params that look like user inputs using schema signals (not hardcoded)
                # User input params typically have: examples, descriptions, no technical keywords
                properties = params_schema.get('properties', {})
                user_input_params = []
                
                for p in required_params[:10]:  # Limit to first 10
                    param_schema = properties.get(p, {})
                    param_lower = p.lower()
                    
                    # Schema-driven detection: has examples OR long description
                    has_examples = bool(param_schema.get('examples'))
                    has_description = len(param_schema.get('description', '')) > 30
                    
                    # Exclude technical/system params (IDs, tokens, keys, formats)
                    is_technical = any(kw in param_lower for kw in ['_id', 'token', 'key', 'api', 'auth', 'format', 'page_'])
                    
                    if (has_examples or has_description) and not is_technical:
                        user_input_params.append(p)
                
                if user_input_params:
                    form_field_requirements.append(
                        f"   • If using {tool_id}, form should collect: {', '.join(user_input_params[:3])}"
                    )
        
        form_guidance = (
            "Form node for collecting INITIAL data from users (e.g., search criteria, contact details, preferences). "
            "⚠️ DO NOT USE for approval/review/authorization steps - use 'hitl' type instead! "
            "Use 'form' only when: (1) Starting a workflow with user input, (2) Collecting data BEFORE any processing, "
            "(3) User is PROVIDING information (not reviewing/approving results).\n\n"
            "🎯 CRITICAL FORM DESIGN RULES:\n"
            "1. Analyze ALL downstream agent_with_tools nodes in your workflow\n"
            "2. Check what parameters those tools require (see tool requirements above)\n"
            "3. For any parameter that will be filled from form data (e.g., recipient_email, user_name), "
            "the form MUST include a corresponding field\n"
            "4. Form field 'name' should match the parameter name you'll reference later\n"
            "5. Example: If you plan to use '${{form_node.email}}' in a downstream node, "
            "the form MUST have a field with name='email'\n\n"
        )
        
        if form_field_requirements:
            form_guidance += "💡 Based on available tools, consider these fields:\n" + "\n".join(form_field_requirements[:5])
        
        node_schemas.append({
            "type": "object",
            "description": form_guidance,
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["form"]},
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "label": {"type": "string"},
                            "required": {"type": "boolean"}
                        },
                        "required": ["name", "type", "label"]
                    }
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Usually empty [] for forms (typically first step). Specify node IDs if form uses previous node data (e.g., ['node_1'])."
                }
            },
            "required": ["id", "type", "fields"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # hitl - Human-in-the-loop for ALL approvals/reviews
        # ===================================================================
        # Cross-reference: planner_prompt.yaml defines <hitl> marker
        # Cross-reference: code_generator_prompt.yaml explains approval patterns
        # Cross-reference: langgraph_adapter.py._build_form_node() handles execution
        node_schemas.append({
            "type": "object",
            "description": (
                "Human-in-the-Loop (HITL) node for ALL approval, review, authorization, validation, or decision steps. "
                "✅ ALWAYS USE 'hitl' (NOT 'form') when: (1) Reviewing/approving results from previous steps, "
                "(2) Authorizing an action (e.g., 'approve flight', 'authorize payment'), "
                "(3) Making decisions based on processed data, (4) Validating/confirming proposed actions. "
                "Examples: approve_flight_booking, review_draft_email, authorize_expense, validate_proposal. "
                "⚠️ CRITICAL: If the step involves 'approve', 'review', 'authorize', 'validate', 'confirm' → USE 'hitl' NOT 'form'!"
            ),
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["hitl"]},
                "approval_type": {
                    "type": "string",
                    "enum": ["approve_reject", "edit_approve"]
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node IDs whose results need approval (e.g., ['node_2', 'node_3']). HITL nodes typically depend on previous processing steps."
                }
            },
            "required": ["id", "type", "approval_type"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # loop - Iteration/Loop execution (CRITICAL for token savings!)
        # ===================================================================
        # Cross-reference: planner_prompt.yaml defines <loop> marker with sub-steps (e.g., 2.1, 2.2)
        # Cross-reference: code_generator_prompt.yaml explains loop patterns and loop_body
        # Cross-reference: langgraph_adapter.py.loop_executor handles execution
        node_schemas.append({
            "type": "object",
            "description": (
                "Loop node for iterating over arrays/collections. "
                "🎯 USE THIS when same operation repeats 3+ times with different inputs. "
                "💡 BENEFITS: 90% token savings vs separate nodes! "
                "Example: 'Fetch from multiple platforms' → ONE loop node (not 4 separate nodes). "
                "\n\n"
                "WHEN TO USE:\n"
                "- Same tool_id repeated multiple times\n"
                "- User says 'fetch from A, B, C, D' or 'send to A, B, C'\n"
                "- Multiple items with identical operation\n"
                "\n"
                "HOW IT WORKS:\n"
                "1. Loop iterates over array (e.g., ['item_a', 'item_b', 'item_c'])\n"
                "2. For each item, executes child nodes in loop_body\n"
                "3. Results auto-aggregated and available via '${{loop_id.results}}'\n"
                "\n"
                "✅ LANGGRAPH-NATIVE EXAMPLES:\n\n"
                "**Simple (auto-extraction):**\n"
                "{\n"
                "  'id': 'process_loop',\n"
                "  'type': 'loop',\n"
                "  'loop_over': 'step_1_output',  # Auto-extracts array from response\n"
                "  'loop_body': ['process_item'],\n"
                "  'dependencies': [1]\n"
                "}\n\n"
                "**Explicit path (nested data):**\n"
                "{\n"
                "  'id': 'data_loop',\n"
                "  'type': 'loop',\n"
                "  'loop_over': 'node_1.items',  # Direct path to nested array field\n"
                "  'loop_body': ['process_item'],\n"
                "  'dependencies': [1]\n"
                "}\n\n"
                "Loop body accesses current item via '${{loop_item}}'."
            ),
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["loop"]},
                "loop_over": {
                    "type": "string",
                    "description": (
                        "✅ CRITICAL - Specify the array to iterate over (LangGraph-native):\n\n"
                        "**SIMPLE (recommended):**\n"
                        "- 'step_N_output' (e.g., 'step_1_output') - auto-extracts array from response\n"
                        "- 'node_N' (e.g., 'node_1') - auto-extracts array from response\n\n"
                        "**EXPLICIT (for nested data):**\n"
                        "- 'node_1.items' - for array nested in 'items' field\n"
                        "- 'step_2_output.data' - for API responses with data wrapper\n"
                        "- 'node_3.results' - for lists nested in 'results' field\n\n"
                        "⚠️ Use step/node numbers, NOT node IDs!\n"
                        "✅ If dependencies=[1], use 'step_1_output' or 'node_1.messages'"
                    )
                },
                "loop_body": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of node IDs to execute for each iteration. "
                        "These nodes must be defined elsewhere in the workflow. "
                        "Use '${{loop_item}}' in child node params to access current iteration value."
                    )
                },
                "exit_condition": {
                    "type": "string",
                    "description": (
                        "Optional condition to break loop early. "
                        "Example: '${{loop_item.status}} == \"complete\"'"
                    )
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node IDs that must complete before loop starts (e.g., ['node_1'] - the node that fetches the array to loop over)."
                },
                "description": {"type": "string", "description": "Clear description of what this loop does (taken from plan step)"}
            },
            "required": ["id", "type", "loop_over", "loop_body", "description"],
            "additionalProperties": False
        })
        
        logger.info(f"[SCHEMA_BUILDER] Built {len(node_schemas)} node schema variants")
        return node_schemas


# ===================================================================
# Singleton instance for reuse
# ===================================================================
_schema_builder_instance: Optional[WorkflowSchemaBuilder] = None


def get_schema_builder() -> WorkflowSchemaBuilder:
    """Get or create singleton schema builder instance."""
    global _schema_builder_instance
    if _schema_builder_instance is None:
        _schema_builder_instance = WorkflowSchemaBuilder()
    return _schema_builder_instance
