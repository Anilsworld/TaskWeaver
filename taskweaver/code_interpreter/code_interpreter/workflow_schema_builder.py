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
                "  ✅ Did I use 'hitl' (NOT 'form') for ALL approval/review steps?\n"
                "  ✅ If I used ${{@variable}}, did I add a code_execution node that creates it?\n"
                "  ✅ Are ALL my placeholders in correct format: ${{node_id.field}} or ${{@variable}}?\n"
                "     ❌ WRONG: ${{user_email}} (missing dot or @ symbol)\n"
                "     ✅ RIGHT: ${{collect_form.user_email}} OR ${{@user_email}}\n"
                "  ✅ Do all my placeholders (${{node_id.field}}) reference actual node IDs and fields?\n"
                "  ✅ Does my form collect all data needed by downstream tool parameters?\n\n"
                "If ANY answer is NO, you MUST fix it before returning!\n\n"
                "🔗 WORKFLOW CONNECTIVITY (CRITICAL!):\n"
                "- ALL workflows MUST have 'edges' array to connect nodes\n"
                "- For N nodes, you need at least (N-1) edges\n"
                "- Edge format: {'type': 'sequential', 'from': 'node_a', 'to': 'node_b'}\n"
                "- Start nodes: No incoming edges\n"
                "- End nodes: No outgoing edges\n"
                "- Missing edges = disconnected workflow = FAILURE!\n\n"
                "📤 CODE NODES:\n"
                "- Use ONLY response fields from tool schemas (see 'Returns:' above)\n"
                "- Access: node_id.field_name\n"
                "- Never invent field names!"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "description": (
                            "Workflow Intermediate Representation (IR)\n\n"
                            "⚠️ BEFORE RETURNING THIS WORKFLOW, VERIFY:\n"
                            "1. Every agent_with_tools node HAS a 'params' field with extracted values\n"
                            "2. All REQUIRED parameters (see tool schemas above) have non-null values\n"
                            "3. Dates are in YYYY-MM-DD format with current year\n"
                            "4. All nodes are connected via 'edges' array following the Planner's sequence\n\n"
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
                            # 3️⃣ EDGES
                            # ============================================================
                            "edges": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {
                                            "type": "object",
                                            "description": "Sequential edge",
                                            "properties": {
                                                "type": {"type": "string", "enum": ["sequential"]},
                                                "from": {"type": "string"},
                                                "to": {"type": "string"}
                                            },
                                            "required": ["type", "from", "to"],
                                            "additionalProperties": False
                                        },
                                        {
                                            "type": "object",
                                            "description": "Conditional edge",
                                            "properties": {
                                                "type": {"type": "string", "enum": ["conditional"]},
                                                "from": {"type": "string"},
                                                "to": {"type": "string"},
                                                "condition": {"type": "string"}
                                            },
                                            "required": ["type", "from", "to", "condition"],
                                            "additionalProperties": False
                                        }
                                    ]
                                }
                            }
                        },
                        "required": ["nodes", "edges", "triggers"],
                        "additionalProperties": False
                    }
                },
                "required": ["workflow"],
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
            "  A) Tool node outputs (explicit node ID required):\n"
            "     Use: '${{node_id.response_field}}'\n"
            "     Example: {{ location_id: '${{search_data.data.id}}' }}\n\n"
            "  B) Code node variables (auto-resolution with @ prefix):\n"
            "     Use: '${{@variable_name}}' (@ prefix = system finds it automatically)\n"
            "     Example: {{ message: '${{@formatted_text}}', title: '${{@report_title}}' }}\n"
            "     System searches backward through code_execution nodes automatically\n\n"
            "  ⚠️ CRITICAL: For code variables, ALWAYS use @ prefix!\n"
            "     ✅ CORRECT: '${{@formatted_data}}', '${{@report_title}}', '${{@summary}}'\n"
            "     ❌ WRONG: '${{this_node_id.formatted_data}}', '${{code_node.summary}}'\n"
            "     (Don't use generic placeholders - use @ for auto-resolution!)\n\n"
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
                        "❌ Wrong: {{ parameters: {{ param1: 'value1' }} }}  # Don't nest!\n\n"
                        "⚠️ PLACEHOLDER FORMAT (if using placeholders):\n"
                        "✅ Valid: ${{{{node_id.field}}}} or ${{{{@variable}}}}\n"
                        "❌ Invalid: ${{{{field}}}} (missing node_id or @ symbol)\n"
                        "Examples:\n"
                        "  ✅ ${{{{collect_form.email}}}}\n"
                        "  ✅ ${{{{@user_data}}}}\n"
                        "  ❌ ${{{{email}}}} ← WRONG! Use ${{{{collect_form.email}}}} instead\n"
                    ),
                    "additionalProperties": True  # Allow any params (tool-specific)
                },
                "description": {"type": "string"}
            },
            "required": ["id", "type", "tool_id"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # code_execution - Python code node
        # ===================================================================
        # Build code generation guidance from actual response schemas
        code_guidance = (
            "Python code to execute. Access previous nodes AND create variables for downstream use.\n\n"
            "🔑 RULE 1: Use ACTUAL Node IDs (not 'node_id' placeholder!):\n"
            "✅ CORRECT - If previous node ID is 'fetch_data':\n"
            "  results = fetch_data.data.results\n"
            "  report_text = f\"Found: {fetch_data.data}\"\n\n"
            "❌ WRONG - Don't use generic 'node_id':\n"
            "  results = node_id.data.results           # Who is 'node_id'??\n"
            "  report_text = f\"{previous_node.data}\"  # Be specific!\n\n"
            "🔑 RULE 2: Create Descriptive Variables for Downstream Nodes:\n"
            "✅ CORRECT:\n"
            "  report_summary = f\"Results: {search_data.data}\"  # Clear variable name\n"
            "  formatted_text = process(fetch_api.data)          # Descriptive\n"
            "  status_message = validate.status                  # Explicit\n\n"
            "Variables created in this code can be referenced by downstream nodes using:\n"
            "  - '${{@report_summary}}' (system finds it automatically - no node ID needed!)\n"
            "  - '${{@formatted_text}}' (backward search through code nodes)\n"
            "  - '${{@status_message}}' (most recent definition wins)\n\n"
            "⚠️ Use '@' prefix for ALL code node variables! No node IDs needed.\n"
            "✅ CORRECT: '${{@variable_name}}' (auto-resolved by system)\n"
            "❌ WRONG: '${{node_id.variable_name}}' (fragile, error-prone)\n\n"
            "🔑 RULE 3: Check Response Schemas (see '📤 Returns:' above):\n"
            "- Tool nodes return: data, error, successful (from schema)\n"
            "- Code nodes return: variables you create in the code\n"
            "- NEVER invent field names not in schema!\n\n"
            "✅ COMPLETE EXAMPLE:\n"
            "Suppose previous node ID is 'fetch_data' which returns {data, error, successful}:\n"
            "  api_results = fetch_data.data              # Access using ACTUAL ID\n"
            "  top_result = api_results.items[0]          # Process the data\n"
            "  summary_text = f\"Result: {top_result}\"    # Create variable for next node"
        )
        
        node_schemas.append({
            "type": "object",
            "description": "Python code execution node",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["code_execution"]},
                "code": {
                    "type": "string",
                    "description": code_guidance
                },
                "description": {"type": "string"}
            },
            "required": ["id", "type", "code"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # agent_only - Deprecated, but keep for compatibility
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": "Agent-only node (deprecated, use code_execution)",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["agent_only"]},
                "code": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["id", "type", "code"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # form - User input collection ONLY (NOT for approvals!)
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": (
                "Form node for collecting INITIAL data from users (e.g., search criteria, contact details, preferences). "
                "⚠️ DO NOT USE for approval/review/authorization steps - use 'hitl' type instead! "
                "Use 'form' only when: (1) Starting a workflow with user input, (2) Collecting data BEFORE any processing, "
                "(3) User is PROVIDING information (not reviewing/approving results)."
            ),
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
                }
            },
            "required": ["id", "type", "fields"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # hitl - Human-in-the-loop for ALL approvals/reviews
        # ===================================================================
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
                }
            },
            "required": ["id", "type", "approval_type"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # parallel - Parallel execution group
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": (
                "Parallel execution node - executes multiple nodes simultaneously. "
                "Use this when multiple operations can run independently (e.g., fetching from multiple APIs). "
                "Example: Fetch emails from Gmail, Outlook, Instagram, and Facebook at the same time."
            ),
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["parallel"]},
                "parallel_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of nodes to execute in parallel (these nodes must be defined elsewhere in the workflow)"
                },
                "description": {"type": "string"}
            },
            "required": ["id", "type", "parallel_nodes"],
            "additionalProperties": False
        })
        
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
