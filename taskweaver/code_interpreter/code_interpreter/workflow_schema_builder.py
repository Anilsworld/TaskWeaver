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
        """Initialize builder - uses shared ToolSchemaCache singleton."""
        # Import here to avoid circular dependency
        from taskweaver.code_interpreter.code_interpreter.dynamic_model_validator import ToolSchemaCache
        
        self.schema_cache = ToolSchemaCache()
        # Ensure schemas are loaded
        self.schema_cache.load_schemas()
    
    def _get_tool_schemas(self, tool_ids: List[str]) -> Dict[str, Dict]:
        """
        Get schemas for specific tools using shared cache.
        
        Args:
            tool_ids: List of tool IDs to get schemas for
        
        Returns:
            Dict mapping tool_id → tool schema
        """
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
                "⚠️ SELF-CHECK Before Returning Workflow:\n"
                "Ask yourself:\n"
                "  ✅ Did I read the user's request and extract ALL relevant values?\n"
                "  ✅ Does every agent_with_tools node have a populated params field?\n"
                "  ✅ Did I check each tool's REQUIRED params and provide values?\n"
                "  ✅ Are dates in YYYY-MM-DD format with current year?\n"
                "  ✅ Did I use parameter examples from schemas as formatting guides?\n"
                "  ✅ Did I generate sensible defaults for missing but required fields?\n\n"
                "If ANY answer is NO, you MUST fix it before returning!\n\n"
                "🔗 WORKFLOW CONNECTIVITY:\n"
                "- Start nodes: depends_on = []\n"
                "- Other nodes: depends_on = [prerequisite node IDs]\n"
                "- Missing depends_on causes workflow failure!\n\n"
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
                            "4. All nodes (except start) have 'depends_on' field\n\n"
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
                                                "type": {"const": "schedule"},
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
                                                "type": {"const": "event"},
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
                                                "type": {"const": "sequential"},
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
                                                "type": {"const": "conditional"},
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
        tool_lines = ["📋 AVAILABLE TOOLS (schemas from Composio):"]
        
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
                "type": {"const": "agent_with_tools"},
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
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CRITICAL: List of node IDs this node depends on. "
                        "Empty array [] ONLY for start nodes. "
                        "All other nodes MUST list their dependencies. "
                        "Example: ['search_flights'] means this node runs AFTER search_flights completes."
                    )
                },
                "description": {"type": "string"}
            },
            "required": ["id", "type", "tool_id", "depends_on"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # code_execution - Python code node
        # ===================================================================
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
        
        node_schemas.append({
            "type": "object",
            "description": "Python code execution node",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": "code_execution"},
                "code": {
                    "type": "string",
                    "description": code_guidance
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CRITICAL: List of node IDs this node depends on. "
                        "Empty array [] ONLY for start nodes. "
                        "All other nodes MUST list their dependencies."
                    )
                },
                "description": {"type": "string"}
            },
            "required": ["id", "type", "code", "depends_on"],
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
                "type": {"const": "agent_only"},
                "code": {"type": "string"},
                "depends_on": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"}
            },
            "required": ["id", "type", "code"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # form - User input collection
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": "Form node for collecting user input",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": "form"},
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
                "depends_on": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["id", "type", "fields"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # hitl - Human-in-the-loop approval
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": "Human approval/review node",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": "hitl"},
                "approval_type": {
                    "type": "string",
                    "enum": ["approve_reject", "edit_approve"]
                },
                "depends_on": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["id", "type", "approval_type"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # loop - Iteration over arrays
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": "Loop node for iterating over arrays",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": "loop"},
                "iterate_over": {
                    "type": "string",
                    "description": "Placeholder for array (e.g., '${{items}}')"
                },
                "loop_body": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node IDs to execute in loop"
                },
                "depends_on": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["id", "type", "iterate_over", "loop_body"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # parallel - Parallel execution group
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": "Parallel execution node",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": "parallel"},
                "parallel_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node IDs to execute in parallel"
                },
                "depends_on": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["id", "type", "parallel_nodes"],
            "additionalProperties": False
        })
        
        # ===================================================================
        # sub_workflow - Sub-workflow invocation
        # ===================================================================
        node_schemas.append({
            "type": "object",
            "description": "Sub-workflow invocation node",
            "properties": {
                "id": {"type": "string"},
                "type": {"const": "sub_workflow"},
                "workflow_id": {"type": "string"},
                "inputs": {"type": "object"},
                "depends_on": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["id", "type", "workflow_id"],
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
