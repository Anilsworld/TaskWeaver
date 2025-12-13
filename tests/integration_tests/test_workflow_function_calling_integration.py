"""
Integration tests for Workflow Function Calling with REAL OpenAI API calls.

⚠️ REQUIREMENTS:
1. Set OPENAI_API_KEY environment variable
2. Set RUN_INTEGRATION_TESTS=true to enable these tests
3. These tests will consume OpenAI API tokens

Usage:
    # Run integration tests
    set RUN_INTEGRATION_TESTS=true
    set OPENAI_API_KEY=your_key_here
    pytest tests/integration_tests/test_workflow_function_calling_integration.py -v
"""

import json
import os
import pytest

# Skip all tests if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable."
)


class TestOpenAIFunctionCallingIntegration:
    """Integration tests with real OpenAI API."""
    
    def setup_method(self):
        """Check API key is available."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        # Get API config from environment or use defaults
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06")
        
        print(f"\n🔑 Using API: {self.api_base}")
        print(f"📦 Using Model: {self.model}")
    
    def test_real_openai_function_calling(self):
        """Test actual OpenAI function calling with simple workflow."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Simple function schema
        function_schema = {
            "name": "create_workflow_ir",
            "description": "Create a simple workflow",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "properties": {
                            "nodes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "type": {"type": "string", "enum": ["agent_with_tools"]},
                                        "tool_id": {
                                            "type": "string",
                                            "enum": ["GMAIL_GET_MAIL_V2", "SLACKBOT_SEND_MESSAGE"]
                                        }
                                    },
                                    "required": ["id", "type", "tool_id"]
                                }
                            },
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"const": "sequential"},
                                        "from": {"type": "string"},
                                        "to": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["nodes", "edges"]
                    }
                },
                "required": ["workflow"]
            }
        }
        
        # Call OpenAI
        print("\n📡 Calling OpenAI API...")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a workflow compiler. Generate workflows via function calls."
                },
                {
                    "role": "user",
                    "content": (
                        "Generate a workflow that:\n"
                        "1. Fetches emails from Gmail\n"
                        "2. Sends a summary to Slack\n\n"
                        "Use these tools:\n"
                        "- GMAIL_GET_MAIL_V2: Fetch emails from Gmail\n"
                        "- SLACKBOT_SEND_MESSAGE: Send message to Slack"
                    )
                }
            ],
            tools=[{"type": "function", "function": function_schema}],
            tool_choice={"type": "function", "function": {"name": "create_workflow_ir"}},
            temperature=0.0
        )
        
        # Extract result
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        print(f"\n✅ Received {len(tool_calls)} tool call(s)")
        
        # Assertions
        assert tool_calls is not None, "No tool calls returned"
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        
        tool_call = tool_calls[0]
        assert tool_call.function.name == "create_workflow_ir"
        
        # Parse arguments
        arguments = json.loads(tool_call.function.arguments)
        print(f"\n📋 Generated workflow:\n{json.dumps(arguments, indent=2)}")
        
        # Validate structure
        assert "workflow" in arguments
        workflow = arguments["workflow"]
        assert "nodes" in workflow
        assert "edges" in workflow
        assert len(workflow["nodes"]) >= 2, "Should have at least 2 nodes"
        
        # Validate tool_ids are from enum
        tool_ids = [node["tool_id"] for node in workflow["nodes"]]
        valid_tools = {"GMAIL_GET_MAIL_V2", "SLACKBOT_SEND_MESSAGE"}
        for tool_id in tool_ids:
            assert tool_id in valid_tools, f"Invalid tool_id: {tool_id}"
        
        print(f"\n✅ Test passed! Generated {len(workflow['nodes'])} nodes, {len(workflow['edges'])} edges")


class TestCodeGeneratorFunctionCallingIntegration:
    """Integration tests for full CodeGenerator with real OpenAI."""
    
    def setup_method(self):
        """Setup CodeGenerator with real API configuration."""
        from injector import Injector
        from taskweaver.config.config_mgt import AppConfigSource
        from taskweaver.logging import LoggingModule
        from taskweaver.memory.plugin import PluginModule
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06")
        
        print(f"\n🔑 Using API: {self.api_base}")
        print(f"📦 Using Model: {self.model}")
        
        # Create injector with real config
        app_injector = Injector([PluginModule, LoggingModule])
        app_config = AppConfigSource(
            config={
                "app_dir": os.path.dirname(os.path.abspath(__file__)),
                "llm.api_type": "openai",
                "llm.api_key": self.api_key,
                "llm.api_base": self.api_base,
                "llm.model": self.model,
                "code_generator.use_function_calling": True,
            },
        )
        app_injector.binder.bind(AppConfigSource, to=app_config)
        
        from taskweaver.code_interpreter.code_interpreter import CodeGenerator
        self.code_generator = app_injector.create_object(CodeGenerator)
    
    def test_extract_tool_ids_with_real_selector_output(self):
        """Test tool ID extraction with realistic selector output."""
        # Simulate realistic output from composio_action_selector
        composio_actions = """
Available Composio Actions (Top 5):

- GMAIL_GET_MAIL_V2: Get emails from Gmail with advanced filtering
  Required params: max_results (int)
  Optional params: query (str), label_ids (list)
  Response: list of email objects with id, subject, body, from, to

- GMAIL_SEND_EMAIL_V2: Send email via Gmail
  Required params: to (str), subject (str), body (str)
  Optional params: cc (list), bcc (list), attachments (list)
  Response: message object with id, thread_id

- SLACKBOT_SEND_MESSAGE: Send message to Slack channel
  Required params: channel (str), text (str)
  Optional params: attachments (list), thread_ts (str)
  Response: message object with ts, channel

- SLACKBOT_CREATE_CHANNEL: Create new Slack channel
  Required params: name (str)
  Optional params: is_private (bool), description (str)
  Response: channel object with id, name

- OUTLOOK_SEND_EMAIL: Send email via Outlook
  Required params: to (str), subject (str), body (str)
  Optional params: cc (list), importance (str)
  Response: message object with id
"""
        
        tool_ids = self.code_generator._extract_tool_ids_from_actions(composio_actions)
        
        print(f"\n✅ Extracted {len(tool_ids)} tool IDs: {tool_ids}")
        
        assert len(tool_ids) == 5
        assert "GMAIL_GET_MAIL_V2" in tool_ids
        assert "GMAIL_SEND_EMAIL_V2" in tool_ids
        assert "SLACKBOT_SEND_MESSAGE" in tool_ids
        assert "SLACKBOT_CREATE_CHANNEL" in tool_ids
        assert "OUTLOOK_SEND_EMAIL" in tool_ids
    
    def test_build_schema_with_realistic_tools(self):
        """Test schema building with realistic tool list."""
        tool_ids = [
            "GMAIL_GET_MAIL_V2",
            "GMAIL_SEND_EMAIL_V2",
            "SLACKBOT_SEND_MESSAGE",
            "OUTLOOK_SEND_EMAIL",
            "HUBSPOT_CREATE_CONTACT"
        ]
        
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        print(f"\n✅ Built schema with {len(tool_ids)} tools")
        
        # Validate schema structure
        assert schema["name"] == "create_workflow_ir"
        
        # Check tool_id enum has all tools
        nodes_schema = schema["parameters"]["properties"]["workflow"]["properties"]["nodes"]
        agent_with_tools = nodes_schema["items"]["oneOf"][0]
        enum_tools = agent_with_tools["properties"]["tool_id"]["enum"]
        
        assert set(enum_tools) == set(tool_ids)
        print(f"✅ Schema enum validated: {len(enum_tools)} tools")
    
    def test_validate_realistic_workflow(self):
        """Test validation with realistic multi-node workflow."""
        workflow = {
            "nodes": [
                {
                    "id": "fetch_gmail",
                    "type": "agent_with_tools",
                    "tool_id": "GMAIL_GET_MAIL_V2",
                    "params": {
                        "max_results": 10,
                        "query": "is:unread"
                    },
                    "description": "Fetch unread emails"
                },
                {
                    "id": "approval_form",
                    "type": "hitl",
                    "approval_type": "approve_reject",
                    "depends_on": ["fetch_gmail"]
                },
                {
                    "id": "send_to_slack",
                    "type": "agent_with_tools",
                    "tool_id": "SLACKBOT_SEND_MESSAGE",
                    "params": {
                        "channel": "#general",
                        "text": "${{from_step:fetch_gmail.summary}}"
                    },
                    "depends_on": ["approval_form"]
                }
            ],
            "edges": [
                {
                    "type": "sequential",
                    "from": "fetch_gmail",
                    "to": "approval_form"
                },
                {
                    "type": "conditional",
                    "source": "approval_form",
                    "condition": "${{approved}} == True",
                    "if_true": "send_to_slack",
                    "if_false": "END"
                }
            ]
        }
        
        is_valid = self.code_generator._validate_workflow_ir(workflow)
        
        print(f"\n✅ Workflow validation: {'PASSED' if is_valid else 'FAILED'}")
        print(f"   - Nodes: {len(workflow['nodes'])}")
        print(f"   - Edges: {len(workflow['edges'])}")
        
        assert is_valid is True
    
    def test_convert_realistic_workflow_to_python(self):
        """Test Python conversion with realistic workflow."""
        workflow = {
            "nodes": [
                {
                    "id": "fetch_emails",
                    "type": "agent_with_tools",
                    "tool_id": "GMAIL_GET_MAIL_V2",
                    "params": {"max_results": 50}
                },
                {
                    "id": "process_data",
                    "type": "agent_only",
                    "code": "summary = '\\n'.join([e['subject'] for e in emails])"
                },
                {
                    "id": "send_summary",
                    "type": "agent_with_tools",
                    "tool_id": "SLACKBOT_SEND_MESSAGE",
                    "params": {
                        "channel": "#notifications",
                        "text": "${{from_step:process_data.summary}}"
                    }
                }
            ],
            "edges": [
                {"type": "sequential", "from": "fetch_emails", "to": "process_data"},
                {"type": "sequential", "from": "process_data", "to": "send_summary"}
            ],
            "triggers": [
                {
                    "type": "schedule",
                    "cron": "0 9 * * MON",
                    "start_node": "fetch_emails"
                }
            ]
        }
        
        python_code = self.code_generator._convert_workflow_json_to_python(workflow)
        
        print(f"\n✅ Generated Python code ({len(python_code)} chars):")
        print(f"\n{python_code[:500]}...")
        
        # Validate it's valid Python
        local_vars = {}
        exec(python_code, {}, local_vars)
        
        assert "result" in local_vars
        assert len(local_vars["result"]["nodes"]) == 3
        assert len(local_vars["result"]["edges"]) == 2
        assert len(local_vars["result"]["triggers"]) == 1


class TestEndToEndFunctionCallingFlow:
    """End-to-end test of full function calling flow."""
    
    def setup_method(self):
        """Setup."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06")
        
        print(f"\n🚀 END-TO-END TEST")
        print(f"🔑 API: {self.api_base}")
        print(f"📦 Model: {self.model}")
    
    @pytest.mark.slow
    def test_full_workflow_generation_flow(self):
        """
        Complete end-to-end test:
        1. Extract tool IDs from realistic action list
        2. Build function schema
        3. Call real OpenAI API
        4. Validate returned workflow
        5. Convert to Python
        """
        from injector import Injector
        from taskweaver.config.config_mgt import AppConfigSource
        from taskweaver.logging import LoggingModule
        from taskweaver.memory.plugin import PluginModule
        
        # Setup CodeGenerator
        app_injector = Injector([PluginModule, LoggingModule])
        app_config = AppConfigSource(
            config={
                "app_dir": os.path.dirname(os.path.abspath(__file__)),
                "llm.api_type": "openai",
                "llm.api_key": self.api_key,
                "llm.api_base": self.api_base,
                "llm.model": self.model,
                "code_generator.use_function_calling": True,
            },
        )
        app_injector.binder.bind(AppConfigSource, to=app_config)
        
        from taskweaver.code_interpreter.code_interpreter import CodeGenerator
        code_generator = app_injector.create_object(CodeGenerator)
        
        # Step 1: Realistic action list
        composio_actions = """
Available Composio Actions:
- GMAIL_GET_MAIL_V2: Fetch emails from Gmail
- SLACKBOT_SEND_MESSAGE: Send message to Slack
"""
        
        print("\n📋 Step 1: Extracting tool IDs...")
        tool_ids = code_generator._extract_tool_ids_from_actions(composio_actions)
        print(f"   ✅ Extracted: {tool_ids}")
        
        # Step 2: Build schema
        print("\n🔧 Step 2: Building function schema...")
        function_schema = code_generator._build_workflow_function_schema(tool_ids)
        print(f"   ✅ Schema built with {len(tool_ids)} tools in enum")
        
        # Step 3: Call OpenAI
        print("\n📡 Step 3: Calling OpenAI API...")
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a workflow compiler. Generate workflows via function calls ONLY. "
                        "Use ONLY tool_id values from the Available Composio Actions list. "
                        "All placeholders MUST use ${{...}} syntax."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"{composio_actions}\n\n"
                        "Generate complete workflow for: Check Gmail and send summary to Slack"
                    )
                }
            ],
            tools=[{"type": "function", "function": function_schema}],
            tool_choice={"type": "function", "function": {"name": "create_workflow_ir"}},
            temperature=0.0
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        workflow_json = json.loads(tool_call.function.arguments)["workflow"]
        print(f"   ✅ Received workflow with {len(workflow_json['nodes'])} nodes")
        
        # Step 4: Validate
        print("\n✅ Step 4: Validating workflow IR...")
        is_valid = code_generator._validate_workflow_ir(workflow_json)
        assert is_valid, "Workflow validation failed"
        print(f"   ✅ Validation passed")
        
        # Step 5: Convert to Python
        print("\n🐍 Step 5: Converting to Python...")
        python_code = code_generator._convert_workflow_json_to_python(workflow_json)
        print(f"   ✅ Generated {len(python_code)} chars of Python code")
        
        # Step 6: Verify Python is executable
        print("\n🔍 Step 6: Verifying Python code...")
        local_vars = {}
        exec(python_code, {}, local_vars)
        assert "result" in local_vars
        print(f"   ✅ Python code executed successfully")
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 END-TO-END TEST PASSED!")
        print("="*60)
        print(f"Generated workflow:")
        print(f"  - Nodes: {len(workflow_json['nodes'])}")
        print(f"  - Edges: {len(workflow_json['edges'])}")
        print(f"  - Tool IDs validated via enum: {tool_ids}")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
