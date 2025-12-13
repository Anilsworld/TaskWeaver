"""
Unit tests for Workflow Function Calling Implementation.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch
from injector import Injector

from taskweaver.config.config_mgt import AppConfigSource
from taskweaver.logging import LoggingModule
from taskweaver.memory.plugin import PluginModule


class TestChatCompletionWithFunctionCalling:
    """Test suite for chat_completion_with_function_calling() method."""
    
    def test_successful_function_call(self):
        """Test successful function calling with valid response."""
        from taskweaver.llm.openai import OpenAIService, OpenAIServiceConfig
        
        # Setup mock response
        mock_tool_call = Mock()
        mock_tool_call.function.name = "create_workflow_ir"
        mock_tool_call.function.arguments = json.dumps({
            "workflow": {
                "nodes": [{"id": "test", "type": "agent_with_tools", "tool_id": "TEST_TOOL"}],
                "edges": []
            }
        })
        
        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        # Make choices subscriptable
        mock_response.choices.__getitem__ = Mock(return_value=mock_choice)
        
        # Create mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create service
        config = Mock(spec=OpenAIServiceConfig)
        config.model = "gpt-4"
        config.max_tokens = 4000
        config.frequency_penalty = 0
        config.presence_penalty = 0
        config.support_system_role = True
        config.require_alternative_roles = False
        
        service = OpenAIService.__new__(OpenAIService)
        service.config = config
        service._client = mock_client
        
        # Execute
        result = service.chat_completion_with_function_calling(
            messages=[{"role": "user", "content": "Generate workflow"}],
            functions=[{"name": "create_workflow_ir", "parameters": {}}],
            function_call={"name": "create_workflow_ir"}
        )
        
        # Assertions
        assert result["function_name"] == "create_workflow_ir"
        assert "workflow" in result["arguments"]
        assert len(result["arguments"]["workflow"]["nodes"]) == 1
    
    def test_invalid_json_in_function_call(self):
        """Test error handling when function returns invalid JSON."""
        from taskweaver.llm.openai import OpenAIService, OpenAIServiceConfig
        
        # Setup mock response with invalid JSON
        mock_tool_call = Mock()
        mock_tool_call.function.name = "create_workflow_ir"
        mock_tool_call.function.arguments = "{invalid json"
        
        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call]
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices.__getitem__ = Mock(return_value=mock_choice)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        config = Mock(spec=OpenAIServiceConfig)
        config.model = "gpt-4"
        config.max_tokens = 4000
        config.frequency_penalty = 0
        config.presence_penalty = 0
        config.support_system_role = True
        config.require_alternative_roles = False
        
        service = OpenAIService.__new__(OpenAIService)
        service.config = config
        service._client = mock_client
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            service.chat_completion_with_function_calling(
                messages=[{"role": "user", "content": "Generate workflow"}],
                functions=[{"name": "create_workflow_ir", "parameters": {}}],
                function_call={"name": "create_workflow_ir"}
            )
        
        assert "Invalid JSON from function call" in str(exc_info.value)
    
    def test_multiple_tool_calls_error(self):
        """Test error when OpenAI returns multiple tool calls."""
        from taskweaver.llm.openai import OpenAIService, OpenAIServiceConfig
        
        # Setup mock response with multiple tool calls
        mock_tool_call1 = Mock()
        mock_tool_call1.function.name = "create_workflow_ir"
        mock_tool_call1.function.arguments = "{}"
        
        mock_tool_call2 = Mock()
        mock_tool_call2.function.name = "another_function"
        mock_tool_call2.function.arguments = "{}"
        
        mock_message = Mock()
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices.__getitem__ = Mock(return_value=mock_choice)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        config = Mock(spec=OpenAIServiceConfig)
        config.model = "gpt-4"
        config.max_tokens = 4000
        config.frequency_penalty = 0
        config.presence_penalty = 0
        config.support_system_role = True
        config.require_alternative_roles = False
        
        service = OpenAIService.__new__(OpenAIService)
        service.config = config
        service._client = mock_client
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            service.chat_completion_with_function_calling(
                messages=[{"role": "user", "content": "Generate workflow"}],
                functions=[{"name": "create_workflow_ir", "parameters": {}}]
            )
        
        assert "Expected exactly one function call, got 2" in str(exc_info.value)
    
    def test_no_tool_calls_error(self):
        """Test error when OpenAI returns no tool calls."""
        from taskweaver.llm.openai import OpenAIService, OpenAIServiceConfig
        
        # Setup mock response with no tool calls
        mock_message = Mock()
        mock_message.tool_calls = None
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices.__getitem__ = Mock(return_value=mock_choice)
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        config = Mock(spec=OpenAIServiceConfig)
        config.model = "gpt-4"
        config.max_tokens = 4000
        config.frequency_penalty = 0
        config.presence_penalty = 0
        config.support_system_role = True
        config.require_alternative_roles = False
        
        service = OpenAIService.__new__(OpenAIService)
        service.config = config
        service._client = mock_client
        
        # Execute and assert exception
        with pytest.raises(Exception) as exc_info:
            service.chat_completion_with_function_calling(
                messages=[{"role": "user", "content": "Generate workflow"}],
                functions=[{"name": "create_workflow_ir", "parameters": {}}]
            )
        
        assert "Expected exactly one function call, got 0" in str(exc_info.value)


class TestExtractToolIds:
    """Test suite for _extract_tool_ids_from_actions() method."""
    
    def setup_method(self):
        """Setup CodeGenerator instance for testing."""
        app_injector = Injector([PluginModule, LoggingModule])
        app_config = AppConfigSource(
            config={
                "app_dir": os.path.dirname(os.path.abspath(__file__)),
                "llm.api_key": "test_key",
                "code_generator.use_function_calling": True,
            },
        )
        app_injector.binder.bind(AppConfigSource, to=app_config)
        
        from taskweaver.code_interpreter.code_interpreter import CodeGenerator
        self.code_generator = app_injector.create_object(CodeGenerator)
    
    def test_extract_single_tool(self):
        """Test extracting a single tool ID."""
        composio_actions = """
Available Composio Actions:
- GMAIL_GET_MAIL_V2: Fetch emails from Gmail
"""
        tool_ids = self.code_generator._extract_tool_ids_from_actions(composio_actions)
        
        assert len(tool_ids) == 1
        assert tool_ids[0] == "GMAIL_GET_MAIL_V2"
    
    def test_extract_multiple_tools(self):
        """Test extracting multiple tool IDs."""
        composio_actions = """
Available Composio Actions:
- GMAIL_GET_MAIL_V2: Fetch emails from Gmail
- SLACKBOT_SEND_MESSAGE: Send message to Slack
- OUTLOOK_SEND_EMAIL: Send email via Outlook
"""
        tool_ids = self.code_generator._extract_tool_ids_from_actions(composio_actions)
        
        assert len(tool_ids) == 3
        assert "GMAIL_GET_MAIL_V2" in tool_ids
        assert "SLACKBOT_SEND_MESSAGE" in tool_ids
        assert "OUTLOOK_SEND_EMAIL" in tool_ids
    
    def test_extract_with_extra_text(self):
        """Test extraction with extra descriptive text."""
        composio_actions = """
Here are the available tools:

Available Composio Actions:
- GMAIL_GET_MAIL_V2: Fetch emails from Gmail (query: "subject:important")
- SLACKBOT_SEND_MESSAGE: Send message to Slack channel

Total: 2 actions
"""
        tool_ids = self.code_generator._extract_tool_ids_from_actions(composio_actions)
        
        assert len(tool_ids) == 2
        assert "GMAIL_GET_MAIL_V2" in tool_ids
        assert "SLACKBOT_SEND_MESSAGE" in tool_ids
    
    def test_extract_empty_input(self):
        """Test extraction with empty input."""
        tool_ids = self.code_generator._extract_tool_ids_from_actions("")
        
        assert len(tool_ids) == 0
    
    def test_extract_no_tools(self):
        """Test extraction when no tools match pattern."""
        composio_actions = """
Some random text without tool IDs
"""
        tool_ids = self.code_generator._extract_tool_ids_from_actions(composio_actions)
        
        assert len(tool_ids) == 0


class TestBuildWorkflowFunctionSchema:
    """Test suite for _build_workflow_function_schema() method."""
    
    def setup_method(self):
        """Setup CodeGenerator instance for testing."""
        app_injector = Injector([PluginModule, LoggingModule])
        app_config = AppConfigSource(
            config={
                "app_dir": os.path.dirname(os.path.abspath(__file__)),
                "llm.api_key": "test_key",
                "code_generator.use_function_calling": True,
            },
        )
        app_injector.binder.bind(AppConfigSource, to=app_config)
        
        from taskweaver.code_interpreter.code_interpreter import CodeGenerator
        self.code_generator = app_injector.create_object(CodeGenerator)
    
    def test_schema_structure(self):
        """Test basic schema structure."""
        tool_ids = ["GMAIL_GET_MAIL_V2", "SLACKBOT_SEND_MESSAGE"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        # Check top-level structure
        assert schema["name"] == "create_workflow_ir"
        assert "description" in schema
        assert "parameters" in schema
        
        # Check workflow structure
        workflow = schema["parameters"]["properties"]["workflow"]
        assert "nodes" in workflow["properties"]
        assert "edges" in workflow["properties"]
        assert "triggers" in workflow["properties"]
        assert "data_scopes" in workflow["properties"]
    
    def test_tool_id_enum_injection(self):
        """Test that tool IDs are correctly injected into enum."""
        tool_ids = ["TOOL_A", "TOOL_B", "TOOL_C"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        # Navigate to agent_with_tools node schema
        nodes_schema = schema["parameters"]["properties"]["workflow"]["properties"]["nodes"]
        agent_with_tools_schema = nodes_schema["items"]["oneOf"][0]
        
        # Check tool_id enum
        tool_id_enum = agent_with_tools_schema["properties"]["tool_id"]["enum"]
        assert tool_id_enum == tool_ids
    
    def test_node_types_mutual_exclusion(self):
        """Test that node types use oneOf for mutual exclusion."""
        tool_ids = ["TEST_TOOL"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        nodes_schema = schema["parameters"]["properties"]["workflow"]["properties"]["nodes"]
        
        # Check oneOf is present
        assert "oneOf" in nodes_schema["items"]
        
        # Check all node types are present
        node_types = [node["properties"]["type"]["const"] for node in nodes_schema["items"]["oneOf"]]
        expected_types = ["agent_with_tools", "agent_only", "form", "hitl", "loop", "sub_workflow"]
        
        assert set(node_types) == set(expected_types)
    
    def test_agent_with_tools_requires_tool_id(self):
        """Test that agent_with_tools requires tool_id."""
        tool_ids = ["TEST_TOOL"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        nodes_schema = schema["parameters"]["properties"]["workflow"]["properties"]["nodes"]
        agent_with_tools = nodes_schema["items"]["oneOf"][0]
        
        assert "tool_id" in agent_with_tools["required"]
        assert agent_with_tools["additionalProperties"] is False
    
    def test_form_no_tool_id(self):
        """Test that form node does NOT have tool_id."""
        tool_ids = ["TEST_TOOL"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        nodes_schema = schema["parameters"]["properties"]["workflow"]["properties"]["nodes"]
        form_node = next(n for n in nodes_schema["items"]["oneOf"] if n["properties"]["type"]["const"] == "form")
        
        assert "tool_id" not in form_node["properties"]
        assert "fields" in form_node["required"]
    
    def test_edge_types(self):
        """Test that edges have sequential and conditional types."""
        tool_ids = ["TEST_TOOL"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        edges_schema = schema["parameters"]["properties"]["workflow"]["properties"]["edges"]
        edge_types = [edge["properties"]["type"]["const"] for edge in edges_schema["items"]["oneOf"]]
        
        assert "sequential" in edge_types
        assert "conditional" in edge_types
    
    def test_triggers_schedule_and_event(self):
        """Test that triggers support schedule and event types."""
        tool_ids = ["TEST_TOOL"]
        schema = self.code_generator._build_workflow_function_schema(tool_ids)
        
        triggers_schema = schema["parameters"]["properties"]["workflow"]["properties"]["triggers"]
        trigger_types = [t["properties"]["type"]["const"] for t in triggers_schema["items"]["oneOf"]]
        
        assert "schedule" in trigger_types
        assert "event" in trigger_types


class TestValidateWorkflowIR:
    """Test suite for _validate_workflow_ir() method."""
    
    def setup_method(self):
        """Setup CodeGenerator instance for testing."""
        app_injector = Injector([PluginModule, LoggingModule])
        app_config = AppConfigSource(
            config={
                "app_dir": os.path.dirname(os.path.abspath(__file__)),
                "llm.api_key": "test_key",
                "code_generator.use_function_calling": True,
            },
        )
        app_injector.binder.bind(AppConfigSource, to=app_config)
        
        from taskweaver.code_interpreter.code_interpreter import CodeGenerator
        self.code_generator = app_injector.create_object(CodeGenerator)
    
    def test_valid_workflow(self):
        """Test validation of a valid workflow."""
        workflow = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools", "tool_id": "TEST_TOOL"},
                {"id": "node2", "type": "form", "fields": []}
            ],
            "edges": [
                {"type": "sequential", "from": "node1", "to": "node2"}
            ]
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is True
    
    def test_missing_nodes_field(self):
        """Test validation fails when nodes field is missing."""
        workflow = {
            "edges": []
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is False
    
    def test_missing_edges_field(self):
        """Test validation fails when edges field is missing."""
        workflow = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools", "tool_id": "TEST_TOOL"}
            ]
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is False
    
    def test_duplicate_node_ids(self):
        """Test validation fails with duplicate node IDs."""
        workflow = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools", "tool_id": "TEST_TOOL"},
                {"id": "node1", "type": "form", "fields": []}  # Duplicate ID
            ],
            "edges": []
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is False
    
    def test_agent_with_tools_missing_tool_id(self):
        """Test validation fails when agent_with_tools missing tool_id."""
        workflow = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools"}  # Missing tool_id
            ],
            "edges": []
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is False
    
    def test_edge_references_unknown_node(self):
        """Test validation fails when edge references unknown node."""
        workflow = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools", "tool_id": "TEST_TOOL"}
            ],
            "edges": [
                {"type": "sequential", "from": "node1", "to": "unknown_node"}  # Unknown node
            ]
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is False
    
    def test_conditional_edge_unknown_source(self):
        """Test validation fails when conditional edge has unknown source."""
        workflow = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools", "tool_id": "TEST_TOOL"}
            ],
            "edges": [
                {
                    "type": "conditional",
                    "source": "unknown",
                    "condition": "True",
                    "if_true": "node1",
                    "if_false": "node1"
                }
            ]
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is False
    
    def test_workflow_with_all_node_types(self):
        """Test validation of workflow using all node types."""
        workflow = {
            "nodes": [
                {"id": "n1", "type": "agent_with_tools", "tool_id": "TOOL1"},
                {"id": "n2", "type": "agent_only", "code": "print('test')"},
                {"id": "n3", "type": "form", "fields": []},
                {"id": "n4", "type": "hitl", "approval_type": "approve_reject"},
                {"id": "n5", "type": "loop", "iterate_over": "${{items}}", "loop_body": ["n1"]},
                {"id": "n6", "type": "sub_workflow", "workflow_id": "sub_wf_1"}
            ],
            "edges": []
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is True
    
    def test_workflow_with_conditional_edges(self):
        """Test validation of workflow with conditional edges."""
        workflow = {
            "nodes": [
                {"id": "n1", "type": "hitl", "approval_type": "approve_reject"},
                {"id": "n2", "type": "agent_with_tools", "tool_id": "TOOL1"},
                {"id": "n3", "type": "agent_with_tools", "tool_id": "TOOL2"}
            ],
            "edges": [
                {
                    "type": "conditional",
                    "source": "n1",
                    "condition": "${{approved}} == True",
                    "if_true": "n2",
                    "if_false": "n3"
                }
            ]
        }
        
        assert self.code_generator._validate_workflow_ir(workflow) is True


class TestConvertWorkflowJsonToPython:
    """Test suite for _convert_workflow_json_to_python() method."""
    
    def setup_method(self):
        """Setup CodeGenerator instance for testing."""
        app_injector = Injector([PluginModule, LoggingModule])
        app_config = AppConfigSource(
            config={
                "app_dir": os.path.dirname(os.path.abspath(__file__)),
                "llm.api_key": "test_key",
                "code_generator.use_function_calling": True,
            },
        )
        app_injector.binder.bind(AppConfigSource, to=app_config)
        
        from taskweaver.code_interpreter.code_interpreter import CodeGenerator
        self.code_generator = app_injector.create_object(CodeGenerator)
    
    def test_basic_conversion(self):
        """Test basic JSON to Python conversion."""
        workflow_json = {
            "nodes": [{"id": "test", "type": "agent_with_tools", "tool_id": "TEST"}],
            "edges": []
        }
        
        python_code = self.code_generator._convert_workflow_json_to_python(workflow_json)
        
        assert "WORKFLOW = " in python_code
        assert "result = WORKFLOW" in python_code
        assert "'nodes'" in python_code or '"nodes"' in python_code
    
    def test_conversion_preserves_structure(self):
        """Test that conversion preserves workflow structure."""
        workflow_json = {
            "nodes": [
                {"id": "node1", "type": "agent_with_tools", "tool_id": "TOOL1"},
                {"id": "node2", "type": "form", "fields": [{"name": "test"}]}
            ],
            "edges": [
                {"type": "sequential", "from": "node1", "to": "node2"}
            ]
        }
        
        python_code = self.code_generator._convert_workflow_json_to_python(workflow_json)
        
        # Should be valid Python that can be parsed
        assert "WORKFLOW = " in python_code
        assert len(python_code) > 50  # Should have substantial content
        assert "node1" in python_code
        assert "node2" in python_code
    
    def test_conversion_handles_nested_structures(self):
        """Test conversion handles nested structures correctly."""
        workflow_json = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "form",
                    "fields": [
                        {"name": "field1", "type": "string", "required": True},
                        {"name": "field2", "type": "number", "required": False}
                    ]
                }
            ],
            "edges": []
        }
        
        python_code = self.code_generator._convert_workflow_json_to_python(workflow_json)
        
        assert "WORKFLOW = " in python_code
        assert "field1" in python_code
        assert "field2" in python_code
    
    def test_conversion_output_is_valid_python(self):
        """Test that converted output can be executed as Python."""
        workflow_json = {
            "nodes": [{"id": "test", "type": "agent_only", "code": "print('hello')"}],
            "edges": []
        }
        
        python_code = self.code_generator._convert_workflow_json_to_python(workflow_json)
        
        # Execute the generated code to verify it's valid Python
        local_vars = {}
        exec(python_code, {}, local_vars)
        
        # Check that WORKFLOW and result are defined
        assert "result" in local_vars
        assert isinstance(local_vars["result"], dict)
        assert "nodes" in local_vars["result"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
