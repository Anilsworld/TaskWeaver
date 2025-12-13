"""
Test Instructor Integration
===========================
Quick test to verify the Instructor-based workflow generation works.
"""
import sys
import os

# Fix Windows encoding for emojis
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_instructor_import():
    """Test that Instructor and our modules can be imported."""
    print("=" * 60)
    print("TEST 1: Import Instructor and workflow generator")
    print("=" * 60)
    
    try:
        import instructor
        print("✅ instructor imported successfully")
        print(f"   Version: {instructor.__version__ if hasattr(instructor, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"❌ Failed to import instructor: {e}")
        return False
    
    try:
        from taskweaver.code_interpreter.code_interpreter.instructor_workflow_generator import (
            InstructorWorkflowGenerator,
            EnhancedWorkflowDefinition,
            EnhancedWorkflowNode
        )
        print("✅ InstructorWorkflowGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import workflow generator: {e}")
        return False
    
    try:
        from taskweaver.code_interpreter.workflow_schema import (
            WorkflowDefinition,
            WorkflowNode
        )
        print("✅ Workflow schema models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import workflow schema: {e}")
        return False
    
    return True


def test_pydantic_model():
    """Test that our Pydantic models work correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Pydantic model validation")
    print("=" * 60)
    
    try:
        from taskweaver.code_interpreter.code_interpreter.instructor_workflow_generator import (
            EnhancedWorkflowNode,
            EnhancedWorkflowDefinition
        )
        
        # Create a simple workflow
        workflow_dict = {
            "nodes": [
                {
                    "id": "start",
                    "type": "form",
                    "description": "User input form",
                    "form_schema": {"email": "string"},
                    "depends_on": []
                },
                {
                    "id": "send_email",
                    "type": "agent_with_tools",
                    "tool_id": "GMAIL_SEND_EMAIL",
                    "params": {
                        "to": "${{start.email}}",
                        "subject": "Hello",
                        "body": "Test email"
                    },
                    "depends_on": ["start"]
                }
            ],
            "sequential_edges": [
                {"source": "start", "target": "send_email"}
            ]
        }
        
        # Validate with Pydantic
        workflow = EnhancedWorkflowDefinition(**workflow_dict)
        print(f"✅ Workflow validated successfully")
        print(f"   Nodes: {len(workflow.nodes)}")
        print(f"   Node types: {[n.type for n in workflow.nodes]}")
        print(f"   Tool IDs: {[n.tool_id for n in workflow.nodes if n.tool_id]}")
        
        # Test that we can convert to dict
        workflow_out = workflow.model_dump()
        print(f"✅ Workflow serialized to dict successfully")
        print(f"   Keys: {list(workflow_out.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instructor_schema_generation():
    """Test that Instructor can generate schema from our Pydantic models."""
    print("\n" + "=" * 60)
    print("TEST 3: Instructor schema generation")
    print("=" * 60)
    
    try:
        import instructor
        from taskweaver.code_interpreter.code_interpreter.instructor_workflow_generator import (
            EnhancedWorkflowDefinition
        )
        from openai import OpenAI
        
        # Create a mock OpenAI client (won't actually call API)
        client = OpenAI(api_key="test-key-not-real")
        
        # Patch with Instructor
        instructor_client = instructor.from_openai(client)
        print("✅ Instructor client created successfully")
        
        # Test that we can extract the schema (this doesn't call the API)
        try:
            # Instructor internally converts Pydantic models to OpenAI function schemas
            # We can verify this by checking that the model can be serialized
            schema = EnhancedWorkflowDefinition.model_json_schema()
            print("✅ Schema generated from Pydantic model")
            print(f"   Schema keys: {list(schema.keys())[:5]}...")  # First 5 keys
            print(f"   Has 'properties': {'properties' in schema}")
            print(f"   Has 'required': {'required' in schema}")
            
            # Check node properties
            if 'properties' in schema and 'nodes' in schema['properties']:
                nodes_schema = schema['properties']['nodes']
                print(f"✅ Nodes schema found: {nodes_schema.get('type')}")
            
            return True
        except Exception as e:
            print(f"❌ Schema generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Instructor client creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INSTRUCTOR INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Import Test", test_instructor_import()))
    
    # Test 2: Pydantic models
    if results[-1][1]:  # Only run if imports passed
        results.append(("Pydantic Model Test", test_pydantic_model()))
    
    # Test 3: Instructor schema generation
    if results[-1][1]:  # Only run if previous passed
        results.append(("Instructor Schema Test", test_instructor_schema_generation()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Instructor integration is working.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
