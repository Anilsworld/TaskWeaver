# Integration Tests for Workflow Function Calling

These tests use **REAL OpenAI API calls** to verify the function calling implementation works end-to-end.

## ⚠️ Prerequisites

1. **OpenAI API Key** - These tests will consume API tokens
2. **Environment Variables** - Set before running

## 🚀 Quick Start

### Option 1: PowerShell Script (Recommended)

```powershell
# 1. Set your API key
$env:OPENAI_API_KEY = "sk-your-key-here"

# 2. Run all integration tests
.\run_integration_tests.ps1

# 3. Run specific test
.\run_integration_tests.ps1 -TestName "test_real_openai_function_calling"

# 4. Use different model
.\run_integration_tests.ps1 -Model "gpt-4o-mini"
```

### Option 2: Manual pytest

```powershell
# 1. Set environment variables
$env:OPENAI_API_KEY = "sk-your-key-here"
$env:RUN_INTEGRATION_TESTS = "true"
$env:OPENAI_MODEL = "gpt-4o-2024-08-06"  # Optional

# 2. Run tests
python -m pytest tests/integration_tests/test_workflow_function_calling_integration.py -v -s
```

## 📋 Test Coverage

### TestOpenAIFunctionCallingIntegration
- **test_real_openai_function_calling** - Tests basic OpenAI function calling with simple workflow

### TestCodeGeneratorFunctionCallingIntegration
- **test_extract_tool_ids_with_real_selector_output** - Tool ID extraction with realistic output
- **test_build_schema_with_realistic_tools** - Schema generation with real tool lists
- **test_validate_realistic_workflow** - Multi-node workflow validation
- **test_convert_realistic_workflow_to_python** - Python code generation

### TestEndToEndFunctionCallingFlow
- **test_full_workflow_generation_flow** ⭐ - Complete end-to-end workflow generation
  - Extracts tool IDs
  - Builds function schema  
  - Calls OpenAI API
  - Validates workflow
  - Converts to Python
  - Executes generated code

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key |
| `RUN_INTEGRATION_TESTS` | ✅ Yes | `false` | Set to `true` to enable tests |
| `OPENAI_API_BASE` | ❌ No | `https://api.openai.com/v1` | API base URL |
| `OPENAI_MODEL` | ❌ No | `gpt-4o-2024-08-06` | Model to use |

## 💰 Cost Estimation

These tests make **REAL API calls** and will consume tokens:

| Test | Estimated Tokens | Estimated Cost* |
|------|------------------|-----------------|
| test_real_openai_function_calling | ~500 | $0.01 |
| test_extract_tool_ids_with_real_selector_output | ~0 (local only) | $0.00 |
| test_build_schema_with_realistic_tools | ~0 (local only) | $0.00 |
| test_validate_realistic_workflow | ~0 (local only) | $0.00 |
| test_convert_realistic_workflow_to_python | ~0 (local only) | $0.00 |
| test_full_workflow_generation_flow | ~1000 | $0.02 |
| **TOTAL (all tests)** | **~1500** | **~$0.03** |

*Based on gpt-4o-2024-08-06 pricing ($2.50/1M input, $10/1M output)

## 🎯 Success Criteria

All tests should:
1. ✅ Call OpenAI API successfully
2. ✅ Receive valid function call response
3. ✅ Validate tool_id is from enum (no hallucination)
4. ✅ Generate valid workflow IR
5. ✅ Convert to executable Python code

## 🐛 Troubleshooting

### Tests are skipped
**Problem:** All tests show `SKIPPED`

**Solution:** Set `RUN_INTEGRATION_TESTS=true`

```powershell
$env:RUN_INTEGRATION_TESTS = "true"
```

### API Key Error
**Problem:** `OPENAI_API_KEY not set`

**Solution:** Set your API key

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

### Rate Limit Error
**Problem:** `RateLimitError: You exceeded your rate limit`

**Solution:** Wait a few seconds and retry, or use a different API key

### Model Not Found
**Problem:** `InvalidRequestError: The model 'gpt-4o-...' does not exist`

**Solution:** Use a different model

```powershell
$env:OPENAI_MODEL = "gpt-4o-mini"
```

## 📊 Interpreting Results

### Successful Test Output
```
📡 Calling OpenAI API...
✅ Received 1 tool call(s)

📋 Generated workflow:
{
  "workflow": {
    "nodes": [...],
    "edges": [...]
  }
}

✅ Test passed! Generated 2 nodes, 1 edges
```

### Failed Test
Look for:
- ❌ **Invalid tool_id** - Enum validation failed (hallucination detected)
- ❌ **Validation failed** - Workflow structure incorrect
- ❌ **Python execution failed** - Generated code has syntax errors

## 🔒 Security Notes

⚠️ **NEVER commit your API key to git!**

These tests are designed to run in local development only. For CI/CD:
1. Use GitHub Secrets or similar
2. Set `RUN_INTEGRATION_TESTS=true` only for specific workflows
3. Monitor API usage to prevent unexpected costs

## 📈 Adding New Integration Tests

1. Add test method to appropriate class
2. Use `pytest.skip()` if prerequisites not met
3. Add cost estimate to table above
4. Document expected behavior

Example:
```python
def test_my_new_integration(self):
    """Test description."""
    if not self.api_key:
        pytest.skip("API key not set")
    
    # Your test code here
    ...
```

## 📝 Related Documentation

- [Unit Tests](../unit_tests/test_workflow_function_calling.py) - Mock-based tests (no API calls)
- [Implementation](../../taskweaver/code_interpreter/code_interpreter/code_generator.py) - CodeGenerator with function calling
- [OpenAI Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)
