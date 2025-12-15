# Implementation Summary: agent_only Dual-Mode + Cross-References

**Date:** December 14, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 What Was Implemented

### 1. agent_only Dual Execution Modes

agent_only nodes now support TWO execution modes:

| Mode | When to Use | How It Works |
|------|-------------|--------------|
| **reasoning** | "Analyze sentiment", "Which is better?", "Provide recommendation" | Pure LLM call at runtime - no code |
| **code** | "Transform JSON", "Calculate average", "Parse data" | Executes TaskWeaver-generated Python |

---

## 📝 Changes Made

### File 1: `workflow_schema_builder.py`

#### agent_only Schema (Lines 569-598)
```python
# BEFORE:
"description": "Agent-only node (deprecated, use code_execution)"

# AFTER:
"description": (
    "Agent-only node for AI processing WITHOUT external tools. "
    "Two modes: (1) REASONING - pure LLM analysis/recommendation, "
    "(2) CODE - executes TaskWeaver-generated Python code."
)

# NEW FIELD:
"agent_mode": {
    "type": "string",
    "enum": ["reasoning", "code"],
    "description": (
        "'reasoning' = Pure LLM call for analysis (no code). "
        "'code' = Execute TaskWeaver-generated Python code. "
        "Default: 'code' (if code is generated)"
    )
}
```

#### code_execution Deprecation (Lines 520-523)
```python
# BEFORE:
# code_execution - Python code node

# AFTER:
# code_execution - DEPRECATED: Use agent_only with agent_mode='code' instead
# This node type is redundant with agent_only (agent_mode='code')
# Kept for backward compatibility only
```

#### Cross-Reference Comments Added
- ✅ agent_with_tools (Lines 423-425)
- ✅ form (Lines 604-606)
- ✅ hitl (Lines 654-656)
- ✅ parallel (Lines 679-681)
- ✅ loop (Lines 703-705)
- ✅ agent_only (Lines 572-574)

---

### File 2: `planner_prompt.yaml`

#### agent_only Marker Update (Lines 77-84)
```yaml
# ADDED:
* Two execution modes (auto-determined during generation):
  - REASONING: Pure AI analysis/recommendations (e.g., "Analyze sentiment")
  - CODE: AI-generated Python for transformations (e.g., "Transform JSON")
* Cross-reference: workflow_schema_builder.py defines agent_mode field
```

---

### File 3: `code_generator_prompt.yaml`

#### agent_only Documentation (Lines 408-421)
```yaml
# ADDED:
**TWO EXECUTION MODES** (decided during generation):
1. REASONING mode: Pure LLM call for subjective analysis
   - Use for: "Which option is better?", "Analyze sentiment"
   - No code generated - LLM provides analysis at runtime
2. CODE mode: TaskWeaver-generated Python for transformations
   - Use for: "Transform JSON", "Calculate average"
   - Code generated during workflow creation, executed at runtime

* Cross-reference: workflow_schema_builder.py defines agent_mode field
* Cross-reference: langgraph_adapter._build_agent_node() handles both modes
```

---

### File 4: `langgraph_adapter.py`

#### Updated Docstring (Lines 1853-1870)
```python
def _build_agent_node(self, node: WorkflowNode):
    """
    Build an agent_only node with dual execution modes.
    
    🎯 TWO EXECUTION PATHS:
    1. REASONING mode (agent_mode='reasoning' or no code):
       - Pure LLM call for subjective analysis/recommendations
       - No code execution, just AI reasoning
    
    2. CODE mode (agent_mode='code' and has code):
       - Executes TaskWeaver-generated Python code
    """
```

#### New LLM Reasoning Path (Lines 2270-2352)
```python
# BEFORE:
if not has_code:
    self.logger.error("No code found")
    return {**state, "errors": errors}  # ERROR!

# AFTER:
if not has_code:
    agent_mode = getattr(node, 'agent_mode', 'reasoning')
    
    if agent_mode == 'reasoning':
        # PATH A: Pure LLM reasoning
        llm = get_langchain_model(temperature=0.7)
        response = await llm.ainvoke(messages)
        reasoning_output = response.content
        
        # Store output and return state
        return state
    else:
        # PATH B: Code mode but code missing - ERROR
        return {**state, "errors": errors}
```

**Key Features:**
- ✅ Builds context from previous step outputs
- ✅ Calls LLM with task description
- ✅ Stores reasoning output in state
- ✅ Saves checkpoint
- ✅ Broadcasts completion event

---

## 🔗 Cross-Reference Map

| File | References | Referenced By |
|------|-----------|---------------|
| **planner_prompt.yaml** | → workflow_schema_builder.py | ← All (defines markers) |
| **workflow_schema_builder.py** | → langgraph_adapter.py | ← planner, code_gen |
| **code_generator_prompt.yaml** | → workflow_schema_builder.py, langgraph_adapter.py | ← (generates workflows) |
| **langgraph_adapter.py** | - | ← All (executes workflows) |

### Cross-Reference Comments Format:
```python
# Cross-reference: planner_prompt.yaml defines <marker> syntax
# Cross-reference: code_generator_prompt.yaml explains usage patterns
# Cross-reference: langgraph_adapter.py._method() handles execution
```

---

## 📊 Execution Flow

### REASONING Mode:
```
1. Planner: "Analyze sentiment <agent_only>"
   ↓
2. Code Generator: Doesn't generate code (reasoning task)
   → Sets agent_mode='reasoning'
   ↓
3. Workflow IR: {type: "agent_only", agent_mode: "reasoning", code: null}
   ↓
4. Runtime: langgraph_adapter._build_agent_node()
   → Detects has_code=False, agent_mode='reasoning'
   → Calls LLM for analysis
   → Returns reasoning output
```

### CODE Mode:
```
1. Planner: "Transform JSON to CSV <agent_only>"
   ↓
2. Code Generator: Generates Python code via TaskWeaver
   → Sets agent_mode='code'
   ↓
3. Workflow IR: {type: "agent_only", agent_mode: "code", code: "import json..."}
   ↓
4. Runtime: langgraph_adapter._build_agent_node()
   → Detects has_code=True
   → Executes code via TaskWeaver
   → Returns code execution result
```

---

## ✅ Backward Compatibility

### Existing Workflows:
- ✅ Workflows with `agent_only` + code → Continue working (default agent_mode='code')
- ✅ Workflows with `code_execution` → Still work (node type not removed, just deprecated)
- ✅ Workflows without agent_mode field → Default to 'reasoning' if no code, 'code' if has code

### Migration Path:
```python
# Old workflow (still works):
{
    "type": "agent_only",
    "code": "..."  # Will execute code (agent_mode defaults to 'code')
}

# New workflow (explicit):
{
    "type": "agent_only",
    "agent_mode": "reasoning"  # Pure LLM call
}

{
    "type": "agent_only",
    "agent_mode": "code",
    "code": "..."  # Explicit code execution
}
```

---

## 🧪 Testing Checklist

### Test Case 1: Reasoning Mode
```python
node = WorkflowNode(
    type="agent_only",
    agent_mode="reasoning",
    description="Analyze which product is best value for money"
)
# Expected: LLM provides recommendation
```

### Test Case 2: Code Mode
```python
node = WorkflowNode(
    type="agent_only",
    agent_mode="code",
    code="result = {'avg': sum(data) / len(data)}"
)
# Expected: Executes code via TaskWeaver
```

### Test Case 3: Legacy (no agent_mode, no code)
```python
node = WorkflowNode(
    type="agent_only",
    description="Analyze data"
)
# Expected: Defaults to reasoning mode
```

### Test Case 4: Legacy (no agent_mode, has code)
```python
node = WorkflowNode(
    type="agent_only",
    code="result = transform(data)"
)
# Expected: Executes code (backward compatible)
```

---

## 📈 Benefits

1. ✅ **Single Source of Truth**: agent_only is now the unified node type for AI processing
2. ✅ **Clear Distinction**: Two modes make it obvious when code is needed vs pure reasoning
3. ✅ **Backward Compatible**: Existing workflows continue working
4. ✅ **Cross-Referenced**: All files link to each other for maintainability
5. ✅ **Execution Flexibility**: Can choose between fast reasoning or deterministic code

---

## 🚀 Next Steps (Optional Enhancements)

### Future Enhancements (NOT in scope for now):
1. Add `agent_mode` auto-detection during workflow generation
   - Keywords: "analyze", "assess" → reasoning
   - Keywords: "transform", "calculate" → code

2. Add metrics tracking:
   - Track reasoning vs code execution times
   - Monitor LLM token usage for reasoning mode

3. Add reasoning caching:
   - Cache LLM reasoning for identical contexts
   - Reduce latency and cost

---

## 📝 Documentation Updates Needed

- [ ] Update user-facing docs with agent_mode examples
- [ ] Add migration guide for old workflows
- [ ] Document when to use reasoning vs code mode
- [ ] Add API reference for agent_mode field

---

## ✅ COMPLETE

All changes implemented and cross-referenced. The agent_only node type now supports dual execution modes with backward compatibility!
