# code_execution Merged Into agent_only

**Date:** December 15, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 What Changed

### Before (Two Separate Types):
```json
// Option 1: Pure reasoning
{"type": "agent_only", "description": "Analyze sentiment"}

// Option 2: Code execution
{"type": "code_execution", "code": "result = ..."}
```

**Problem:** LLM had to choose between two similar types → confusion

---

### After (Unified Type):
```json
// Option 1: Reasoning mode
{"type": "agent_only", "agent_mode": "reasoning", "description": "Analyze sentiment"}

// Option 2: Code mode
{"type": "agent_only", "agent_mode": "code", "code": "result = ..."}
```

**Solution:** Single type with `agent_mode` field → clear distinction

---

## 📝 Changes Made

### 1. workflow_schema_builder.py

#### Removed code_execution from Schema (Lines 557-570)
```python
# ❌ REMOVED: code_execution schema
# node_schemas.append({
#     "type": "object",
#     "description": "Python code execution node",
#     "properties": {
#         "type": {"type": "string", "enum": ["code_execution"]},
#         ...
#     }
# })
```

#### Updated agent_only Schema (Lines 575-605)
```python
node_schemas.append({
    "type": "object",
    "description": (
        "Agent-only node for AI processing WITHOUT external tools. "
        "🎯 TWO MODES:\n"
        "1. REASONING (agent_mode='reasoning'): Pure LLM analysis\n"
        "2. CODE (agent_mode='code'): Execute Python code\n"
    ),
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["agent_only"]},  # ← Only agent_only
        "agent_mode": {
            "type": "string",
            "enum": ["reasoning", "code"],  # ← Mode selector
            ...
        },
        "code": {"type": "string"},  # ← Optional (required if agent_mode='code')
        "description": {"type": "string"}
    },
    "required": ["id", "type"],  # ← agent_mode NOT required (defaults based on code presence)
})
```

**Key Changes:**
- ✅ `code_execution` type removed from enum
- ✅ `agent_mode` field added to `agent_only`
- ✅ `code` field now conditional (not always required)

---

### 2. langgraph_adapter.py (Line 905-908)

#### Backward Compatibility Mapping
```python
elif node_type_str == 'code_execution':
    # ⚠️ BACKWARD COMPATIBILITY: code_execution → agent_only
    self.logger.warning("code_execution is deprecated, mapping to agent_only")
    node_func = self._build_agent_node(node)  # ← Routes to same handler
```

**What This Does:**
- Old workflows with `type: code_execution` still work
- Internally mapped to `agent_only` path
- Uses existing `_build_agent_node()` logic

---

## 🔄 Execution Flow

### New Workflow Generated (LLM Output):
```json
{
  "type": "agent_only",
  "agent_mode": "code",
  "code": "result = calculate(data)"
}
```

### Routing in langgraph_adapter.py:
```python
# Line 886-889: Detects type='agent_only'
elif node_type_str == 'agent_only':
    node_func = self._build_agent_node(node)
    
# _build_agent_node() checks agent_mode:
if has_code or agent_mode == 'code':
    # Execute code via TaskWeaver
else:
    # Pure LLM reasoning
```

---

## ✅ Benefits

| Before | After |
|--------|-------|
| 2 node types (confusion) | 1 node type (clarity) |
| LLM guesses which to use | LLM sets agent_mode |
| code_execution + agent_only | agent_only with modes |
| Separate execution paths | Unified execution path |

---

## 🧪 Expected Test Result

**Prompt:** "Fetch my unread emails from Gmail and tell me which ones are most urgent"

### Before (What We Saw):
```json
{
  "nodes": [
    {"type": "agent_with_tools", "tool_id": "GMAIL_FETCH_EMAILS"},
    {"type": "code_execution", "code": "..."}  // ❌ Wrong choice
  ]
}
```

### After (Expected):
```json
{
  "nodes": [
    {"type": "agent_with_tools", "tool_id": "GMAIL_FETCH_EMAILS"},
    {
      "type": "agent_only",
      "agent_mode": "reasoning",  // ✅ Correct choice
      "description": "Analyze which emails are urgent"
    }
  ]
}
```

**OR if code is needed:**
```json
{
  "type": "agent_only",
  "agent_mode": "code",
  "code": "urgent_emails = [e for e in emails if is_urgent(e)]"
}
```

---

## 🔍 What the LLM Sees Now

### Schema (Simplified):
```
Available node types:
1. agent_with_tools - Execute external tools
2. agent_only - AI processing (TWO MODES):
   - agent_mode='reasoning': Pure LLM analysis
   - agent_mode='code': Execute Python code
3. form - Collect user input
4. hitl - Human approval
5. parallel - Execute simultaneously
6. loop - Iterate over collection
```

**Note:** `code_execution` is **NOT** in the list → LLM can't choose it

---

## 📊 Logs to Watch

When testing, look for:

```
[SCHEMA_TRACE] Node types in schema: ['agent_with_tools', 'agent_only', 'form', 'hitl', 'parallel', 'loop']
```

**Should NOT see:** `'code_execution'` in the list

---

## ✅ COMPLETE

- ✅ Schema updated (code_execution removed)
- ✅ agent_only enhanced (agent_mode added)
- ✅ Backward compatibility (old workflows still work)
- ✅ Execution path unified (_build_agent_node handles both)

**Next:** Test with the same prompt to see if LLM now chooses `agent_only` instead of `code_execution`!

