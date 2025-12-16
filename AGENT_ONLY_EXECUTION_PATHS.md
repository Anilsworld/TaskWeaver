# agent_only Node Execution Paths - ACTUAL CODE ANALYSIS

## 🔍 Current Implementation (langgraph_adapter.py)

### Routing: Line 886-889
```python
elif node_type_str == 'agent_only':
    node_func = self._build_agent_node(node)  # ← Goes to _build_agent_node()
```

---

## 📋 _build_agent_node() Logic Flow

### Step 1: Check if code exists (Lines 1887-1891)
```python
has_code = bool(
    hasattr(node, 'code') and 
    node.code and 
    node.code.strip()
)
```

### Step 2: IF has_code=True (Lines 1896-2268)
```python
if has_code and HAS_TASKWEAVER:
    # Execute code via TaskWeaver
    session = await sync_to_async(self._get_taskweaver_session)(...)
    
    # Inject variables from state
    session_vars = {}
    for node_id, output_data in step_outputs.items():
        session_vars[f'step_{step_num}_result'] = output_data
    
    # Execute code
    result = await code_interpreter.execute_code(code, ...)
    
    if result.success:
        # Store outputs
        return state
    else:
        # TRY ADAPTIVE CODE GENERATION (Lines 2173-2236)
        adaptive_executor = get_adaptive_executor(...)
        success, adaptive_result = await adaptive_executor.execute_with_adaptation(...)
        
        if success:
            return state  # Success with adaptive code
        else:
            raise RuntimeError("Code execution failed")
```

### Step 3: ELSE if has_code=False (Lines 2270-2281)
```python
# ✅ CLEAN ARCHITECTURE: No code = error
if not has_code:
    self.logger.error(f"❌ [AGENT_NODE] No code found for node {node.step_number}")
    errors = state.get("errors", [])
    errors.append(f"Step {node.step_number}: No TaskWeaver code attached to node")
    return {**state, "errors": errors}  # ← RETURNS ERROR!
```

---

## 🚨 CURRENT BEHAVIOR: No LLM Reasoning Path!

### agent_only Execution Outcomes:

| Scenario | Current Behavior |
|----------|------------------|
| `has_code=True` + success | ✅ Execute code, return results |
| `has_code=True` + fails | ⚡ Try adaptive code gen, or error |
| `has_code=False` | ❌ **RETURN ERROR** |

**KEY FINDING:** There is **NO pure LLM reasoning path** currently!

---

## 🎯 What User Is Asking For

The user says:
> "we have a logic (not sure if its already implemented) within agentonly to decide its just some llm call with reasoning or codeexecution"

**Translation:**
`agent_only` should have **TWO paths**:

### Path A: Pure LLM Reasoning (NO code execution)
- Use case: "Analyze sentiment", "Rank results", "Provide recommendation"
- Implementation: Call LLM with context, get text response
- **STATUS:** ❌ NOT IMPLEMENTED

### Path B: Code Execution
- Use case: "Transform data", "Calculate metrics", "Parse JSON"
- Implementation: Execute TaskWeaver-generated code
- **STATUS:** ✅ IMPLEMENTED (lines 1896-2268)

---

## 🔧 Missing Implementation

### What Needs to Be Added (Lines 2270-2281 replacement):

```python
# NEW LOGIC: Branch based on task type
if not has_code:
    # Check if this is a reasoning-only task (no code needed)
    is_reasoning_task = self._is_pure_reasoning_task(node)
    
    if is_reasoning_task:
        # PATH A: Pure LLM reasoning (NO code execution)
        self.logger.info(f"🧠 [LLM_REASONING] Step {node.step_number}: Pure reasoning task")
        
        # Build context from previous steps
        context = self._build_agent_context(node, state)
        
        # Call LLM for reasoning
        from apps.py_workflows.shared.llm.factory import get_langchain_model
        llm = get_langchain_model()
        
        prompt = f"""
        Task: {node.description or node.node_name}
        Context: {context}
        
        Provide your analysis/reasoning:
        """
        
        response = await llm.ainvoke(prompt)
        reasoning_output = response.content
        
        # Store output
        output_data = {
            'reasoning': reasoning_output,
            'type': 'llm_reasoning'
        }
        
        state_updates = store_node_output_dual(
            state=state,
            node=node,
            primary_output=output_data,
            output_type="reasoning"
        )
        
        return {**state, **state_updates}
    else:
        # PATH B: Code required but missing - ERROR
        self.logger.error(f"❌ [AGENT_NODE] No code found for code-execution node {node.step_number}")
        errors = state.get("errors", [])
        errors.append(f"Step {node.step_number}: No TaskWeaver code attached to node")
        return {**state, "errors": errors}
```

---

## 🤔 How to Determine Task Type?

### Option 1: Check node metadata
```python
def _is_pure_reasoning_task(self, node):
    # Check if node explicitly marked as reasoning-only
    return node.metadata.get('execution_mode') == 'reasoning'
```

### Option 2: Check description keywords
```python
def _is_pure_reasoning_task(self, node):
    reasoning_keywords = ['analyze', 'assess', 'evaluate', 'recommend', 
                          'rank', 'prioritize', 'explain', 'summarize']
    description = (node.description or node.node_name or '').lower()
    return any(keyword in description for keyword in reasoning_keywords)
```

### Option 3: During workflow generation, decide and mark
```python
# In workflow_schema_builder.py or code_generator.py
# When generating agent_only node, add metadata:
node['metadata']['execution_mode'] = 'reasoning'  # or 'code'
```

---

## ✅ RECOMMENDATION

### 1. Add execution_mode to Schema
In `workflow_schema_builder.py`, add to agent_only schema:
```python
"execution_mode": {
    "type": "string",
    "enum": ["reasoning", "code"],
    "description": "How to execute: 'reasoning' for pure LLM, 'code' for Python execution"
}
```

### 2. Update Planner Prompt
In `planner_prompt.yaml`:
```yaml
- `<agent_only>` - Pure AI processing WITHOUT external tools
  * Use when: AI needs to analyze/reason OR execute transformations
  * Two modes:
    - REASONING: "Analyze sentiment", "Rank by priority" (pure AI thinking)
    - CODE: "Transform JSON", "Calculate average" (executes Python)
```

### 3. Update Code Generator
In `code_generator_prompt.yaml`:
```yaml
**agent_only nodes have TWO execution modes:**
1. REASONING mode: No code generated, LLM provides analysis at runtime
2. CODE mode: Generate Python code for data transformation

**Use REASONING mode when:**
- Task is subjective analysis ("Which option is better?")
- No data manipulation needed
- Output is recommendation/explanation

**Use CODE mode when:**
- Data transformation/calculation needed
- Deterministic operations
- Working with structured data
```

### 4. Implement in langgraph_adapter.py
Replace lines 2270-2281 with the branching logic shown above.

---

## 🎯 SUMMARY

**Current State:**
- agent_only ONLY does code execution
- No code = ERROR

**Desired State:**
- agent_only decides: reasoning OR code
- Two paths based on task type

**Implementation:**
1. Add `execution_mode` metadata field
2. Update prompts to distinguish use cases
3. Add LLM reasoning path in `_build_agent_node()`
4. Deprecate/remove `code_execution` type (redundant)

