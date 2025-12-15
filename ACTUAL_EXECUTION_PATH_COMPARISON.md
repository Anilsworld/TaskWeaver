# Actual Execution Path Comparison: agent_only vs code_execution

## 🔍 Code Path Analysis (langgraph_adapter.py)

### Routing Logic (Lines 886-908)

```python
# LINE 886-889: agent_only
elif node_type_str == 'agent_only':
    node_func = self._build_agent_node(node)  # ← Calls _build_agent_node
    
# LINE 905-908: code_execution  
elif node_type_str == 'code_execution':
    node_func = self._build_code_execution_node(node)  # ← Calls _build_code_execution_node
```

**Result:** They call DIFFERENT functions!

---

## ⚙️ agent_only Path: `_build_agent_node()` (Lines 1853-1952+)

### What It Does:
1. **Expects code to ALREADY exist** in `node.code`
2. **NO LLM call at runtime**
3. Retrieves TaskWeaver session
4. Injects previous step outputs as variables
5. **Executes the pre-generated code** via TaskWeaver

### Key Code Snippet:
```python
# Line 1886-1891: Check if code exists
has_code = bool(
    hasattr(node, 'code') and 
    node.code and 
    node.code.strip()
)

# Line 1896-1901: Execute existing code
if has_code and HAS_TASKWEAVER:
    self.logger.info(f"🔨 [TW] Executing code for step {node.step_number}")
    code = node.code  # ← Code already exists!
    
    # Line 1904-1908: Get TaskWeaver session
    session = await sync_to_async(self._get_taskweaver_session)(...)
    
    # Line 1912-1951: Inject variables from state
    session_vars = {}
    for node_id, output_data in step_outputs.items():
        session_vars[f'step_{step_num}_result'] = output_data
    
    # Execute via TaskWeaver (not shown, but calls session.send_message(code))
```

**Summary:** Executes **PRE-GENERATED** code (from workflow generation phase)

---

## 💻 code_execution Path: `_build_code_execution_node()` (Lines 2515-2614+)

### What It Does:
1. **Code does NOT exist yet**
2. **Calls LLM at runtime** to generate code
3. Generates code based on task description and context
4. Cleans up markdown formatting
5. **Executes the dynamically generated code**

### Key Code Snippet:
```python
# Line 2544-2549: Broadcast code generation starting
await self._broadcast_progress("code_generation_started_event", {
    "step": node.step_number,
    "task": node.description or node.node_name
})

# Line 2556-2557: Get LLM
from apps.py_workflows.shared.llm.factory import get_langchain_model
llm = get_langchain_model(temperature=0.0)

# Line 2559-2595: Build system prompt for code generation
system_prompt = """You are an expert Python programmer. Generate safe, production-quality Python code.

**TASK:** {task}
**CONTEXT:** {context}

**REQUIREMENTS:**
1. Write clean, well-documented Python code
2. Use only standard library imports
3. Store results in a variable called `result`
...
"""

# Line 2602-2603: CALL LLM TO GENERATE CODE
response = await llm.ainvoke(messages)
generated_code = response.content  # ← Code generated NOW at runtime!

# Line 2605-2609: Clean up markdown
if "```python" in generated_code:
    generated_code = generated_code.split("```python")[1].split("```")[0].strip()

# Then execute (lines continue...)
```

**Summary:** **GENERATES** code at runtime using LLM, then executes it

---

## 📊 Side-by-Side Comparison

| Aspect | `agent_only` | `code_execution` |
|--------|-------------|------------------|
| **Function Called** | `_build_agent_node()` | `_build_code_execution_node()` |
| **Code Source** | Pre-generated (exists in `node.code`) | Generated at runtime by LLM |
| **LLM Call at Runtime?** | ❌ NO | ✅ YES |
| **When Code Created** | During workflow generation | During workflow execution |
| **Execution Engine** | TaskWeaver session | Dynamic exec/eval (likely) |
| **Use Case** | Known transformations (planned ahead) | Dynamic/adaptive tasks (decide at runtime) |

---

## 🎯 CONCLUSION: They ARE Different!

### agent_only:
- **"Pre-compiled"** approach
- Code generated once during workflow creation
- Fast execution (no LLM overhead)
- Predictable behavior

### code_execution:
- **"Just-in-time compilation"** approach  
- Code generated fresh each execution
- Slower (LLM latency)
- Adaptive to runtime context

---

## ✅ RECOMMENDATION: Keep Both

They serve **fundamentally different purposes**:

1. **agent_only**: Use for workflows where you know the logic upfront
   - Example: "Analyze sentiment of customer feedback"
   - Code generated: Once during workflow generation
   - Execution: Fast, deterministic

2. **code_execution**: Use for dynamic/adaptive scenarios
   - Example: "Process data in a format unknown until runtime"
   - Code generated: Fresh each execution based on actual data
   - Execution: Slower but flexible

---

## 🔧 FIXES NEEDED:

1. ✅ Remove "deprecated" comment from schema_builder.py
2. ✅ Document the distinction clearly in prompts
3. ✅ Add cross-reference comments explaining:
   - agent_only = pre-generated code execution
   - code_execution = runtime code generation + execution
4. ⚠️ **DO NOT merge them** - they have different purposes!
