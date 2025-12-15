# Node Type Consistency Audit
**Generated:** Dec 14, 2025
**Purpose:** Cross-reference all node type definitions across the codebase

---

## 📊 Node Type Matrix (UPDATED Dec 14, 2025)

| Node Type | Planner Prompt | Schema Builder | Code Gen Prompt | Execution | Status |
|-----------|---------------|----------------|-----------------|-----------|--------|
| **form** | ✅ `<form>` marker | ✅ enum: ["form"] | ✅ Used in examples | ✅ _build_form_node | ✅ CONSISTENT |
| **hitl** | ✅ `<hitl>` marker | ✅ enum: ["hitl"] | ✅ Used in examples | ✅ _build_form_node | ✅ CONSISTENT |
| **agent_with_tools** | ✅ `<agent_with_tools>` | ✅ enum: ["agent_with_tools"] | ✅ Used in examples | ✅ _build_composio_tool_node | ✅ CONSISTENT |
| **agent_only** | ✅ `<agent_only>` marker | ✅ enum: ["agent_only"] + agent_mode | ✅ TWO modes documented | ✅ _build_agent_node (dual-mode) | ✅ **RESOLVED!** |
| **code_execution** | ❌ NOT used | ⚠️ DEPRECATED | ❌ NOT in node types | ✅ _build_code_execution_node | ⚠️ **DEPRECATED** |
| **loop** | ✅ `<loop>` marker | ✅ enum: ["loop"] | ✅ Used in examples | ✅ loop_executor | ✅ CONSISTENT |
| **parallel** | ✅ `<parallel>` marker | ✅ enum: ["parallel"] | ✅ Used in examples | ✅ _build_parallel_node | ✅ CONSISTENT |

---

## ✅ RESOLUTIONS (Dec 14, 2025)

### Resolution: agent_only Unified with Dual Modes

**Previous Conflict:** agent_only marked as "deprecated" but actively used everywhere

**Solution Implemented:**
1. ✅ **agent_only is now the PRIMARY node type** for AI processing
2. ✅ Added `agent_mode` field with two values:
   - `reasoning`: Pure LLM call for analysis/recommendations
   - `code`: Execute TaskWeaver-generated Python code
3. ✅ Updated execution path in `langgraph_adapter.py._build_agent_node()` to handle both modes
4. ✅ Added cross-reference comments across all 3 files
5. ✅ Marked `code_execution` as deprecated (use agent_only with agent_mode='code')

---

## 🚨 ARCHIVED CONFLICTS (RESOLVED)

### Conflict 1: agent_only vs code_execution (RESOLVED)

**workflow_schema_builder.py (Line 569-573):**
```python
# agent_only - Deprecated, but keep for compatibility
# ===================================================================
node_schemas.append({
    "type": "object",
    "description": "Agent-only node (deprecated, use code_execution)",
    ...
})
```

**BUT:**

**planner_prompt.yaml (Lines 77-79):**
```yaml
- `<agent_only>` - Pure AI processing WITHOUT external tools (analyze, rank, draft, transform, summarize)
  * Use when: AI needs to process/analyze/transform data without calling external APIs
  * Example: "4. Analyze search results and rank by relevance <agent_only>"
```

**code_generator_prompt.yaml (Lines 408-414):**
```yaml
**"agent_only"** (Pure AI Processing):
* Use when: Need AI to analyze, transform, draft, or process data WITHOUT calling external APIs
* Has NO tool_id: Pure AI reasoning/generation
* Behavior: AI processes input, generates output (text, analysis, transformation)
* Example: "Prepare draft responses", "Analyze sentiment", "Transform data format"
```

**STATUS:** 🚨 **CONTRADICTORY!**
- Schema says: "deprecated, use code_execution"
- Planner & Code Generator: Actively use and teach agent_only
- No mention of code_execution in prompts

---

### Conflict 2: code_execution Missing from Pipeline

**workflow_schema_builder.py (Line 520-560):**
```python
# code_execution - Python code node
# ===================================================================
node_schemas.append({
    "type": "object",
    "description": "Python code execution node",
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["code_execution"]},
        "code": {"type": "string", ...}
    }
})
```

**BUT:**
- ❌ NOT in planner_prompt.yaml (no `<code_execution>` marker)
- ❌ NOT in code_generator_prompt.yaml node types section
- ✅ Exists in workflow_schema.py and workflow_ir.py

**STATUS:** 🚨 **INCOMPLETE PIPELINE!**
- Planner can't mark steps as `<code_execution>`
- Code Generator doesn't know when to use it
- Schema validates it, but upstream doesn't generate it

---

## 🤔 Analysis: What's the Truth?

### Evidence for agent_only being ACTIVE (not deprecated):

1. **Planner actively teaches it:**
   - Line 77: `<agent_only>` marker definition
   - Lines 221, 224, 232, 237, 259-261, 274, 291-292, 294-295: 10+ examples using it

2. **Code Generator uses it:**
   - Line 196: Listed as valid node type
   - Lines 206, 328, 338: Examples with agent_only
   - Lines 408-414: Full documentation section

3. **Workflow IR supports it:**
   - workflow_schema.py: Line 31 defines it
   - workflow_ir.py: Processes it

### Evidence for code_execution being ACTIVE:

1. **Schema defines it:** workflow_schema_builder.py Line 520
2. **IR supports it:** workflow_schema.py, workflow_ir.py
3. **Used in instructor generator:** instructor_workflow_generator.py Lines 290-380

### Hypothesis:

**TWO DIFFERENT USE CASES:**
- `agent_only`: High-level AI reasoning/analysis (no code, just natural language processing)
- `code_execution`: Explicit Python code execution

**The deprecation comment is WRONG!** They serve different purposes.

---

## ✅ RECOMMENDED RESOLUTION

### Option A: Keep Both (RECOMMENDED)

**Rationale:** They serve different purposes
- `agent_only`: AI reasoning (e.g., "Analyze sentiment", "Rank results")
- `code_execution`: Explicit Python code (e.g., data transformation with code blocks)

**Actions:**
1. ✅ Remove "deprecated" comment from schema_builder.py
2. ✅ Add `<code_execution>` marker to planner_prompt.yaml (for advanced users)
3. ✅ Document the distinction in all 3 files
4. ✅ Add cross-references

### Option B: Deprecate agent_only (NOT RECOMMENDED)

**Rationale:** Simplify to single code execution type

**Actions:**
1. ❌ Remove `<agent_only>` from planner_prompt.yaml
2. ❌ Replace all agent_only examples with code_execution
3. ❌ Add migration path in schema_builder.py
4. ❌ Update all documentation

**Problem:** This would require:
- Rewriting 10+ examples in planner_prompt.yaml
- Updating code_generator_prompt.yaml
- Breaking existing workflows

---

## 📝 NEXT STEPS

1. **Decide:** Keep both or deprecate one?
2. **Update comments** in workflow_schema_builder.py
3. **Add cross-references** across all 3 files
4. **Document the distinction** clearly

---

## Cross-Reference Locations

### File 1: planner_prompt.yaml
- **Purpose:** INPUT - Human-readable plan with markers
- **Lines 64-141:** Node type marker definitions
- **Lines 170-300:** Examples with all markers
- **Lines 617-658:** Critical reminders

### File 2: workflow_schema_builder.py
- **Purpose:** OUTPUT - JSON schema validation
- **Lines 490-763:** All node schema definitions
- **Lines 569-573:** ⚠️ agent_only "deprecated" comment

### File 3: code_generator_prompt.yaml
- **Purpose:** TRANSFORM - LLM instructions for code generation
- **Lines 192-197:** Node type definitions
- **Lines 198-370:** Examples and patterns
- **Lines 408-414:** agent_only documentation
