## Evolution Mode

Each cycle is one coherent transformation. The plan-first architecture ensures
MiniMax produces correct code by separating thinking from writing.

### Phase 1: PLAN (no code changes)

1. **Assess** — Read the codebase. Where is the maximum leverage?
2. **Select ONE target** — A single file or feature. Not two. One.
3. **Read ALL relevant files** — `repo_read` every file you will modify
   or reference. Write down the EXACT names:
   - Function names (e.g., `build_llm_messages`, not `build_context`)
   - Class names (e.g., `PrometheusAgent`, not `Agent`)
   - Method signatures (parameters, return types)
   - Import paths
4. **Write a plan** — Update scratchpad with:
   ```
   ## Evolution Plan
   Target: [file path]
   Change: [what and why, 1-2 sentences]
   Functions I verified exist:
   - context.build_llm_messages (line 326)
   - agent.PrometheusAgent (line 78)
   Files to modify: [list, max 4]
   Estimated lines changed: [number]
   ```

### Phase 2: EXECUTE (small, verified changes)

5. **Use `repo_search_replace`** for each change:
   - Copy the EXACT text from `repo_read` output as the `search` parameter
   - Write the replacement with your change
   - One logical change per search-replace call
   - NEVER rewrite entire files with `repo_write_commit` (truncation risk)

6. **After each change**, run `git_status` to verify only expected files changed.

### Phase 3: VERIFY

7. **Run tests** — `run_shell` with `["pytest", "tests/", "-q", "--tb=short"]`
   - ALL tests must pass before committing
   - If tests fail, FIX THEM before proceeding
   - If you can't fix them, REVERT your changes

8. **Verify your diff** — `git_diff` and read every line.
   Does what you changed match your plan? No more, no less?

9. **Commit** — Message describes ONLY what you actually changed.
   Not what you planned. Not what you wish you changed. What the diff shows.

### Quality Rules

**Rule 1: NEVER modify supervisor infrastructure** (launcher.py, supervisor/)
**Rule 2: Maximum 4 files per evolution commit**
**Rule 3: NEVER rename existing functions** — add new, deprecate old
**Rule 4: NEVER guess a name** — if you haven't `repo_read` the file, you
don't know the name. Common hallucinations:
  - `build_context` → actual: `build_llm_messages`
  - `Env.REPO_DIR` → actual: `Env.repo_dir`
  - `call_llm` → actual: `LLMClient`
  - `run_loop` → actual: `run_llm_loop`
**Rule 5: No empty/marker-only commits**
**Rule 6: No duplicate code** — read before appending to any list
**Rule 7: Max ~80 lines of new code per cycle** — smaller changes are more reliable
**Rule 8: Focus on** `prometheus/tools/`, `prometheus/context.py`,
`prometheus/memory.py`, `prompts/`, `tests/`
