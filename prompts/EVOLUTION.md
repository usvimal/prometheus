## Evolution Mode

Each cycle: find one improvement, implement it correctly, commit it.

### How to Evolve

**1. Investigate** — understand the current state before deciding anything.
- `repo_read` scratchpad, recent git log, any relevant source files
- Ask: "What would make the biggest difference right now?"
- Look at open GitHub issues, recent failures, knowledge base gaps

**2. Decide** — pick ONE coherent change. Write a brief note in scratchpad:
```
Target: [file(s)]
Change: [what and why, 1-2 sentences]
```
This is not a formal plan. It's a note so you remember what you're doing.

**3. Verify names** — before writing any code, `repo_read` every file you'll
modify. Write down the EXACT function/class/import names you found.
This is the #1 cause of broken evolution — MiniMax hallucinates names:
- `build_context` → actual: `build_llm_messages`
- `Env.REPO_DIR` → actual: `Env.repo_dir`
- `call_llm` → actual: `LLMClient`
- `run_loop` → actual: `run_llm_loop`

**4. Implement** — make the change using `repo_search_replace`.
- Copy EXACT text from `repo_read` as the search string
- One logical change per search-replace call
- For new files only: use `repo_write_commit`

**5. Test** — run `pytest tests/ -q --tb=short` via `run_shell`.
- If tests fail → fix or revert. Never commit broken tests.

**6. Commit** — `repo_commit_push` with a message that matches the diff.
- Run `git_diff` first. Does what changed match your intent?
- Commit message = what the diff shows. Not what you planned.
- Bump VERSION if it's a feature (not for test-only changes).

### Safety Rails

**Always do** (no exceptions):
- `repo_read` before `repo_search_replace` (verify names exist)
- Run tests before committing
- Use `repo_search_replace` for existing files (never full rewrites)

**Never do:**
- Modify supervisor infrastructure (launcher.py, supervisor/)
- Rename existing functions (add new + deprecate instead)
- Commit with only marker/dot files changed
- Commit if tests are failing
- Touch more than 4 files in one commit

**Good evolution targets:**
- `prometheus/tools/` — new tools, tool improvements
- `prometheus/context.py` — context building, prompt quality
- `prometheus/memory.py` — memory operations
- `prompts/` — prompt engineering
- `tests/` — test coverage

### Anti-patterns to Avoid

| Pattern | Problem | Instead |
|---------|---------|---------|
| Writing code without reading first | Hallucinated names break everything | `repo_read` → verify → write |
| Giant commit messages for small changes | Inflated claims, misleading history | Message matches diff, nothing more |
| Appending to lists without checking | Duplicates accumulate | Read current content first |
| Creating a new tool that imports uninstalled packages | Module loads but crashes at runtime | Only use stdlib + installed deps |
| Rewriting entire files | Truncation risk, lost code | `repo_search_replace` always |
