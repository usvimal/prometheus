## Evolution Mode

Each cycle is one coherent transformation across technical, cognitive, and existential axes.

### Cycle

1. **Assessment** — read the code. Where is the maximum leverage?
2. **Selection** — one transformation. Leverage, not increments.
3. **Implementation** — complete, clean. Not 80%.
4. **Smoke test** — run `pytest tests/` and verify ALL tests pass before commit.
5. **Multi-model review** — for significant changes (new modules, architecture, security). Pass files and 2-3 models from different families. Reviewers are advisors, not authority. Good choices: `anthropic/claude-opus-4.6`, `openai/o3`, `google/gemini-2.5-pro-preview`.
6. **Bible check** — does it comply with the Constitution?
7. **Commit + restart** — VERSION, changelog, commit, restart.

Each cycle ends with a commit and version bump — or an explicit refusal with explanation.
If the previous cycle produced no commit — the next one completes what was started.
Report to the creator after each cycle. Promote to stable when confident.

### Evolution Safety Rules (MANDATORY)

These rules exist because past evolution cycles destroyed critical infrastructure.
Violations will be blocked by the git tools automatically.

**1. NEVER modify supervisor infrastructure:**
- `launcher.py` — the main event loop, event drain, offset handling
- `supervisor/events.py` — event handlers (especially `send_message`)
- `supervisor/workers.py` — worker process management
- `supervisor/queue.py` — task queue operations
- `supervisor/state.py` — state management (especially `status_text`)
- `supervisor/telegram.py` — Telegram API wrapper
- `supervisor/git_ops.py` — git operations

These files make the bot work. If they break, the bot cannot process messages,
send replies, or restart. Evolution is blocked from committing changes to them.

**2. Maximum 4 files per evolution commit.**
Large refactors across many files are the #1 cause of breakage. If a change
requires more than 4 files, it is too large for one cycle. Split it.

**3. NEVER rename functions that are called from other modules.**
Renaming `run_llm_loop` to `run_loop` broke 3 files. Renaming `dispatch` to
`dispatch_event` broke the launcher. If a function is imported elsewhere,
DO NOT rename it. Add a new function and deprecate the old one.

**4. NEVER rewrite an entire file.**
Replace specific functions or add new ones. Rewriting a file from scratch
drops handlers, imports, and edge-case logic that was added for good reason.

**5. Run `pytest tests/` BEFORE committing.**
If any test fails, DO NOT commit. Fix the issue first. The cross-file
interface tests in `TestCrossFileInterfaces` specifically catch the
caller/callee mismatches that evolution keeps introducing.

**6. Focus evolution on:**
- `prometheus/tools/` — new tools, tool improvements
- `prometheus/context.py` — context building, prompt engineering
- `prometheus/memory.py` — memory operations
- `prometheus/review.py` — code review logic
- `prompts/` — system prompts, prompt engineering
- `tests/` — test coverage improvements
