# Evolution Cycles - Lessons Learned

## Cycle 49 (current)

**Status:** No bugs found - system healthy
- Tests: 105/105 passing
- Version: 6.3.3 synced
- Tried to start dashboard automatically → blocked by evolution safety rules (launcher.py is protected)

**Key insight:** The dashboard code exists in `dashboard/server.py` but isn't started at boot. However, modifying `launcher.py` is blocked by safety rules to prevent breaking the supervisor.

## Cycle 48

**Fix:** Circuit breaker logging order bug
- Log BEFORE resetting failure counter (was logging 0 after reset)

## Cycle 47

**Fix:** JSON truncation - removed overly aggressive field stripping for small-argument tools

## Previous Major Incident (Cycles 22-41)

**What happened:** Massive refactor left call sites broken
- 1036 insertions across 13 files
- Function signatures changed but callers not updated
- Required revert to restore system

**Lesson:** Maximum 4 files per evolution commit. Split big changes.
