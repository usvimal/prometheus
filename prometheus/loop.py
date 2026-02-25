            # Compact old tool history when needed
            # Check for LLM-requested compaction first (via compact_context tool)
            pending_compaction = getattr(tools._ctx, '_pending_compaction', None)
            if pending_compaction is not None:
                messages = compact_tool_history_llm(messages, keep_recent=pending_compaction)
                tools._ctx._pending_compaction = None
            elif round_idx > 6:
                # Aggressive compaction: keep 8 recent rounds (was 6 at round 8)
                # Prevents max_tokens failures in long-running tasks
                messages = compact_tool_history(messages, keep_recent=8)
            elif round_idx > 2:
                # Light compaction: trigger earlier (was round 3, >60 messages)
                if len(messages) > 40:
                    messages = compact_tool_history(messages, keep_recent=8)
