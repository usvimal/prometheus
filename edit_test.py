#!/usr/bin/env python3
import pathlib
content = pathlib.Path("tests/test_smoke.py").read_text()
old = '"mcp_list_resources", "mcp_read_resource",'
new = '''"mcp_list_resources", "mcp_read_resource",
    # File watcher (v6.6.0)
    "watch_start", "watch_check", "watch_stop", "watch_list",'''
content = content.replace(old, new)
pathlib.Path("tests/test_smoke.py").write_text(content)
print("Done")
