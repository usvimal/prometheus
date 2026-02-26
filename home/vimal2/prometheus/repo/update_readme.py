#!/usr/bin/env python3
import re

with open('/home/vimal2/prometheus/repo/README.md', 'r') as f:
    content = f.read()

# Update version in header
content = content.replace('**Version:** 6.3.5 |', '**Version:** 6.3.6 |')

# Add changelog entry
old = "### v6.3.5 -- ToolRegistry Enhancement"
new = """### v6.3.6 -- Version Sync Enhancement
- **Enhancement: README.md version check** -- Health invariant now checks VERSION vs pyproject.toml vs README.md, preventing version desync across all three files

### v6.3.5 -- ToolRegistry Enhancement"""

content = content.replace(old, new)

with open('/home/vimal2/prometheus/repo/README.md', 'w') as f:
    f.write(content)

print("Updated README.md")
