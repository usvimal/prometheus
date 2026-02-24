#!/usr/bin/env python3
"""Fix system role issue in loop.py"""

import re

with open('/home/vimal2/prometheus/repo/prometheus/loop.py', 'r') as f:
    content = f.read()

# Replace all "role": "system" with "role": "user" in messages.append
content = content.replace('"role": "system"', '"role": "user"')

with open('/home/vimal2/prometheus/repo/prometheus/loop.py', 'w') as f:
    f.write(content)

print("Fixed all instances!")
