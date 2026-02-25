#!/usr/bin/env python3
import subprocess
content = subprocess.check_output(['git', '-C', '/home/vimal2/prometheus/repo', 'show', 'd2d1bea:prometheus/loop.py'])
with open('/home/vimal2/prometheus/repo/prometheus/loop.py', 'wb') as f:
    f.write(content)
print('Done')
