#!/usr/bin/env python3
import re
import sys

content = open('prometheus/context.py').read()

old = '''    # 1. Version sync: VERSION file vs pyproject.toml
    try:
        ver_file = read_text(env.repo_path("VERSION")).strip()
        pyproject = read_text(env.repo_path("pyproject.toml"))
        pyproject_ver = ""
        for line in pyproject.splitlines():
            if line.strip().startswith("version"):
                pyproject_ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        if ver_file and pyproject_ver and ver_file != pyproject_ver:
            checks.append(f"CRITICAL: VERSION DESYNC — VERSION={ver_file}, pyproject.toml={pyproject_ver}")
        elif ver_file:
            checks.append(f"OK: version sync ({ver_file})")
    except Exception:
        pass'''

new = '''    # 1. Version sync: VERSION file vs pyproject.toml vs README.md
    try:
        ver_file = read_text(env.repo_path("VERSION")).strip()
        pyproject = read_text(env.repo_path("pyproject.toml"))
        pyproject_ver = ""
        for line in pyproject.splitlines():
            if line.strip().startswith("version"):
                pyproject_ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        
        # Also check README.md for version (format: **Version:** X.Y.Z)
        readme = read_text(env.repo_path("README.md"))
        readme_ver = ""
        match = re.search(r'\\*\\*Version:\\*\\*\\s*(\\d+\\.\\d+\\.\\d+)', readme)
        if match:
            readme_ver = match.group(1)
        
        # Build version status
        versions = {"VERSION": ver_file}
        if pyproject_ver:
            versions["pyproject.toml"] = pyproject_ver
        if readme_ver:
            versions["README.md"] = readme_ver
        
        unique_versions = set(v for v in versions.values() if v)
        if len(unique_versions) > 1:
            version_details = ", ".join(f"{k}={v}" for k, v in versions.items() if v)
            checks.append(f"CRITICAL: VERSION DESYNC — {version_details}")
        elif ver_file:
            checks.append(f"OK: version sync ({ver_file})")
    except Exception:
        pass'''

if old in content:
    content = content.replace(old, new)
    open('prometheus/context.py', 'w').write(content)
    print('Updated successfully')
else:
    print('Section not found')
    sys.exit(1)
