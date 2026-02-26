"""Skill system tools: user-definable instruction packages for the agent.

Skills are markdown files (SKILL.md) with YAML frontmatter stored in
~/prometheus/data/skills/<name>/SKILL.md. They extend the agent's
capabilities through natural language instructions injected into the
LLM system prompt.

Follows the OpenClaw/AgentSkills pattern: file-based, YAML frontmatter,
two-stage prompt injection (summary + on-demand full read).
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolEntry, ToolContext

log = logging.getLogger(__name__)

SKILLS_DIR = "skills"
SKILL_FILENAME = "SKILL.md"

# --- Validation ---

_VALID_NAME = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,58}[a-zA-Z0-9]$|^[a-zA-Z0-9]$')
_RESERVED = frozenset({"_index", "con", "prn", "aux", "nul", "__pycache__"})


def _sanitize_name(name: str) -> str:
    """Validate and sanitize skill name. Raises ValueError on bad input."""
    if not name or not isinstance(name, str):
        raise ValueError("Skill name must be a non-empty string")
    name = name.strip().lower()
    if '/' in name or '\\' in name or '..' in name:
        raise ValueError(f"Invalid characters in skill name: {name}")
    if not _VALID_NAME.match(name):
        raise ValueError(
            f"Invalid skill name: {name}. "
            "Use lowercase alphanumeric, hyphens, underscores (2-60 chars)."
        )
    if name in _RESERVED:
        raise ValueError(f"Reserved skill name: {name}")
    return name


def _safe_skill_dir(ctx: ToolContext, name: str) -> tuple:
    """Build and verify skill directory path is within skills directory.

    Returns:
        tuple[Path, str]: (skill_dir_path, sanitized_name)
    """
    sanitized = _sanitize_name(name)
    skills_root = ctx.drive_path(SKILLS_DIR)
    skill_dir = skills_root / sanitized
    resolved = skill_dir.resolve()
    root_resolved = skills_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"Path escape detected: {name}")
    return skill_dir, sanitized


# --- YAML Frontmatter Parsing (no pyyaml dependency) ---

def parse_skill_file(text: str) -> Dict[str, Any]:
    """Parse SKILL.md content: extract YAML frontmatter + markdown body.

    Returns dict with: name, description, enabled, auto_invoke, version, body
    """
    result = {
        "name": "",
        "description": "",
        "enabled": True,
        "auto_invoke": True,
        "version": 1,
        "body": text,
    }

    stripped = text.strip()
    if not stripped.startswith("---"):
        return result

    # Find closing ---
    end_idx = stripped.find("---", 3)
    if end_idx == -1:
        return result

    frontmatter = stripped[3:end_idx].strip()
    body = stripped[end_idx + 3:].strip()
    result["body"] = body

    # Parse simple YAML key: value pairs
    for line in frontmatter.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()

        # Remove quotes
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        if key == "name":
            result["name"] = val
        elif key == "description":
            result["description"] = val
        elif key == "enabled":
            result["enabled"] = val.lower() in ("true", "yes", "1", "on")
        elif key in ("auto_invoke", "auto-invoke", "autoinvoke"):
            result["auto_invoke"] = val.lower() in ("true", "yes", "1", "on")
        elif key == "version":
            try:
                result["version"] = int(val)
            except ValueError:
                pass

    return result


def build_skill_frontmatter(name: str, description: str,
                            enabled: bool = True, auto_invoke: bool = True,
                            version: int = 1) -> str:
    """Build YAML frontmatter string for a SKILL.md file."""
    return (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"enabled: {'true' if enabled else 'false'}\n"
        f"auto_invoke: {'true' if auto_invoke else 'false'}\n"
        f"version: {version}\n"
        f"---\n"
    )


# --- Skill scanning (used by context.py and dashboard) ---

def scan_skills(drive_root: Path) -> List[Dict[str, Any]]:
    """Scan all skills and return parsed metadata + body.

    Returns list of dicts with: name, description, enabled, auto_invoke,
    version, body, path.
    """
    skills_root = drive_root / SKILLS_DIR
    if not skills_root.exists():
        return []

    results = []
    for entry in sorted(skills_root.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / SKILL_FILENAME
        if not skill_file.exists():
            continue
        try:
            text = skill_file.read_text(encoding="utf-8")
            parsed = parse_skill_file(text)
            # Use directory name as canonical name if frontmatter name is empty
            if not parsed["name"]:
                parsed["name"] = entry.name
            parsed["path"] = str(skill_file)
            parsed["dir_name"] = entry.name
            results.append(parsed)
        except Exception:
            log.debug(f"Failed to parse skill: {entry.name}", exc_info=True)
            continue

    return results


# --- Tool handlers ---

def _skill_list(ctx: ToolContext) -> str:
    """List all skills with metadata."""
    skills = scan_skills(ctx.drive_root)
    if not skills:
        return "No skills installed. Use skill_create to add one."

    lines = [f"**{len(skills)} skill(s) installed:**\n"]
    for s in skills:
        status = "enabled" if s["enabled"] else "disabled"
        auto = " [auto]" if s["auto_invoke"] and s["enabled"] else ""
        lines.append(
            f"- **{s['name']}** ({status}{auto}): {s['description'][:100]}"
        )
    return "\n".join(lines)


def _skill_read(ctx: ToolContext, name: str) -> str:
    """Read full SKILL.md content for a skill."""
    try:
        skill_dir, sanitized = _safe_skill_dir(ctx, name)
    except ValueError as e:
        return f"⚠️ Invalid skill name: {e}"

    skill_file = skill_dir / SKILL_FILENAME
    if not skill_file.exists():
        return f"Skill '{sanitized}' not found. Use skill_list to see available skills."
    return skill_file.read_text(encoding="utf-8")


def _skill_create(ctx: ToolContext, name: str, description: str,
                  content: str, auto_invoke: bool = True) -> str:
    """Create a new skill with YAML frontmatter + content."""
    try:
        skill_dir, sanitized = _safe_skill_dir(ctx, name)
    except ValueError as e:
        return f"⚠️ Invalid skill name: {e}"

    if skill_dir.exists():
        return f"⚠️ Skill '{sanitized}' already exists. Use skill_update to modify it."

    # Create directory and SKILL.md
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = build_skill_frontmatter(sanitized, description, True, auto_invoke)
    skill_file = skill_dir / SKILL_FILENAME
    skill_file.write_text(frontmatter + "\n" + content, encoding="utf-8")
    return f"✅ Skill '{sanitized}' created."


def _skill_update(ctx: ToolContext, name: str, content: str) -> str:
    """Update an existing skill's body content (preserves frontmatter settings)."""
    try:
        skill_dir, sanitized = _safe_skill_dir(ctx, name)
    except ValueError as e:
        return f"⚠️ Invalid skill name: {e}"

    skill_file = skill_dir / SKILL_FILENAME
    if not skill_file.exists():
        return f"⚠️ Skill '{sanitized}' not found."

    # Parse existing to preserve frontmatter
    existing = parse_skill_file(skill_file.read_text(encoding="utf-8"))
    frontmatter = build_skill_frontmatter(
        existing["name"] or sanitized,
        existing["description"],
        existing["enabled"],
        existing["auto_invoke"],
        existing["version"],
    )
    skill_file.write_text(frontmatter + "\n" + content, encoding="utf-8")
    return f"✅ Skill '{sanitized}' updated."


def _skill_toggle(ctx: ToolContext, name: str, enabled: bool = True) -> str:
    """Enable or disable a skill."""
    try:
        skill_dir, sanitized = _safe_skill_dir(ctx, name)
    except ValueError as e:
        return f"⚠️ Invalid skill name: {e}"

    skill_file = skill_dir / SKILL_FILENAME
    if not skill_file.exists():
        return f"⚠️ Skill '{sanitized}' not found."

    existing = parse_skill_file(skill_file.read_text(encoding="utf-8"))
    frontmatter = build_skill_frontmatter(
        existing["name"] or sanitized,
        existing["description"],
        enabled,
        existing["auto_invoke"],
        existing["version"],
    )
    skill_file.write_text(frontmatter + "\n" + existing["body"], encoding="utf-8")
    status = "enabled" if enabled else "disabled"
    return f"✅ Skill '{sanitized}' {status}."


# --- Tool registration ---

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("skill_list", {
            "name": "skill_list",
            "description": (
                "List all installed skills with their name, description, "
                "and enabled/auto-invoke status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, _skill_list),
        ToolEntry("skill_read", {
            "name": "skill_read",
            "description": (
                "Read the full SKILL.md content for a specific skill. "
                "Use this to get detailed instructions before applying a skill."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (e.g. 'crypto-analysis', 'code-review')",
                    },
                },
                "required": ["name"],
            },
        }, _skill_read),
        ToolEntry("skill_create", {
            "name": "skill_create",
            "description": (
                "Create a new skill. Skills are instruction packages that teach you "
                "how to perform specific tasks. They get injected into your system prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (lowercase, hyphens/underscores allowed)",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what the skill does",
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown instructions for the skill",
                    },
                    "auto_invoke": {
                        "type": "boolean",
                        "description": "Include full content in system prompt automatically (default: true)",
                    },
                },
                "required": ["name", "description", "content"],
            },
        }, _skill_create),
        ToolEntry("skill_update", {
            "name": "skill_update",
            "description": "Update an existing skill's instruction content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "New markdown content for the skill",
                    },
                },
                "required": ["name", "content"],
            },
        }, _skill_update),
        ToolEntry("skill_toggle", {
            "name": "skill_toggle",
            "description": "Enable or disable a skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable, false to disable",
                    },
                },
                "required": ["name", "enabled"],
            },
        }, _skill_toggle),
    ]
