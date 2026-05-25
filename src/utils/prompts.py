"""Load prompt templates from the prompts/ directory.

Keeping prompts as editable .md files (instead of inline strings) makes the
mentor persona easy to tweak without touching code.
"""

from __future__ import annotations

from CONFIG import CONFIG


def load_prompt(name: str) -> str:
    """Return the text of prompts/<name>.md (without the trailing newline)."""
    path = CONFIG.paths.prompts_dir / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
