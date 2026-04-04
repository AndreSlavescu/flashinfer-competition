"""Local bootstrap helpers for repo scripts.

These helpers only use the standard library so entrypoints can fail with a
clear setup message before importing optional runtime dependencies.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BOOTSTRAP_COMMAND = ".venv/bin/python -m pip install -r requirements-agent-loop.txt"


def require_dependencies(
    requirements: dict[str, str],
    *,
    entrypoint: str,
    include_env_hint: bool = False,
) -> None:
    """Exit with a concise setup message when required modules are missing."""
    missing: list[tuple[str, str]] = []
    for module_name, install_name in requirements.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, install_name))

    if not missing:
        return

    print(f"Missing Python dependencies for {entrypoint}:")
    for module_name, install_name in missing:
        print(f"  - import '{module_name}' (install via {install_name})")
    print()
    print("Bootstrap this repo with:")
    print(f"  {BOOTSTRAP_COMMAND}")
    if include_env_hint:
        print("Then populate OPENAI_API_KEY in .env (see .env.example).")
    raise SystemExit(1)


def load_repo_dotenv() -> None:
    """Load .env from the repo root after dependency checks pass."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
