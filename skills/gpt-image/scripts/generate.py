#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Skill launcher for the shared gpt-image CLI.

Resolution order:
1. Full plugin install: import ../../../src/gpt_image_cli/cli.py.

This avoids ambient Python/PATH execution and preserves one canonical local
implementation for the plugin CLI.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

def _import_local_main():
    """Return main() from the plugin's local src/gpt_image_cli/cli.py only."""
    script_path = Path(__file__).resolve()
    if len(script_path.parents) <= 3:
        return None
    cli_path = script_path.parents[3] / "src" / "gpt_image_cli" / "cli.py"
    if not cli_path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("gpt_image_cli.cli", cli_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "main", None)


def main() -> int:
    cli_main = _import_local_main()
    if cli_main is not None:
        return int(cli_main() or 0)

    print(
        "error: could not find the local gpt-image CLI backend. Reinstall the plugin with its local src tree, then retry the same command.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
