# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin validation utilities."""

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def validate_plugin(plugin_path: str | Path) -> list[str]:
    """Validate plugin structure and manifest. Returns list of errors (empty if valid)."""
    plugin_dir = Path(plugin_path)
    errors = []

    if not plugin_dir.exists():
        return [f"Plugin not found: {plugin_dir}"]

    manifest = plugin_dir / "pyproject.toml"
    if not manifest.exists():
        errors.append("Missing pyproject.toml")
    else:
        try:
            data = tomllib.loads(manifest.read_text())
            lf = data.get("tool", {}).get("lichtfeld", {})
            if not lf:
                errors.append("pyproject.toml: missing [tool.lichtfeld] section")
            project = data.get("project", {})
            for field in ("name", "version", "description"):
                if field not in project:
                    errors.append(f"pyproject.toml: missing project.{field}")
            if "hot_reload" not in lf:
                errors.append("pyproject.toml: missing tool.lichtfeld.hot_reload")
        except Exception as e:
            errors.append(f"pyproject.toml: {e}")

    init_py = plugin_dir / "__init__.py"
    if not init_py.exists():
        errors.append("Missing __init__.py")
    else:
        content = init_py.read_text()
        if "def on_load" not in content:
            errors.append("Missing on_load() function")
        if "def on_unload" not in content:
            errors.append("Missing on_unload() function")

    errors.extend(_check_venv(plugin_dir))

    return errors


def _check_venv(plugin_dir: Path) -> list[str]:
    """Check venv health for a plugin directory."""
    errors = []
    venv = plugin_dir / ".venv"

    has_deps = (plugin_dir / "pyproject.toml").exists() and \
        any((plugin_dir / "pyproject.toml").read_text().find(s) >= 0
            for s in ("dependencies",))

    if not venv.exists():
        if has_deps:
            errors.append("venv: missing — run plugin install to create")
        return errors

    if sys.platform == "win32":
        python = venv / "Scripts" / "python.exe"
    else:
        python = venv / "bin" / "python"

    if not python.exists():
        errors.append(f"venv: broken — missing {python.relative_to(plugin_dir)}")
        return errors

    stamp = venv / ".deps_installed"
    if has_deps and not stamp.exists():
        errors.append("venv: dependencies not installed")

    return errors
