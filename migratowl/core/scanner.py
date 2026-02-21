"""Dependency manifest scanning and parsing for multiple ecosystems."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from migratowl.models.schemas import Dependency, Ecosystem

# Mapping of manifest filenames to their ecosystem.
MANIFEST_PATTERNS: dict[str, Ecosystem] = {
    "requirements.txt": Ecosystem.PYTHON,
    "pyproject.toml": Ecosystem.PYTHON,
    "Pipfile": Ecosystem.PYTHON,
    "package.json": Ecosystem.NODEJS,
}

# Directories to skip when walking the project tree.
_SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
    }
)

# Regex for extracting name and version from a PEP 508 dependency specifier.
# Handles: pkg==1.0, pkg>=1.0, pkg~=1.0, pkg>=1.0,<2 (takes first version).
_PEP508_RE = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"  # package name
    r"\s*(?:~=|==|>=|<=|!=|>|<)"  # first operator
    r"\s*([0-9][0-9A-Za-z.*]*)"  # version
)

# Regex for requirements.txt lines (simpler, allows extras like pkg[extra]==1.0).
_REQ_LINE_RE = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"  # package name
    r"(\[[^\]]*\])?"  # optional extras
    r"\s*(~=|==|>=|<=|!=|>|<)"  # operator
    r"\s*([0-9][0-9A-Za-z.*]*)"  # version
)

# Regex for npm version strings: ^4.18.0, ~4.17.21, >=1.0.0, 18.2.0
_NPM_VERSION_RE = re.compile(r"[\^~>=<]*\s*(\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?)")


async def scan_project(project_path: str | Path) -> list[Dependency]:
    """Walk the project tree, find manifest files, and parse them all."""
    root = Path(project_path)
    all_deps: list[Dependency] = []

    for dirpath, dirnames, filenames in root.walk():
        # Prune skipped directories in-place so walk() doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        for fname in filenames:
            if fname in MANIFEST_PATTERNS:
                filepath = dirpath / fname
                parser = _PARSERS.get(fname)
                if parser:
                    deps = await parser(filepath)
                    all_deps.extend(deps)

    # Deduplicate by (name, current_version) — multiple manifests (e.g. requirements.txt + Pipfile) often list the same packages.
    seen: set[tuple[str, str]] = set()
    unique: list[Dependency] = []
    for dep in all_deps:
        key = (dep.name.lower(), dep.current_version)
        if key not in seen:
            seen.add(key)
            unique.append(dep)
    return unique


async def _parse_requirements_txt(path: Path) -> list[Dependency]:
    """Parse a requirements.txt file into Dependency objects."""
    deps: list[Dependency] = []
    text = path.read_text(encoding="utf-8")

    for line in text.splitlines():
        line = line.strip()
        # Skip comments, blank lines, and -r/-c includes.
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        match = _REQ_LINE_RE.match(line)
        if match:
            name = match.group(1)
            version = match.group(5)
            deps.append(
                Dependency(
                    name=name,
                    current_version=version,
                    ecosystem=Ecosystem.PYTHON,
                    manifest_path=str(path),
                )
            )

    return deps


async def _parse_pyproject_toml(path: Path) -> list[Dependency]:
    """Parse a pyproject.toml file, extracting dependencies and optional-dependencies."""
    deps: list[Dependency] = []
    text = path.read_bytes()
    data = tomllib.loads(text.decode("utf-8"))

    project = data.get("project", {})
    if not project:
        return deps

    # Main dependencies.
    dep_strings: list[str] = project.get("dependencies", [])

    # Optional dependencies (all groups).
    optional = project.get("optional-dependencies", {})
    for group_deps in optional.values():
        dep_strings.extend(group_deps)

    for dep_str in dep_strings:
        dep_str = dep_str.strip()
        match = _PEP508_RE.match(dep_str)
        if match:
            name = match.group(1)
            version = match.group(3)
            deps.append(
                Dependency(
                    name=name,
                    current_version=version,
                    ecosystem=Ecosystem.PYTHON,
                    manifest_path=str(path),
                )
            )

    return deps


async def _parse_pipfile(path: Path) -> list[Dependency]:
    """Parse a Pipfile (TOML format) into Dependency objects."""
    deps: list[Dependency] = []
    text = path.read_bytes()
    data = tomllib.loads(text.decode("utf-8"))

    for section in ("packages", "dev-packages"):
        packages = data.get(section, {})
        for name, version_spec in packages.items():
            if isinstance(version_spec, dict):
                version_spec = version_spec.get("version", "*")
            version_spec = str(version_spec).strip().strip('"').strip("'")

            # Skip wildcard / unversioned.
            if version_spec == "*":
                continue

            # Extract version number from specifier.
            match = re.search(r"(\d+[0-9A-Za-z.*]*)", version_spec)
            if match:
                deps.append(
                    Dependency(
                        name=name,
                        current_version=match.group(1),
                        ecosystem=Ecosystem.PYTHON,
                        manifest_path=str(path),
                    )
                )

    return deps


async def _parse_package_json(path: Path) -> list[Dependency]:
    """Parse a package.json file, extracting dependencies and devDependencies."""
    deps: list[Dependency] = []
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    for section in ("dependencies", "devDependencies"):
        packages = data.get(section, {})
        for name, version_spec in packages.items():
            match = _NPM_VERSION_RE.match(version_spec)
            if match:
                deps.append(
                    Dependency(
                        name=name,
                        current_version=match.group(1),
                        ecosystem=Ecosystem.NODEJS,
                        manifest_path=str(path),
                    )
                )

    return deps


# Dispatcher mapping filename to parser function.
_PARSERS = {
    "requirements.txt": _parse_requirements_txt,
    "pyproject.toml": _parse_pyproject_toml,
    "Pipfile": _parse_pipfile,
    "package.json": _parse_package_json,
}
