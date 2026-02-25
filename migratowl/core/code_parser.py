"""Tree-sitter based code parser for dependency usage detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser

from migratowl.models.schemas import CodeUsage

logger = logging.getLogger(__name__)

# --- Extension to language mapping ---

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
}

# --- Tree-sitter query patterns for imports per language ---

IMPORT_QUERIES: dict[str, str] = {
    "python": """
(import_statement name: (dotted_name) @module)
(import_from_statement module_name: (dotted_name) @module)
""",
    "javascript": """
(call_expression
  function: (identifier) @_func
  arguments: (arguments (string) @module)
  (#eq? @_func "require"))
(import_statement source: (string) @module)
""",
    "typescript": """
(call_expression
  function: (identifier) @_func
  arguments: (arguments (string) @module)
  (#eq? @_func "require"))
(import_statement source: (string) @module)
""",
}

# Python-only: find call sites, base classes, and decorators for imported symbols.
_PYTHON_CALL_SITE_QUERY = """
(call function: (identifier) @func)
(class_definition superclasses: (argument_list (identifier) @base))
(decorator (identifier) @dec)
(decorator (call function: (identifier) @dec_call))
"""

# Python-only: find all from-import statements for symbol map building.
_PYTHON_FROM_IMPORT_QUERY_STR = "(import_from_statement) @stmt"

# --- Query caches (compiled once per language, reused across calls) ---

_IMPORT_QUERY_CACHE: dict[str, Query] = {}
_CALL_SITE_QUERY_CACHE: dict[str, Query] = {}
_FROM_IMPORT_QUERY_CACHE: dict[str, Query] = {}


def _get_import_query(language: str) -> Query:
    if language not in _IMPORT_QUERY_CACHE:
        lang_obj = get_language(language)  # type: ignore[arg-type]
        _IMPORT_QUERY_CACHE[language] = Query(lang_obj, IMPORT_QUERIES[language])
    return _IMPORT_QUERY_CACHE[language]


def _get_call_site_query(language: str) -> Query:
    if language not in _CALL_SITE_QUERY_CACHE:
        lang_obj = get_language(language)  # type: ignore[arg-type]
        _CALL_SITE_QUERY_CACHE[language] = Query(lang_obj, _PYTHON_CALL_SITE_QUERY)
    return _CALL_SITE_QUERY_CACHE[language]


def _get_from_import_query(language: str) -> Query:
    if language not in _FROM_IMPORT_QUERY_CACHE:
        lang_obj = get_language(language)  # type: ignore[arg-type]
        _FROM_IMPORT_QUERY_CACHE[language] = Query(lang_obj, _PYTHON_FROM_IMPORT_QUERY_STR)
    return _FROM_IMPORT_QUERY_CACHE[language]


def _build_imported_symbol_map(root: Any) -> dict[str, str]:
    """Query AST and return {imported_symbol_lower: source_module} from 'from X import Y' stmts.

    Uses a QueryCursor to locate import_from_statement nodes, then a TreeCursor to
    visit each 'name' field child, correctly handling multiple imported names and aliases.
    """
    mapping: dict[str, str] = {}
    query = _get_from_import_query("python")
    cursor = QueryCursor(query)
    captures = cursor.captures(root)
    for node in captures.get("stmt", []):
        mod_node = node.child_by_field_name("module_name")
        if mod_node and mod_node.text:
            module = mod_node.text.decode()
            tc = node.walk()
            if tc.goto_first_child():
                while True:
                    if tc.field_name == "name":
                        name_node = tc.node
                        if name_node is not None:
                            if name_node.type == "aliased_import":
                                alias = name_node.child_by_field_name("alias")
                                if alias and alias.text:
                                    mapping[alias.text.decode().lower()] = module
                            elif name_node.text:
                                # dotted_name: use the last component ('A.B' → 'B')
                                name = name_node.text.decode().split(".")[-1]
                                mapping[name.lower()] = module
                    if not tc.goto_next_sibling():
                        break
    return mapping


def _extract_call_sites(
    tree: Any,
    source_lines: list[str],
    file_path: str | Path,
    symbol_map: dict[str, str],
) -> list[CodeUsage]:
    """Query the tree for call sites, base classes, and decorators of imported symbols.

    Each returned CodeUsage has symbol='{source_module}.{identifier}' so
    _filter_usages_for_dep can match on the module prefix.
    """
    if not symbol_map:
        return []

    query = _get_call_site_query("python")
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    _capture_to_type = {
        "func": "call",
        "base": "base_class",
        "dec": "decorator",
        "dec_call": "decorator",
    }

    usages: list[CodeUsage] = []
    for capture_name, usage_type in _capture_to_type.items():
        for node in captures.get(capture_name, []):
            if not node.text:
                continue
            identifier = node.text.decode()
            module = symbol_map.get(identifier.lower())
            if module is None:
                continue
            line_number = node.start_point[0] + 1
            snippet = source_lines[line_number - 1] if line_number <= len(source_lines) else ""
            usages.append(
                CodeUsage(
                    file_path=str(file_path),
                    line_number=line_number,
                    usage_type=usage_type,
                    symbol=f"{module}.{identifier}",
                    code_snippet=snippet.strip(),
                )
            )
    return usages


def _get_language(file_path: str | Path) -> str | None:
    """Look up language from file extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(ext)


def _strip_quotes(text: str) -> str:
    """Strip surrounding quotes from a string literal."""
    if len(text) >= 2 and text[0] in ("'", '"') and text[-1] in ("'", '"'):
        return text[1:-1]
    return text


def _usage_type_from_parent(node_type: str, parent_type: str | None) -> str:
    """Determine usage_type string from node context."""
    if parent_type == "import_from_statement":
        return "import_from"
    return "import"


async def parse_file(file_path: str | Path, language: str) -> list[CodeUsage]:
    """Parse a file with tree-sitter and extract import usages."""
    file_path = Path(file_path)

    if language not in IMPORT_QUERIES:
        logger.debug("Unsupported language %r for file %s; skipping", language, file_path)
        return []

    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()

    parser = get_parser(language)  # type: ignore[arg-type]
    tree = parser.parse(source.encode())

    query = _get_import_query(language)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    usages: list[CodeUsage] = []

    module_nodes = captures.get("module", [])

    for node in module_nodes:
        raw_text = node.text.decode("utf-8") if node.text else ""
        symbol = _strip_quotes(raw_text)
        line_number = node.start_point[0] + 1  # 1-indexed
        parent_type = node.parent.type if node.parent else None

        # For JS/TS require() calls, the module node is a string inside
        # arguments of a call_expression whose function is "require".
        # Determine usage type based on the parent chain.
        if parent_type == "arguments":
            # This is a require() call argument
            usage_type = "import"
        else:
            usage_type = _usage_type_from_parent(node.type, parent_type)

        snippet = source_lines[line_number - 1] if line_number <= len(source_lines) else ""

        usages.append(
            CodeUsage(
                file_path=str(file_path),
                line_number=line_number,
                usage_type=usage_type,
                symbol=symbol,
                code_snippet=snippet.strip(),
            )
        )

    # For Python: also detect call sites, base classes, and decorators of imported symbols.
    if language == "python":
        symbol_map = _build_imported_symbol_map(tree.root_node)
        usages.extend(_extract_call_sites(tree, source_lines, file_path, symbol_map))

    return usages


def _filter_usages_for_dep(usages: list[CodeUsage], dep_name: str) -> list[CodeUsage]:
    """Filter usages where symbol matches dep_name (case-insensitive, handles dotted names).

    PyPI package names use hyphens (Flask-Login) while Python import names use
    underscores (flask_login). Both forms are tried.
    """
    dep_lower = dep_name.lower().replace("-", "_")
    filtered: list[CodeUsage] = []
    for u in usages:
        sym_lower = u.symbol.lower().replace("-", "_")
        # Match if symbol equals dep_name, or dep_name is a prefix segment
        # e.g. dep_name="flask" matches symbol="flask" or "flask.Flask"
        if sym_lower == dep_lower or sym_lower.startswith(dep_lower + "."):
            filtered.append(u)
    return filtered


async def find_usages(project_path: str | Path, dep_name: str) -> list[CodeUsage]:
    """Walk project files, parse each, and filter for dep_name usages."""
    project_path = Path(project_path)
    all_usages: list[CodeUsage] = []

    for ext, language in EXTENSION_MAP.items():
        for file_path in project_path.rglob(f"*{ext}"):
            # Skip hidden dirs, node_modules, __pycache__, .venv
            parts = file_path.parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__", ".venv") for p in parts):
                continue
            try:
                file_usages = await parse_file(file_path, language)
                all_usages.extend(file_usages)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to parse %s: %s", file_path, exc, exc_info=True)
                continue

    return _filter_usages_for_dep(all_usages, dep_name)
