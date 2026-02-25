"""Tests for tree-sitter based code parser."""

from __future__ import annotations

import unittest.mock
from pathlib import Path

import pytest
from tree_sitter import Query
from tree_sitter_language_pack import get_parser as ts_get_parser

from migratowl.core.code_parser import (
    _CALL_SITE_QUERY_CACHE,
    _FROM_IMPORT_QUERY_CACHE,
    _IMPORT_QUERY_CACHE,
    _build_imported_symbol_map,
    _get_import_query,
    _get_language,
    find_usages,
    parse_file,
)
from migratowl.models.schemas import CodeUsage

# --- Fixtures ---

PYTHON_FIXTURE = """\
import requests
from flask import Flask, jsonify
from os import path

app = Flask(__name__)
response = requests.get("https://api.example.com")
data = response.json()
"""

JS_FIXTURE = """\
const express = require('express');
import axios from 'axios';
import { useState } from 'react';

const app = express();
const result = axios.get('/api');
"""


@pytest.fixture()
def python_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample_python.py"
    f.write_text(PYTHON_FIXTURE)
    return f


@pytest.fixture()
def js_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample_js.js"
    f.write_text(JS_FIXTURE)
    return f


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(PYTHON_FIXTURE)
    return project


# --- Language detection tests ---


def test_get_language_python() -> None:
    assert _get_language("example.py") == "python"


def test_get_language_javascript() -> None:
    assert _get_language("app.js") == "javascript"


def test_get_language_typescript() -> None:
    assert _get_language("component.ts") == "typescript"
    assert _get_language("component.tsx") == "typescript"


def test_get_language_unknown() -> None:
    assert _get_language("data.xyz") is None


# --- Parse file tests ---


@pytest.mark.asyncio()
async def test_parse_file_python_imports(python_file: Path) -> None:
    usages = await parse_file(python_file, "python")
    symbols = {u.symbol for u in usages}
    assert "requests" in symbols
    assert "flask" in symbols
    assert "os" in symbols


@pytest.mark.asyncio()
async def test_parse_file_javascript_imports(js_file: Path) -> None:
    usages = await parse_file(js_file, "javascript")
    symbols = {u.symbol for u in usages}
    assert "express" in symbols
    assert "axios" in symbols
    assert "react" in symbols


# --- find_usages tests ---


@pytest.mark.asyncio()
async def test_find_usages_filters_by_dep(sample_project: Path) -> None:
    usages = await find_usages(sample_project, "requests")
    assert len(usages) >= 1
    for u in usages:
        assert "requests" in u.symbol.lower()


@pytest.mark.asyncio()
async def test_find_usages_empty_for_unused_dep(sample_project: Path) -> None:
    usages = await find_usages(sample_project, "numpy")
    assert usages == []


# --- CodeUsage field validation ---


@pytest.mark.asyncio()
async def test_find_usages_normalizes_hyphenated_dep_name(tmp_path: Path) -> None:
    """PyPI uses hyphens (Flask-Login) but imports use underscores (flask_login).
    find_usages must match both."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "auth.py").write_text("from flask_login import LoginManager, login_required\n")

    usages = await find_usages(proj, "Flask-Login")
    assert len(usages) >= 1
    assert any("flask_login" in u.symbol for u in usages)


@pytest.mark.asyncio()
async def test_code_usage_has_correct_fields(python_file: Path) -> None:
    usages = await parse_file(python_file, "python")
    assert len(usages) > 0
    for u in usages:
        assert isinstance(u, CodeUsage)
        assert isinstance(u.file_path, str)
        assert isinstance(u.line_number, int)
        assert u.line_number > 0
        assert isinstance(u.symbol, str)
        assert len(u.symbol) > 0
        assert isinstance(u.usage_type, str)
        assert u.usage_type in ("import", "import_from", "call", "base_class", "decorator")


# --- Call site detection tests ---


class TestCallSiteDetection:
    @pytest.mark.asyncio
    async def test_detects_constructor_call_of_imported_symbol(self, tmp_path: Path) -> None:
        """'from flask_sqlalchemy import SQLAlchemy; db = SQLAlchemy()' must produce
        a call-type usage pointing at the instantiation line."""
        source = "from flask_sqlalchemy import SQLAlchemy\ndb = SQLAlchemy()\n"
        f = tmp_path / "app.py"
        f.write_text(source)

        usages = await parse_file(f, "python")

        call_usages = [u for u in usages if u.usage_type == "call"]
        assert any("flask_sqlalchemy" in u.symbol and u.line_number == 2 for u in call_usages), (
            f"Expected flask_sqlalchemy call at line 2, got: {call_usages}"
        )

    @pytest.mark.asyncio
    async def test_detects_base_class_of_imported_symbol(self, tmp_path: Path) -> None:
        """'from flask_wtf import FlaskForm; class Foo(FlaskForm):' must produce
        a base_class-type usage at the class definition line."""
        source = "from flask_wtf import FlaskForm\nclass UserForm(FlaskForm):\n    pass\n"
        f = tmp_path / "form.py"
        f.write_text(source)

        usages = await parse_file(f, "python")

        base_usages = [u for u in usages if u.usage_type == "base_class"]
        assert any("flask_wtf" in u.symbol and u.line_number == 2 for u in base_usages), (
            f"Expected flask_wtf base_class at line 2, got: {base_usages}"
        )

    @pytest.mark.asyncio
    async def test_detects_decorator_of_imported_symbol(self, tmp_path: Path) -> None:
        """'from flask_user import login_required; @login_required' must produce
        a decorator-type usage at the decorator line."""
        source = "from flask_user import login_required\n\n@login_required\ndef view():\n    pass\n"
        f = tmp_path / "views.py"
        f.write_text(source)

        usages = await parse_file(f, "python")

        dec_usages = [u for u in usages if u.usage_type == "decorator"]
        assert any("flask_user" in u.symbol and u.line_number == 3 for u in dec_usages), (
            f"Expected flask_user decorator at line 3, got: {dec_usages}"
        )

    @pytest.mark.asyncio
    async def test_find_usages_returns_call_site_for_dep(self, tmp_path: Path) -> None:
        """find_usages must include call-site usages so Flask-WTF and Flask-SQLAlchemy
        match their actual instantiation lines, not just the import line."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("from flask_wtf import FlaskForm\nclass ContactForm(FlaskForm):\n    pass\n")

        usages = await find_usages(proj, "Flask-WTF")

        assert len(usages) >= 2  # import + base_class
        assert any(u.usage_type == "base_class" for u in usages)
        assert any(u.line_number == 2 for u in usages)


# --- Fix 1: Query cache tests ---


class TestQueryCache:
    @pytest.mark.asyncio
    async def test_import_cache_populated_after_parse(self, tmp_path: Path) -> None:
        f = tmp_path / "t.py"
        f.write_text("import requests\n")
        await parse_file(f, "python")
        assert "python" in _IMPORT_QUERY_CACHE
        assert isinstance(_IMPORT_QUERY_CACHE["python"], Query)

    def test_same_import_query_object_returned(self) -> None:
        q1 = _get_import_query("python")
        q2 = _get_import_query("python")
        assert q1 is q2

    @pytest.mark.asyncio
    async def test_call_site_cache_populated(self, tmp_path: Path) -> None:
        f = tmp_path / "t.py"
        f.write_text("from flask import Flask\napp = Flask(__name__)\n")
        await parse_file(f, "python")
        assert "python" in _CALL_SITE_QUERY_CACHE
        assert isinstance(_CALL_SITE_QUERY_CACHE["python"], Query)


# --- Fix 2: JS/TS require() query tests ---


class TestJavaScriptRequireQuery:
    @pytest.mark.asyncio
    async def test_identifier_call_with_string_not_captured(self, tmp_path: Path) -> None:
        """foo('express') must not appear — arbitrary function call is a false positive."""
        f = tmp_path / "t.js"
        f.write_text("foo('express');\n")
        usages = await parse_file(f, "javascript")
        assert not any(u.symbol == "express" for u in usages)

    @pytest.mark.asyncio
    async def test_require_is_captured(self, tmp_path: Path) -> None:
        """require('express') must be captured as a module import."""
        f = tmp_path / "t.js"
        f.write_text("const express = require('express');\n")
        usages = await parse_file(f, "javascript")
        assert any(u.symbol == "express" for u in usages)

    @pytest.mark.asyncio
    async def test_typescript_identifier_call_not_captured(self, tmp_path: Path) -> None:
        """Same false-positive fix applies to TypeScript."""
        f = tmp_path / "t.ts"
        f.write_text("foo('express');\n")
        usages = await parse_file(f, "typescript")
        assert not any(u.symbol == "express" for u in usages)


# --- Fix 3: _build_imported_symbol_map query-based tests ---


class TestBuildImportedSymbolMap:
    def _parse_python(self, source: str):  # type: ignore[return]
        parser = ts_get_parser("python")
        return parser.parse(source.encode()).root_node

    @pytest.mark.asyncio
    async def test_from_import_query_cache_populated(self, tmp_path: Path) -> None:
        f = tmp_path / "t.py"
        f.write_text("from flask import Flask\n")
        await parse_file(f, "python")
        assert "python" in _FROM_IMPORT_QUERY_CACHE
        assert isinstance(_FROM_IMPORT_QUERY_CACHE["python"], Query)

    def test_flask_imports_both_mapped(self) -> None:
        root = self._parse_python("from flask import Flask, jsonify\n")
        result = _build_imported_symbol_map(root)
        assert result.get("flask") == "flask"
        assert result.get("jsonify") == "flask"

    def test_aliased_import_mapped(self) -> None:
        root = self._parse_python("from flask_sqlalchemy import SQLAlchemy as db\n")
        result = _build_imported_symbol_map(root)
        assert result == {"db": "flask_sqlalchemy"}

    def test_plain_import_not_mapped(self) -> None:
        root = self._parse_python("import requests\n")
        result = _build_imported_symbol_map(root)
        assert result == {}

    def test_os_path_join_mapped(self) -> None:
        root = self._parse_python("from os.path import join\n")
        result = _build_imported_symbol_map(root)
        assert result.get("join") == "os.path"


# --- Fix 4: Logging and validation tests ---


class TestLoggingAndValidation:
    @pytest.mark.asyncio
    async def test_unsupported_language_returns_empty(self, tmp_path: Path) -> None:
        """parse_file with unsupported language returns [] without raising KeyError."""
        f = tmp_path / "t.rb"
        f.write_text("require 'rails'\n")
        result = await parse_file(f, "ruby")
        assert result == []

    @pytest.mark.asyncio
    async def test_find_usages_logs_debug_on_failure(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """find_usages emits a DEBUG log when a file parse raises."""
        import logging

        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "good.py").write_text("import requests\n")

        with unittest.mock.patch(
            "migratowl.core.code_parser.parse_file",
            side_effect=RuntimeError("parse error"),
        ):
            with caplog.at_level(logging.DEBUG, logger="migratowl.core.code_parser"):
                await find_usages(proj, "requests")

        assert any("parse error" in r.message or "Failed to parse" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_find_usages_continues_after_failure(self, tmp_path: Path) -> None:
        """find_usages continues processing remaining files after one failure."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "a.py").write_text("import requests\n")
        (proj / "b.py").write_text("import requests\n")

        parse_calls: list[Path] = []
        original = parse_file

        async def mock_parse(fp: Path, lang: str) -> list[CodeUsage]:
            parse_calls.append(fp)
            if len(parse_calls) == 1:
                raise RuntimeError("simulated failure")
            return await original(fp, lang)

        with unittest.mock.patch("migratowl.core.code_parser.parse_file", new=mock_parse):
            usages = await find_usages(proj, "requests")

        assert len(parse_calls) == 2
        assert any("requests" in u.symbol for u in usages)
