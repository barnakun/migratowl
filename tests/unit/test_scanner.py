"""Tests for migratowl.core.scanner — dependency manifest detection and parsing."""

from pathlib import Path

from migratowl.core.scanner import (
    MANIFEST_PATTERNS,
    _parse_package_json,
    _parse_pipfile,
    _parse_pyproject_toml,
    _parse_requirements_txt,
    scan_project,
)
from migratowl.models.schemas import Ecosystem

# --- MANIFEST_PATTERNS ---


class TestManifestPatterns:
    def test_requirements_txt_maps_to_python(self) -> None:
        assert MANIFEST_PATTERNS["requirements.txt"] == Ecosystem.PYTHON

    def test_pyproject_toml_maps_to_python(self) -> None:
        assert MANIFEST_PATTERNS["pyproject.toml"] == Ecosystem.PYTHON

    def test_pipfile_maps_to_python(self) -> None:
        assert MANIFEST_PATTERNS["Pipfile"] == Ecosystem.PYTHON

    def test_package_json_maps_to_nodejs(self) -> None:
        assert MANIFEST_PATTERNS["package.json"] == Ecosystem.NODEJS

    def test_all_expected_patterns_present(self) -> None:
        expected = {"requirements.txt", "pyproject.toml", "Pipfile", "package.json"}
        assert expected.issubset(set(MANIFEST_PATTERNS.keys()))


# --- _parse_requirements_txt ---


class TestParseRequirementsTxt:
    async def test_pinned_versions(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("requests==2.28.0\nflask==2.3.0\n")
        deps = await _parse_requirements_txt(req)
        assert len(deps) == 2
        assert deps[0].name == "requests"
        assert deps[0].current_version == "2.28.0"
        assert deps[0].ecosystem == Ecosystem.PYTHON
        assert deps[1].name == "flask"
        assert deps[1].current_version == "2.3.0"

    async def test_gte_versions(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("httpx>=0.24\npydantic>=2.0\n")
        deps = await _parse_requirements_txt(req)
        assert len(deps) == 2
        assert deps[0].name == "httpx"
        assert deps[0].current_version == "0.24"
        assert deps[1].name == "pydantic"
        assert deps[1].current_version == "2.0"

    async def test_compatible_release(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("django~=4.2.0\n")
        deps = await _parse_requirements_txt(req)
        assert len(deps) == 1
        assert deps[0].name == "django"
        assert deps[0].current_version == "4.2.0"

    async def test_comments_and_blank_lines_skipped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("# a comment\nrequests==2.28.0\n\n# another\nflask==2.3.0\n")
        deps = await _parse_requirements_txt(req)
        assert len(deps) == 2

    async def test_dash_r_includes_skipped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("-r base.txt\nrequests==2.28.0\n")
        deps = await _parse_requirements_txt(req)
        assert len(deps) == 1
        assert deps[0].name == "requests"

    async def test_manifest_path_set(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("requests==2.28.0\n")
        deps = await _parse_requirements_txt(req)
        assert deps[0].manifest_path == str(req)

    async def test_no_version_skipped(self, tmp_path: Path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("requests\nflask==2.3.0\n")
        deps = await _parse_requirements_txt(req)
        assert len(deps) == 1
        assert deps[0].name == "flask"


# --- _parse_pyproject_toml ---


class TestParsePyprojectToml:
    async def test_project_dependencies(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nname = "test"\ndependencies = ["httpx>=0.24", "pydantic>=2.0"]\n')
        deps = await _parse_pyproject_toml(toml)
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert names == {"httpx", "pydantic"}

    async def test_optional_dependencies(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text(
            '[project]\nname = "test"\ndependencies = ["httpx>=0.24"]\n'
            "\n[project.optional-dependencies]\ndev = "
            '["pytest>=8.0", "ruff>=0.5"]\n'
        )
        deps = await _parse_pyproject_toml(toml)
        names = {d.name for d in deps}
        assert "httpx" in names
        assert "pytest" in names
        assert "ruff" in names

    async def test_no_project_section(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[build-system]\nrequires = ["hatchling"]\n')
        deps = await _parse_pyproject_toml(toml)
        assert deps == []

    async def test_version_with_upper_bound(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nname = "test"\ndependencies = ["langgraph>=1.0.7,<2"]\n')
        deps = await _parse_pyproject_toml(toml)
        assert len(deps) == 1
        assert deps[0].name == "langgraph"
        assert deps[0].current_version == "1.0.7"

    async def test_manifest_path_set(self, tmp_path: Path) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\nname = "test"\ndependencies = ["httpx>=0.24"]\n')
        deps = await _parse_pyproject_toml(toml)
        assert deps[0].manifest_path == str(toml)


# --- _parse_package_json ---


class TestParsePackageJson:
    async def test_dependencies_and_dev_dependencies(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text('{"dependencies": {"express": "^4.18.0"}, "devDependencies": {"jest": "^29.0.0"}}')
        deps = await _parse_package_json(pkg)
        assert len(deps) == 2
        names = {d.name for d in deps}
        assert names == {"express", "jest"}
        for d in deps:
            assert d.ecosystem == Ecosystem.NODEJS

    async def test_caret_version_extraction(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text('{"dependencies": {"express": "^4.18.0"}}')
        deps = await _parse_package_json(pkg)
        assert deps[0].current_version == "4.18.0"

    async def test_tilde_version_extraction(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text('{"dependencies": {"lodash": "~4.17.21"}}')
        deps = await _parse_package_json(pkg)
        assert deps[0].current_version == "4.17.21"

    async def test_exact_version(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text('{"dependencies": {"react": "18.2.0"}}')
        deps = await _parse_package_json(pkg)
        assert deps[0].current_version == "18.2.0"

    async def test_no_deps_section(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text('{"name": "test", "version": "1.0.0"}')
        deps = await _parse_package_json(pkg)
        assert deps == []

    async def test_manifest_path_set(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text('{"dependencies": {"express": "^4.18.0"}}')
        deps = await _parse_package_json(pkg)
        assert deps[0].manifest_path == str(pkg)


# --- _parse_pipfile ---


class TestParsePipfile:
    async def test_basic_pipfile(self, tmp_path: Path) -> None:
        pf = tmp_path / "Pipfile"
        pf.write_text('[packages]\nrequests = "==2.28.0"\nflask = ">=2.3.0"\n\n[dev-packages]\npytest = ">=8.0"\n')
        deps = await _parse_pipfile(pf)
        names = {d.name for d in deps}
        assert "requests" in names
        assert "flask" in names
        assert "pytest" in names
        for d in deps:
            assert d.ecosystem == Ecosystem.PYTHON

    async def test_star_version_skipped(self, tmp_path: Path) -> None:
        pf = tmp_path / "Pipfile"
        pf.write_text('[packages]\nrequests = "*"\nflask = "==2.3.0"\n')
        deps = await _parse_pipfile(pf)
        assert len(deps) == 1
        assert deps[0].name == "flask"


# --- scan_project ---


class TestScanProject:
    async def test_finds_requirements_txt(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("requests==2.28.0\n")
        deps = await scan_project(project)
        assert len(deps) == 1
        assert deps[0].name == "requests"

    async def test_finds_nested_manifests(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        sub = project / "subdir"
        sub.mkdir()
        (sub / "requirements.txt").write_text("flask==2.3.0\n")
        deps = await scan_project(project)
        assert len(deps) == 1
        assert deps[0].name == "flask"

    async def test_multiple_manifest_types(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("requests==2.28.0\nflask==2.3.0\n")
        (project / "pyproject.toml").write_text(
            '[project]\nname = "test"\ndependencies = ["httpx>=0.24", "pydantic>=2.0"]\n'
        )
        (project / "package.json").write_text(
            '{"dependencies": {"express": "^4.18.0"}, "devDependencies": {"jest": "^29.0.0"}}'
        )
        deps = await scan_project(project)
        names = {d.name for d in deps}
        assert "requests" in names
        assert "flask" in names
        assert "httpx" in names
        assert "pydantic" in names
        assert "express" in names
        assert "jest" in names

    async def test_skips_node_modules(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        nm = project / "node_modules" / "some-pkg"
        nm.mkdir(parents=True)
        (nm / "package.json").write_text('{"dependencies": {"inner": "1.0.0"}}')
        (project / "package.json").write_text('{"dependencies": {"express": "^4.18.0"}}')
        deps = await scan_project(project)
        assert len(deps) == 1
        assert deps[0].name == "express"

    async def test_skips_venv(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        venv = project / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "requirements.txt").write_text("something==1.0.0\n")
        (project / "requirements.txt").write_text("requests==2.28.0\n")
        deps = await scan_project(project)
        assert len(deps) == 1
        assert deps[0].name == "requests"

    async def test_empty_project(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        deps = await scan_project(project)
        assert deps == []

    async def test_accepts_string_path(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("requests==2.28.0\n")
        deps = await scan_project(str(project))
        assert len(deps) == 1

    async def test_deduplicates_same_package_across_manifests(self, tmp_path: Path) -> None:
        """requirements.txt and Pipfile listing the same package must yield one dep, not two."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==2.3.0\nrequests==2.28.0\n")
        (project / "Pipfile").write_text('[packages]\nflask = "==2.3.0"\nrequests = "==2.28.0"\n')
        deps = await scan_project(project)
        names = [d.name.lower() for d in deps]
        assert names.count("flask") == 1
        assert names.count("requests") == 1
