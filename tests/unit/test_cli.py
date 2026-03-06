"""Tests for the CLI interface."""

import json
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from migratowl.interfaces.cli import app
from migratowl.models.schemas import AnalysisReport

runner = CliRunner()


def _make_report_json(project_path: str = "/tmp/project") -> str:
    report = AnalysisReport(
        project_path=project_path,
        timestamp="2026-01-01T00:00:00+00:00",
        total_dependencies=5,
        outdated_count=1,
        critical_count=0,
        assessments=[],
        patches=[],
        errors=[],
    )
    return report.model_dump_json()


def _fake_settings(**overrides):
    """Create a fake settings object with sensible defaults for CLI tests."""
    defaults = {"use_local_llm": False, "openai_api_key": "sk-test", "openai_model": "gpt-4o-mini"}
    defaults.update(overrides)
    return type("S", (), defaults)()


class TestAnalyzeCommand:
    def test_analyze_command_calls_analyzer(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ) as mock_analyze:
            result = runner.invoke(app, ["analyze", str(project_dir)])

            assert result.exit_code == 0
            mock_analyze.assert_called_once_with(str(project_dir), fix_mode=False)

    def test_analyze_nonexistent_path_fails(self) -> None:
        result = runner.invoke(app, ["analyze", "/nonexistent/path/that/does/not/exist"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_analyze_with_output_flag(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.json"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(app, ["analyze", str(project_dir), "--output", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert data["project_path"] == str(project_dir)


class TestFormatFlag:
    def test_format_markdown_writes_markdown_output(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.md"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(output_file), "--format", "markdown"]
            )

            assert result.exit_code == 0
            assert output_file.exists()
            content = output_file.read_text()
            assert "# MigratOwl Analysis Report" in content

    def test_format_json_is_default(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.json"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(app, ["analyze", str(project_dir), "--output", str(output_file)])

            assert result.exit_code == 0
            data = json.loads(output_file.read_text())
            assert "project_path" in data

    def test_format_inferred_from_json_extension(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.json"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(app, ["analyze", str(project_dir), "--output", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert "project_path" in data

    def test_format_inferred_from_md_extension(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.md"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            # No --format flag, but .md extension should infer markdown
            result = runner.invoke(app, ["analyze", str(project_dir), "--output", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            content = output_file.read_text()
            assert "# MigratOwl Analysis Report" in content

    def test_extension_appended_for_json(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_base = tmp_path / "report"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(output_base), "--format", "json"]
            )

            assert result.exit_code == 0
            expected_file = tmp_path / "report.json"
            assert expected_file.exists()
            data = json.loads(expected_file.read_text())
            assert "project_path" in data

    def test_extension_appended_for_markdown(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_base = tmp_path / "report"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(output_base), "--format", "markdown"]
            )

            assert result.exit_code == 0
            expected_file = tmp_path / "report.md"
            assert expected_file.exists()
            content = expected_file.read_text()
            assert "# MigratOwl Analysis Report" in content

    def test_matching_extension_and_format_ok(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.json"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(output_file), "--format", "json"]
            )

            assert result.exit_code == 0
            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert "project_path" in data

    def test_conflict_extension_vs_format_errors(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_file = tmp_path / "report.json"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(output_file), "--format", "markdown"]
            )

            assert result.exit_code == 1
            assert "conflict" in result.output.lower()

    def test_invalid_format_value_errors(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(tmp_path / "report"), "--format", "jsonfd"]
            )

            assert result.exit_code == 1
            assert "invalid" in result.output.lower() or "must be" in result.output.lower()

    def test_unknown_extension_errors(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(
                app, ["analyze", str(project_dir), "--output", str(tmp_path / "report.gdfs")]
            )

            assert result.exit_code == 1
            assert "extension" in result.output.lower()

    def test_no_format_no_extension_defaults_json(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        output_base = tmp_path / "report"

        monkeypatch.setattr("migratowl.interfaces.cli.settings", _fake_settings())

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            # No --format, no extension → defaults to json, appends .json
            result = runner.invoke(app, ["analyze", str(project_dir), "--output", str(output_base)])

            assert result.exit_code == 0
            expected_file = tmp_path / "report.json"
            assert expected_file.exists()
            data = json.loads(expected_file.read_text())
            assert "project_path" in data


class TestApiKeyValidation:
    def test_missing_api_key_shows_error(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        monkeypatch.setattr(
            "migratowl.interfaces.cli.settings", type("S", (), {"use_local_llm": False, "openai_api_key": ""})()
        )

        result = runner.invoke(app, ["analyze", str(project_dir)])

        assert result.exit_code == 1
        assert "MIGRATOWL_OPENAI_API_KEY" in result.output

    def test_local_llm_skips_api_key_check(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        monkeypatch.setattr(
            "migratowl.interfaces.cli.settings",
            type("S", (), {"use_local_llm": True, "openai_api_key": "", "openai_model": "llama3.2"})(),
        )

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(app, ["analyze", str(project_dir)])

        assert result.exit_code == 0


class TestModelFlag:
    def test_model_flag_overrides_settings(self, tmp_path, monkeypatch) -> None:
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        fake = _fake_settings()
        monkeypatch.setattr("migratowl.interfaces.cli.settings", fake)

        with patch(
            "migratowl.interfaces.cli.run_analysis",
            new_callable=AsyncMock,
            return_value=_make_report_json(str(project_dir)),
        ):
            result = runner.invoke(app, ["analyze", str(project_dir), "--model", "gpt-4o"])

        assert result.exit_code == 0
        assert fake.openai_model == "gpt-4o"


class TestInitCommand:
    def test_init_creates_env_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        env_file = tmp_path / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "MIGRATOWL_OPENAI_API_KEY" in content

    def test_init_warns_if_env_exists(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("existing")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "already exists" in result.output


class TestServeCommand:
    def test_serve_command_exists(self) -> None:
        # Verify the serve command is registered via Typer's Click-level commands
        import typer.main

        click_app = typer.main.get_command(app)
        assert "serve" in click_app.commands
