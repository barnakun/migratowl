"""Tests for logging configuration and verbose flag."""

import logging
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from migratowl.interfaces.cli import app, configure_logging

runner = CliRunner()


class TestConfigureLogging:
    def teardown_method(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_default_level_is_warning(self) -> None:
        with patch("migratowl.interfaces.cli.settings") as mock_settings:
            mock_settings.log_level = "WARNING"
            configure_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_one_sets_info(self) -> None:
        configure_logging(verbosity=1)
        assert logging.getLogger().level == logging.INFO

    def test_verbose_two_sets_debug(self) -> None:
        configure_logging(verbosity=2)
        assert logging.getLogger().level == logging.DEBUG

    def test_env_log_level_respected(self) -> None:
        with patch("migratowl.interfaces.cli.settings") as mock_settings:
            mock_settings.log_level = "DEBUG"
            configure_logging(verbosity=0)
        assert logging.getLogger().level == logging.DEBUG

    def test_verbosity_overrides_env_setting(self) -> None:
        with patch("migratowl.interfaces.cli.settings") as mock_settings:
            mock_settings.log_level = "DEBUG"
            configure_logging(verbosity=1)
        assert logging.getLogger().level == logging.INFO


class TestVerboseFlag:
    def test_v_flag_accepted(self) -> None:
        fake_analyze = AsyncMock(return_value='{"dependencies": []}')
        with (
            patch("migratowl.interfaces.cli.run_analysis", fake_analyze),
            patch("migratowl.interfaces.cli.configure_logging") as mock_config,
            patch("migratowl.interfaces.cli._render_or_write_report"),
        ):
            runner.invoke(app, ["analyze", "/tmp/fake-project", "-v"])
            mock_config.assert_called_once_with(1)

    def test_vv_flag_accepted(self) -> None:
        fake_analyze = AsyncMock(return_value='{"dependencies": []}')
        with (
            patch("migratowl.interfaces.cli.run_analysis", fake_analyze),
            patch("migratowl.interfaces.cli.configure_logging") as mock_config,
            patch("migratowl.interfaces.cli._render_or_write_report"),
        ):
            runner.invoke(app, ["analyze", "/tmp/fake-project", "-v", "-v"])
            mock_config.assert_called_once_with(2)


class TestAnalyzerLogging:
    @pytest.mark.asyncio
    async def test_scan_node_logs_progress(self, caplog: pytest.LogCaptureFixture) -> None:
        from migratowl.core.analyzer import scan_dependencies_node

        state = {"project_path": "/fake", "ignored_dependencies": []}
        with (
            caplog.at_level(logging.INFO, logger="migratowl.core.analyzer"),
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=[]),
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
        ):
            await scan_dependencies_node(state)
        assert any("Scanning dependencies" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_scan_node_logs_found_count(self, caplog: pytest.LogCaptureFixture) -> None:
        from migratowl.core.analyzer import scan_dependencies_node
        from migratowl.models.schemas import OutdatedDependency

        outdated = [
            OutdatedDependency(
                name="foo",
                current_version="1.0",
                latest_version="2.0",
                ecosystem="python",
                manifest_path="pyproject.toml",
            ),
            OutdatedDependency(
                name="bar",
                current_version="1.0",
                latest_version="2.0",
                ecosystem="python",
                manifest_path="pyproject.toml",
            ),
        ]
        state = {"project_path": "/fake", "ignored_dependencies": []}
        with (
            caplog.at_level(logging.INFO, logger="migratowl.core.analyzer"),
            patch(
                "migratowl.core.analyzer.scanner.scan_project",
                new_callable=AsyncMock,
                return_value=[1, 2, 3],
            ),
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(outdated, []),
            ),
        ):
            await scan_dependencies_node(state)
        assert any("2 outdated" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_cache_hit_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        from migratowl.core.analyzer import check_cache_node

        state = {
            "dep_name": "requests",
            "project_path": "/fake",
            "current_version": "1.0",
            "latest_version": "2.0",
        }
        cached_assessment = {"dep_name": "requests", "overall_severity": "info"}
        with (
            caplog.at_level(logging.INFO, logger="migratowl.core.analyzer"),
            patch("migratowl.core.analyzer.cache.get_cached_assessment", return_value=cached_assessment),
        ):
            await check_cache_node(state)
        assert any("cache hit" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_generate_report_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        from migratowl.core.analyzer import generate_report_node

        state = {
            "project_path": "/fake",
            "impact_assessments": [],
            "patches": [],
            "errors": [],
            "total_dependencies": 0,
        }
        with (
            caplog.at_level(logging.INFO, logger="migratowl.core.analyzer"),
            patch("migratowl.core.analyzer.report.build_report") as mock_build,
            patch("migratowl.core.analyzer.report.export_json", return_value="{}"),
        ):
            mock_build.return_value = "fake_report"
            await generate_report_node(state)
        assert any("Generating report" in r.message for r in caplog.records)
