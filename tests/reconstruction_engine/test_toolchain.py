import subprocess
from pathlib import Path

from src.reconstruction_engine.toolchain import (
    Toolchain,
    ToolStatus,
    ToolchainReport,
)


def test_missing_executable_is_unavailable(tmp_path):
    toolchain = Toolchain(
        colmap_path=str(tmp_path / "missing.exe"),
        openmvs_dir=str(tmp_path),
    )

    status = toolchain._check_tool(
        "missing",
        tmp_path / "missing.exe",
    )

    assert status.available is False
    assert status.launchable is False
    assert status.error == "Executable not found"


def test_successful_process_launch(monkeypatch, tmp_path):
    executable = tmp_path / "fake.exe"
    executable.touch()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    toolchain = Toolchain()
    status = toolchain._check_tool("fake", executable)

    assert status.available is True
    assert status.launchable is True
    assert status.exit_code == 0


def test_nonzero_exit_still_means_launchable(monkeypatch, tmp_path):
    executable = tmp_path / "fake.exe"
    executable.touch()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = Toolchain()._check_tool("fake", executable)

    assert status.available is True
    assert status.launchable is True
    assert status.exit_code == 1


def test_timeout_is_reported(monkeypatch, tmp_path):
    executable = tmp_path / "fake.exe"
    executable.touch()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = Toolchain(timeout=1)._check_tool(
        "fake",
        executable,
    )

    assert status.available is True
    assert status.launchable is False
    assert status.error == "Executable check timed out"


def test_empty_report_is_not_ready():
    assert ToolchainReport().ready is False


def test_report_readiness_requires_every_tool():
    report = ToolchainReport(
        tools={
            "colmap": ToolStatus(
                name="COLMAP",
                path=Path("colmap.exe"),
                available=True,
                launchable=True,
            ),
            "densify": ToolStatus(
                name="DensifyPointCloud",
                path=Path("DensifyPointCloud.exe"),
                available=True,
                launchable=False,
            ),
        }
    )

    assert report.ready is False
