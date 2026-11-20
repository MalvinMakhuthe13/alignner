import sys
import threading
import time
from pathlib import Path

import pytest

from src.reconstruction_engine.process import (
    ProcessCancelledError,
    ProcessExecutionError,
    ProcessRunner,
    ProcessTimeoutError,
)


def test_successful_process_captures_output(tmp_path):
    runner = ProcessRunner(log_dir=tmp_path)

    result = runner.run(
        [sys.executable, "-c", "print('ALIGNNER SUCCESS')"],
        stage="success",
        cwd=tmp_path,
    )

    assert result.success
    assert "ALIGNNER SUCCESS" in result.stdout
    assert result.log_path.exists()


def test_failed_process_preserves_log(tmp_path):
    runner = ProcessRunner(log_dir=tmp_path)

    with pytest.raises(ProcessExecutionError, match="exit code 7"):
        runner.run(
            [
                sys.executable,
                "-c",
                "print('SIMULATED FAILURE', flush=True); raise SystemExit(7)",
            ],
            stage="failure",
            cwd=tmp_path,
        )

    log = tmp_path / "failure.log"
    assert log.exists()
    assert "SIMULATED FAILURE" in log.read_text(encoding="utf-8")


def test_silent_process_times_out(tmp_path):
    runner = ProcessRunner(log_dir=tmp_path)
    started = time.monotonic()

    with pytest.raises(ProcessTimeoutError):
        runner.run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stage="timeout",
            cwd=tmp_path,
            timeout=1,
        )

    assert time.monotonic() - started < 5


def test_running_process_can_be_cancelled(tmp_path):
    marker = tmp_path / "child_started.marker"
    runner = ProcessRunner(log_dir=tmp_path)
    outcome = {}

    def run_process():
        try:
            runner.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import time; "
                        f"Path({str(marker)!r}).write_text('started'); "
                        "time.sleep(20)"
                    ),
                ],
                stage="cancel",
                cwd=tmp_path,
                timeout=30,
            )
            outcome["result"] = "unexpected_success"
        except ProcessCancelledError:
            outcome["result"] = "cancelled"
        except Exception as exc:
            outcome["result"] = repr(exc)

    worker = threading.Thread(target=run_process)
    worker.start()

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if marker.exists():
            break
        time.sleep(0.05)

    if not marker.exists():
        runner.cancel()
        worker.join(timeout=5)
        pytest.fail("Child process did not create its startup marker")

    runner.cancel()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert outcome.get("result") == "cancelled"


def test_cancelled_runner_rejects_new_process_until_reset(tmp_path):
    runner = ProcessRunner(log_dir=tmp_path)
    runner.cancel()

    with pytest.raises(ProcessCancelledError):
        runner.run(
            [sys.executable, "-c", "print('should not run')"],
            stage="pre_cancelled",
            cwd=tmp_path,
        )

    runner.reset_cancellation()

    result = runner.run(
        [sys.executable, "-c", "print('RESET SUCCESS')"],
        stage="after_reset",
        cwd=tmp_path,
    )

    assert result.success
    assert "RESET SUCCESS" in result.stdout
