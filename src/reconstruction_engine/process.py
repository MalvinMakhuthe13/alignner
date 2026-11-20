from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


ProgressCallback = Callable[[str], None]


class ProcessCancelledError(RuntimeError):
    pass


class ProcessTimeoutError(RuntimeError):
    pass


class ProcessExecutionError(RuntimeError):
    pass


@dataclass
class ProcessResult:
    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    log_path: Optional[Path] = None

    @property
    def success(self) -> bool:
        return self.return_code == 0


class ProcessRunner:
    def __init__(
        self,
        log_dir: str | Path,
        default_timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout
        self.progress_callback = progress_callback
        self._cancel_event = threading.Event()
        self._active_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._lock:
            process = self._active_process

        if process is not None and process.poll() is None:
            process.terminate()

    def reset_cancellation(self) -> None:
        with self._lock:
            if self._active_process is not None:
                raise RuntimeError(
                    "Cannot reset cancellation while a process is active"
                )
            self._cancel_event.clear()

    def _report(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def run(
        self,
        command: Sequence[str],
        stage: str,
        cwd: str | Path,
        timeout: Optional[float] = None,
    ) -> ProcessResult:
        if self._cancel_event.is_set():
            raise ProcessCancelledError("Process cancelled before launch")

        command = [str(part) for part in command]
        working_dir = Path(cwd)

        if not working_dir.is_dir():
            raise FileNotFoundError(
                f"Working directory does not exist: {working_dir}"
            )

        if not command:
            raise ValueError("Command cannot be empty")

        stage_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in stage
        )
        log_path = self.log_dir / f"{stage_name}.log"
        effective_timeout = (
            self.default_timeout if timeout is None else timeout
        )

        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("Timeout must be greater than zero")

        started = time.monotonic()
        output_queue: queue.Queue = queue.Queue()
        output_parts: list[str] = []
        return_code = -1

        self._report(f"Starting: {stage}")

        try:
            with log_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as log_file:
                log_file.write("COMMAND:\n")
                log_file.write(subprocess.list2cmdline(command))
                log_file.write("\n\n")
                log_file.flush()

                process = subprocess.Popen(
                    command,
                    cwd=str(working_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                with self._lock:
                    self._active_process = process

                def read_output() -> None:
                    try:
                        assert process.stdout is not None
                        for line in process.stdout:
                            output_queue.put(line)
                    finally:
                        output_queue.put(None)

                reader = threading.Thread(
                    target=read_output,
                    name=f"{stage_name}-output-reader",
                    daemon=True,
                )
                reader.start()

                try:
                    deadline = (
                        started + effective_timeout
                        if effective_timeout is not None
                        else None
                    )
                    output_finished = False

                    while not output_finished:
                        if self._cancel_event.is_set():
                            self._terminate(process)
                            raise ProcessCancelledError(
                                f"Cancelled during stage: {stage}"
                            )

                        if (
                            deadline is not None
                            and time.monotonic() >= deadline
                        ):
                            self._terminate(process)
                            raise ProcessTimeoutError(
                                f"Stage timed out: {stage}"
                            )

                        try:
                            line = output_queue.get(timeout=0.1)
                        except queue.Empty:
                            if (
                                process.poll() is not None
                                and not reader.is_alive()
                            ):
                                break
                            continue

                        if line is None:
                            output_finished = True
                            continue

                        output_parts.append(line)
                        log_file.write(line)
                        log_file.flush()
                        self._report(line.rstrip())

                    return_code = process.wait()
                    reader.join(timeout=2)

                    if self._cancel_event.is_set():
                        raise ProcessCancelledError(
                            f"Cancelled during stage: {stage}"
                        )

                finally:
                    if process.poll() is None:
                        self._terminate(process)

                    with self._lock:
                        self._active_process = None

                    if process.stdout is not None:
                        process.stdout.close()

        except (ProcessCancelledError, ProcessTimeoutError):
            raise

        except OSError as exc:
            raise ProcessExecutionError(
                f"Could not execute stage '{stage}': {exc}"
            ) from exc

        duration = time.monotonic() - started

        result = ProcessResult(
            command=command,
            return_code=return_code,
            stdout="".join(output_parts),
            stderr="",
            duration_seconds=duration,
            log_path=log_path,
        )

        if not result.success:
            raise ProcessExecutionError(
                f"Stage '{stage}' failed with exit code "
                f"{return_code}. Log: {log_path}"
            )

        self._report(f"Completed: {stage} ({duration:.1f}s)")
        return result

