"""
Alignner as a desktop app.

Double-click the Alignner icon: this starts Alignner quietly in the
background and opens it in its own window (no terminal, no "localhost").
Closing the window stops it.

On each start it also:
  * pulls the latest version from GitHub (if this folder is a git clone and
    you're online; skipped quietly otherwise),
  * installs new requirements if requirements.txt changed.

Runs with the Python in the project's .venv (made by install_windows.bat).
Logs: %USERPROFILE%\\.alignner\\alignner.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the alignner folder
HOME = Path(os.environ.get("ALIGNNER_HOME", Path.home() / ".alignner"))
HOME.mkdir(parents=True, exist_ok=True)
LOG = HOME / "alignner.log"
STATE = HOME / "running.json"
IS_WIN = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WIN else 0                 # CREATE_NO_WINDOW


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    if sys.stdout and not str(sys.executable).lower().endswith("pythonw.exe"):
        print(line, flush=True)


# ─── small status window (optional: tkinter ships with Python on Windows) ────
class Splash:
    def __init__(self) -> None:
        self.root = None
        try:
            import tkinter as tk
            self.tk = tk
            r = tk.Tk()
            r.title("Alignner")
            r.overrideredirect(True)
            r.configure(bg="#0b0d10")
            w, h = 380, 120
            r.geometry(f"{w}x{h}+{(r.winfo_screenwidth() - w) // 2}+{(r.winfo_screenheight() - h) // 2}")
            tk.Label(r, text="Alignner", fg="#c8ff00", bg="#0b0d10", font=("Segoe UI", 18, "bold")).pack(pady=(18, 4))
            self.msg = tk.Label(r, text="Starting…", fg="#e6e8eb", bg="#0b0d10", font=("Segoe UI", 10))
            self.msg.pack()
            r.attributes("-topmost", True)
            r.update()
            self.root = r
        except Exception:
            self.root = None

    def say(self, text: str) -> None:
        log(text)
        if self.root:
            try:
                self.msg.configure(text=text)
                self.root.update()
            except Exception:
                pass

    def close(self) -> None:
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def error(self, text: str) -> None:
        log("ERROR: " + text)
        self.close()
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("Alignner", text + f"\n\nDetails: {LOG}")
            r.destroy()
        except Exception:
            pass


# ─── helpers ─────────────────────────────────────────────────────────────────
def console_python() -> str:
    """The venv's python.exe (not pythonw) to run tools with."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def run(cmd: list[str], timeout: float, cwd: Path = ROOT) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def free_port(preferred: int = 8501) -> int:
    for port in [preferred] + list(range(8502, 8600)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port for Alignner.")


def healthy(url: str, path: str = "/_stcore/health", timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def update_from_github(splash: Splash) -> None:
    """Fast-forward to the latest version; never touches local changes."""
    if os.environ.get("ALIGNNER_NO_UPDATE") or not (ROOT / ".git").exists() or not shutil.which("git"):
        return
    splash.say("Checking for updates…")
    code, out = run(["git", "status", "--porcelain", "--untracked-files=no"], 10)
    if code != 0 or out.strip():
        log("update skipped: local changes or git error")
        return
    code, out = run(["git", "pull", "--ff-only", "--quiet"], 25)
    log("git pull: " + ("ok" if code == 0 else "skipped (" + out.strip()[:200] + ")"))


def ensure_requirements(splash: Splash) -> None:
    req = ROOT / "requirements.txt"
    if not req.exists():
        return
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    stamp = HOME / "requirements.sha256"
    if stamp.exists() and stamp.read_text().strip() == digest:
        return
    splash.say("Installing what Alignner needs (first start / after an update)…")
    code, out = run([console_python(), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", str(req)], 1800)
    if code != 0:
        raise RuntimeError("Could not install requirements:\n" + out.strip()[-800:])
    stamp.write_text(digest)


def browser_app_window(url: str):
    """Open the URL as an app window (Edge / Chrome --app), with its own profile
    so we can tell when the window is closed. Returns the process or None."""
    candidates = []
    if IS_WIN:
        for base in [os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""), os.environ.get("LOCALAPPDATA", "")]:
            if not base:
                continue
            candidates += [Path(base) / "Microsoft/Edge/Application/msedge.exe", Path(base) / "Google/Chrome/Application/chrome.exe"]
    else:
        for name in ["microsoft-edge", "google-chrome", "chromium", "chromium-browser"]:
            p = shutil.which(name)
            if p:
                candidates.append(Path(p))
        mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        candidates.append(mac)
    for exe in candidates:
        if exe.exists():
            profile = HOME / "window-profile"
            args = [str(exe), f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
                    "--window-size=1440,920", "--disable-features=Translate"]
            log(f"opening window with {exe.name}")
            return subprocess.Popen(args, creationflags=NO_WINDOW)
    return None


def stop(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(8)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        STATE.unlink()
    except OSError:
        pass


# ─── main ────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Open Alignner as a desktop app.")
    ap.add_argument("--no-window", action="store_true", help="start the server only (for testing)")
    ap.add_argument("--server-cmd", help=argparse.SUPPRESS)        # tests: run something else
    ap.add_argument("--health-path", default="/_stcore/health", help=argparse.SUPPRESS)
    ap.add_argument("--exit-after-start", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    # Already open? Just bring up another window on it.
    try:
        st = json.loads(STATE.read_text())
        if healthy(st["url"], a.health_path):
            log("already running at " + st["url"])
            if not a.no_window:
                if not browser_app_window(st["url"]):
                    webbrowser.open(st["url"])
            return 0
    except Exception:
        pass

    splash = Splash()
    server = None
    try:
        update_from_github(splash)
        if not a.server_cmd:
            ensure_requirements(splash)
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        splash.say("Starting Alignner…")
        cmd = a.server_cmd.format(port=port).split() if a.server_cmd else [
            console_python(), "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", str(port),
            "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none",
        ]
        out = (HOME / "server.log").open("a", encoding="utf-8")
        server = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        STATE.write_text(json.dumps({"url": url, "pid": server.pid}))
        t0 = time.time()
        while not healthy(url, a.health_path):
            if server.poll() is not None:
                raise RuntimeError("Alignner stopped while starting. See " + str(HOME / "server.log"))
            if time.time() - t0 > 90:
                raise RuntimeError("Alignner took too long to start.")
            time.sleep(0.4)
        log(f"ready at {url} in {time.time() - t0:.1f}s")
        if a.exit_after_start:
            splash.close(); stop(server); return 0
        if a.no_window:
            splash.close()
            log("server only — press Ctrl+C to stop")
            server.wait()
            return 0

        # 1) a native window, if pywebview is installed
        try:
            import webview  # type: ignore
            splash.close()
            webview.create_window("Alignner", url, width=1440, height=920, background_color="#0b0d10")
            webview.start()
            return 0
        except ImportError:
            pass
        # 2) an Edge / Chrome app window (no tabs, no address bar)
        win = browser_app_window(url)
        splash.close()
        if win:
            win.wait()
            # the window may hand over to a running browser; keep serving while it is open
            while healthy(url, a.health_path) and _profile_in_use():
                time.sleep(2)
            return 0
        # 3) the default browser
        webbrowser.open(url)
        log("opened in the default browser; Alignner keeps running until you sign out / restart")
        server.wait()
        return 0
    except Exception as e:  # noqa: BLE001
        splash.error(str(e))
        return 1
    finally:
        stop(server)


def _profile_in_use() -> bool:
    """Chromium keeps a lock in its profile while a window is open."""
    lock = HOME / "window-profile" / ("lockfile" if IS_WIN else "SingletonLock")
    return lock.exists() or lock.is_symlink()


if __name__ == "__main__":
    sys.exit(main())
