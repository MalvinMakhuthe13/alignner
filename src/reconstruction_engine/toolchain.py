from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ToolStatus:
    name: str
    path: Optional[Path]
    available: bool
    launchable: bool = False
    exit_code: Optional[int] = None
    version_output: str = ""
    error: Optional[str] = None


@dataclass
class ToolchainReport:
    tools: dict[str, ToolStatus] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.tools) and all(
            tool.available and tool.launchable
            for tool in self.tools.values()
        )

    def summary(self) -> dict:
        return {
            name: {
                "path": str(tool.path) if tool.path else None,
                "available": tool.available,
                "launchable": tool.launchable,
                "exit_code": tool.exit_code,
                "error": tool.error,
            }
            for name, tool in self.tools.items()
        }


class Toolchain:
    def __init__(
        self,
        colmap_path: Optional[str] = None,
        openmvs_dir: Optional[str] = None,
        timeout: int = 15,
    ):
        project_root = Path(__file__).resolve().parents[2]

        self.colmap_path = colmap_path or os.getenv(
            "ALIGNNER_COLMAP_PATH",
            r"D:\colmap\bin\colmap.exe",
        )

        default_openmvs = (
            project_root.parent
            / "alignner_benchmark"
            / "tools"
            / "OpenMVS_v2.4.0"
            / "vc17"
            / "x64"
            / "Release"
        )

        self.openmvs_dir = Path(
            openmvs_dir or os.getenv(
                "ALIGNNER_OPENMVS_DIR",
                str(default_openmvs),
            )
        )

        self.timeout = timeout

        self.openmvs_tools = {
            "interface_colmap": "InterfaceCOLMAP.exe",
            "densify": "DensifyPointCloud.exe",
            "reconstruct_mesh": "ReconstructMesh.exe",
            "refine_mesh": "RefineMesh.exe",
            "texture_mesh": "TextureMesh.exe",
        }

    def _check_tool(self, name: str, path: Path) -> ToolStatus:
        if not path.is_file():
            return ToolStatus(
                name=name,
                path=path,
                available=False,
                error="Executable not found",
            )

        status = ToolStatus(
            name=name,
            path=path,
            available=True,
        )

        try:
            result = subprocess.run(
                [str(path), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout,
                check=False,
                cwd=str(path.parent),
            )

            status.launchable = True
            status.exit_code = result.returncode

        except subprocess.TimeoutExpired:
            status.error = "Executable check timed out"

        except OSError as exc:
            status.error = str(exc)

        return status

    def inspect(self) -> ToolchainReport:
        report = ToolchainReport()

        colmap = Path(self.colmap_path)
        report.tools["colmap"] = self._check_tool(
            "COLMAP",
            colmap,
        )

        for key, filename in self.openmvs_tools.items():
            path = self.openmvs_dir / filename
            report.tools[key] = self._check_tool(
                key,
                path,
            )

        return report
