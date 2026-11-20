from __future__ import annotations

from pathlib import Path

from src.reconstruction_engine.config import ReconstructionConfig


class OpenMVSCommands:
    def __init__(self, executable_dir: str):
        self.executable_dir = Path(executable_dir)

    def _exe(self, filename: str) -> str:
        path = self.executable_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"OpenMVS executable not found: {path}")
        return str(path)

    def interface_colmap(
        self,
        undistorted_path: str,
        output_path: str,
        image_path: str | None = None,
        num_threads: int = 4,
    ) -> list[str]:
        command = [
            self._exe("InterfaceCOLMAP.exe"),
            "--input-file", str(undistorted_path),
            "--output-file", str(output_path),
            "--max-threads", str(num_threads),
        ]

        if image_path is not None:
            command.extend(["--image-folder", str(image_path)])

        return command

    def densify(
        self,
        scene_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self._exe("DensifyPointCloud.exe"),
            str(scene_path),
            "--resolution-level", str(config.resolution_level),
            "--number-views", str(config.densify_num_views),
            "--estimate-colors", "2" if config.estimate_colors else "0",
            "--max-threads", str(config.num_threads),
        ]

    def reconstruct_mesh(
        self,
        scene_path: str,
        output_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self._exe("ReconstructMesh.exe"),
            "--input-file", str(scene_path),
            "--output-file", str(output_path),
            "--min-point-distance", str(config.mesh_min_point_distance),
            "--max-threads", str(config.num_threads),
        ]

    def refine_mesh(
        self,
        scene_path: str,
        mesh_path: str,
        output_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self._exe("RefineMesh.exe"),
            "--input-file", str(scene_path),
            "--mesh-file", str(mesh_path),
            "--output-file", str(output_path),
            "--resolution-level", str(config.mesh_resolution_level),
            "--max-threads", str(config.num_threads),
        ]

    def texture_mesh(
        self,
        scene_path: str,
        mesh_path: str,
        output_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self._exe("TextureMesh.exe"),
            "--input-file", str(scene_path),
            "--mesh-file", str(mesh_path),
            "--output-file", str(output_path),
            "--max-threads", str(config.num_threads),
        ]

