from __future__ import annotations

import json
import shutil
import sqlite3
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.reconstruction_engine.colmap import COLMAPCommands
from src.reconstruction_engine.config import ReconstructionConfig
from src.reconstruction_engine.diagnostics import (
    read_images_binary,
    read_points3d_binary,
    build_sparse_diagnostics_report,
    analyze_camera_pair_graph,
)
from src.reconstruction_engine.openmvs import OpenMVSCommands
from src.reconstruction_engine.process import (
    ProcessCancelledError,
    ProcessRunner,
)
from src.reconstruction_engine.toolchain import Toolchain


ProgressCallback = Callable[[str], None]


class ReconstructionError(RuntimeError):
    pass


@dataclass
class ReconstructionPaths:
    images: Path
    workspace: Path

    @property
    def database(self) -> Path:
        return self.workspace / "database.db"

    @property
    def sparse(self) -> Path:
        return self.workspace / "sparse"

    @property
    def undistorted(self) -> Path:
        return self.workspace / "undistorted"

    @property
    def mvs_scene(self) -> Path:
        return self.workspace / "scene.mvs"

    @property
    def dense_scene(self) -> Path:
        return self.workspace / "scene_dense.mvs"

    @property
    def mesh(self) -> Path:
        return self.workspace / "mesh.ply"

    @property
    def refined_mesh(self) -> Path:
        return self.workspace / "mesh_refined.ply"

    @property
    def textured_mesh(self) -> Path:
        return self.workspace / "mesh_textured.ply"

    @property
    def logs(self) -> Path:
        return self.workspace / "logs"

    @property
    def manifest(self) -> Path:
        return self.workspace / "reconstruction_manifest.json"


@dataclass
class ReconstructionResult:
    success: bool
    workspace: Path
    final_model: Optional[Path] = None
    stages_completed: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    manifest_path: Optional[Path] = None
    diagnostics: Optional[dict] = None


class ReconstructionPipeline:
    def __init__(
        self,
        config: ReconstructionConfig,
        colmap_path: Optional[str] = None,
        openmvs_dir: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        stage_timeout: Optional[float] = None,
    ):
        self.config = config
        self.toolchain = Toolchain(
            colmap_path=colmap_path,
            openmvs_dir=openmvs_dir,
        )

        self.colmap_path = colmap_path or self.toolchain.colmap_path
        self.openmvs_dir = openmvs_dir or str(
            self.toolchain.openmvs_dir
        )

        self.progress_callback = progress_callback
        self.stage_timeout = stage_timeout
        self.paths: Optional[ReconstructionPaths] = None

        self.runner: Optional[ProcessRunner] = None
        self.stages_completed: list[str] = []
        self.diagnostics: Optional[dict] = None
        self.database_diagnostics: Optional[dict] = None

    def _report(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def cancel(self) -> None:
        if self.runner is not None:
            self.runner.cancel()

    def _validate_inputs(
        self,
        images: Path,
        workspace: Path,
    ) -> None:
        if not images.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {images}"
            )

        supported = {
            ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"
        }

        image_files = [
            item for item in images.iterdir()
            if item.is_file() and item.suffix.lower() in supported
        ]

        if len(image_files) < 2:
            raise ReconstructionError(
                "At least two supported images are required."
            )

        if images.resolve() == workspace.resolve():
            raise ReconstructionError(
                "Workspace cannot be the image input directory."
            )

        if workspace.exists() and any(workspace.iterdir()):
            raise ReconstructionError(
                f"Reconstruction workspace must be empty: {workspace}"
            )

        if workspace in images.parents:
            raise ReconstructionError(
                "Workspace cannot be a parent of the image directory."
            )

        if images in workspace.parents:
            raise ReconstructionError(
                "Workspace cannot be inside the image directory."
            )

        workspace.mkdir(parents=True, exist_ok=True)

        free_bytes = shutil.disk_usage(workspace).free

        self._report(
            f"Input validation passed: {len(image_files)} images."
        )
        self._report(
            f"Available workspace disk space: "
            f"{free_bytes / (1024 ** 3):.2f} GB"
        )

    def _preflight(self) -> None:
        report = self.toolchain.inspect()

        if not report.ready:
            failures = []

            for name, tool in report.tools.items():
                if not tool.available or not tool.launchable:
                    failures.append(
                        f"{name}: {tool.error or 'not launchable'}"
                    )

            raise ReconstructionError(
                "Required reconstruction tools are unavailable: "
                + "; ".join(failures)
            )

        self._report("COLMAP and OpenMVS preflight passed.")

    def _run_stage(
        self,
        name: str,
        command: list[str],
        cwd: Path,
        required_outputs: Optional[list[Path]] = None,
    ) -> None:
        assert self.runner is not None

        self._report(f"Stage started: {name}")

        output_snapshots = {}

        for output in required_outputs or []:
            if output.is_file():
                stat = output.stat()
                output_snapshots[output] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            else:
                output_snapshots[output] = None

        self.runner.run(
            command=command,
            stage=name,
            cwd=cwd,
            timeout=self.stage_timeout,
        )

        for output in required_outputs or []:
            if not output.exists():
                raise ReconstructionError(
                    f"Stage '{name}' did not produce: {output}"
                )

            if output.is_file():
                stat = output.stat()

                if stat.st_size == 0:
                    raise ReconstructionError(
                        f"Stage '{name}' produced an empty file: {output}"
                    )

                previous = output_snapshots[output]
                current = (stat.st_size, stat.st_mtime_ns)

                if previous is not None and current == previous:
                    raise ReconstructionError(
                        f"Stage '{name}' did not produce a fresh output: {output}"
                    )

        self.stages_completed.append(name)
        self._report(f"Stage validated: {name}")

    def _select_sparse_model(self) -> Path:
        assert self.paths is not None

        import struct

        models = [
            path for path in self.paths.sparse.iterdir()
            if path.is_dir()
        ]

        candidates = []

        for model in models:
            required = (
                "cameras.bin",
                "images.bin",
                "points3D.bin",
            )

            if not all((model / filename).is_file() for filename in required):
                continue

            images_file = model / "images.bin"

            try:
                with images_file.open("rb") as file:
                    header = file.read(8)

                if len(header) != 8:
                    continue

                registered_images = struct.unpack("<Q", header)[0]

                if registered_images > 0:
                    candidates.append((registered_images, model))

            except (OSError, struct.error):
                continue

        if not candidates:
            raise ReconstructionError(
                "COLMAP did not produce a valid sparse model "
                "with registered images."
            )

        return max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1].name,
            ),
        )[1]

    def _validate_database(self) -> None:
        assert self.paths is not None

        if not self.paths.database.is_file():
            raise ReconstructionError(
                "COLMAP database was not created."
            )

        try:
            with sqlite3.connect(str(self.paths.database)) as connection:
                image_count = connection.execute(
                    "SELECT COUNT(*) FROM images"
                ).fetchone()[0]

                keypoint_count = connection.execute(
                    "SELECT COALESCE(SUM(rows), 0) FROM keypoints"
                ).fetchone()[0]

                match_count = connection.execute(
                    "SELECT COALESCE(SUM(rows), 0) FROM matches"
                ).fetchone()[0]

                verified_match_count = connection.execute(
                    "SELECT COALESCE(SUM(rows), 0) "
                    "FROM two_view_geometries"
                ).fetchone()[0]

                image_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT image_id FROM images ORDER BY image_id"
                    ).fetchall()
                ]

                verified_pairs = connection.execute(
                    "SELECT pair_id, rows FROM two_view_geometries "
                    "WHERE rows > 0 ORDER BY pair_id"
                ).fetchall()

                camera_network = analyze_camera_pair_graph(
                    image_ids=image_ids,
                    verified_pairs=verified_pairs,
                )

        except (sqlite3.Error, ValueError) as exc:
            raise ReconstructionError(
                f"Could not validate COLMAP database: {exc}"
            ) from exc

        if image_count < 2 or keypoint_count == 0:
            raise ReconstructionError(
                "COLMAP database contains insufficient image features."
            )

        if match_count == 0:
            raise ReconstructionError(
                "COLMAP database contains no putative feature matches."
            )

        if verified_match_count == 0:
            raise ReconstructionError(
                "COLMAP database contains no geometrically verified matches."
            )

        self.database_diagnostics = camera_network

        self._report(
            f"Database validated: {image_count} images, "
            f"{keypoint_count} keypoints, {match_count} putative matches, "
            f"{verified_match_count} verified matches."
        )

        self._report(
            f"Camera network: {camera_network['verified_pair_count']} "
            f"verified pairs, {camera_network['component_count']} "
            f"components, {len(camera_network['isolated_image_ids'])} "
            f"isolated images."
        )

    def _collect_diagnostics(self, sparse_model: Path) -> dict:
        assert self.paths is not None

        supported = {
            ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"
        }

        input_image_count = sum(
            1 for item in self.paths.images.iterdir()
            if item.is_file() and item.suffix.lower() in supported
        )

        try:
            images = read_images_binary(sparse_model / "images.bin")
            points = read_points3d_binary(sparse_model / "points3D.bin")

            report = build_sparse_diagnostics_report(
                images=images,
                points=points,
                input_image_count=input_image_count,
            )

        except (OSError, ValueError, KeyError, struct.error) as exc:
            raise ReconstructionError(
                f"Could not calculate sparse reconstruction diagnostics: {exc}"
            ) from exc

        report["camera_network"] = self.database_diagnostics
        self.diagnostics = report

        self._report(
            "Sparse diagnostics: "
            f"{report['images']['registered']}/"
            f"{report['images']['input']} images registered, "
            f"{report['sparse_points']['count']} sparse points."
        )

        return report
    def _write_manifest(
        self,
        success: bool,
        started: float,
        final_model: Optional[Path] = None,
        error: Optional[str] = None,
    ) -> None:
        assert self.paths is not None

        data = {
            "success": success,
            "profile": self.config.profile,
            "configuration": asdict(self.config),
            "images": str(self.paths.images),
            "workspace": str(self.paths.workspace),
            "stages_completed": self.stages_completed,
            "duration_seconds": round(
                time.monotonic() - started, 2
            ),
            "final_model": (
                str(final_model) if final_model else None
            ),
            "diagnostics": self.diagnostics,
            "error": error,
        }

        self.paths.manifest.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def run(
        self,
        image_path: str | Path,
        workspace_path: str | Path,
    ) -> ReconstructionResult:
        started = time.monotonic()

        images = Path(image_path).resolve()
        workspace = Path(workspace_path).resolve()

        self.stages_completed = []
        self.diagnostics = None
        self.database_diagnostics = None

        self._validate_inputs(images, workspace)

        self.paths = ReconstructionPaths(
            images=images,
            workspace=workspace,
        )

        self.paths.logs.mkdir(parents=True, exist_ok=True)

        self.runner = ProcessRunner(
            log_dir=self.paths.logs,
            progress_callback=self._report,
        )

        try:
            self._preflight()

            colmap = COLMAPCommands(self.colmap_path)
            mvs = OpenMVSCommands(self.openmvs_dir)

            paths = self.paths

            paths.sparse.mkdir(parents=True, exist_ok=True)

            self._run_stage(
                "feature_extraction",
                colmap.feature_extractor(
                    str(paths.database),
                    str(paths.images),
                    self.config,
                ),
                paths.workspace,
                [paths.database],
            )

            self._run_stage(
                "image_matching",
                colmap.match(
                    str(paths.database),
                    self.config,
                ),
                paths.workspace,
            )

            self._validate_database()

            self._run_stage(
                "sparse_mapping",
                colmap.mapper(
                    str(paths.database),
                    str(paths.images),
                    str(paths.sparse),
                    self.config,
                ),
                paths.workspace,
                [paths.sparse],
            )

            sparse_model = self._select_sparse_model()
            self._collect_diagnostics(sparse_model)

            self._run_stage(
                "image_undistortion",
                colmap.image_undistorter(
                    str(paths.images),
                    str(sparse_model),
                    str(paths.undistorted),
                ),
                paths.workspace,
                [
                    paths.undistorted / "images",
                    paths.undistorted / "sparse",
                ],
            )

            self._run_stage(
                "openmvs_import",
                mvs.interface_colmap(
                    str(paths.undistorted),
                    str(paths.mvs_scene),
                    num_threads=self.config.num_threads,
                ),
                paths.workspace,
                [paths.mvs_scene],
            )

            self._run_stage(
                "dense_reconstruction",
                mvs.densify(
                    str(paths.mvs_scene),
                    self.config,
                ),
                paths.workspace,
                [paths.dense_scene],
            )

            self._run_stage(
                "mesh_reconstruction",
                mvs.reconstruct_mesh(
                    str(paths.dense_scene),
                    str(paths.mesh),
                    self.config,
                ),
                paths.workspace,
                [paths.mesh],
            )

            final_model = paths.mesh

            if self.config.refine_mesh:
                self._run_stage(
                    "mesh_refinement",
                    mvs.refine_mesh(
                        str(paths.dense_scene),
                        str(paths.mesh),
                        str(paths.refined_mesh),
                        self.config,
                    ),
                    paths.workspace,
                    [paths.refined_mesh],
                )
                final_model = paths.refined_mesh

            if self.config.texture_mesh:
                self._run_stage(
                    "mesh_texturing",
                    mvs.texture_mesh(
                        str(paths.dense_scene),
                        str(final_model),
                        str(paths.textured_mesh),
                        self.config,
                    ),
                    paths.workspace,
                    [paths.textured_mesh],
                )
                final_model = paths.textured_mesh

            self._write_manifest(
                True,
                started,
                final_model=final_model,
            )

            self._report("Reconstruction completed successfully.")

            return ReconstructionResult(
                success=True,
                workspace=paths.workspace,
                final_model=final_model,
                stages_completed=list(self.stages_completed),
                duration_seconds=time.monotonic() - started,
                manifest_path=paths.manifest,
                diagnostics=self.diagnostics,
            )

        except Exception as exc:
            if self.paths is not None:
                self._write_manifest(
                    False,
                    started,
                    error=str(exc),
                )

            self._report(f"Reconstruction failed: {exc}")
            raise








