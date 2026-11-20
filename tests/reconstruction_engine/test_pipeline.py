import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.reconstruction_engine.pipeline as pipeline_module
from src.reconstruction_engine.config import ReconstructionConfig


def build_test_pipeline(monkeypatch, tmp_path, missing_output=None):
    images = tmp_path / "images"
    workspace = tmp_path / "workspace"

    images.mkdir()
    (images / "photo_01.jpg").write_bytes(b"image-one")
    (images / "photo_02.jpg").write_bytes(b"image-two")

    class FakeToolchain:
        def __init__(self, **kwargs):
            self.colmap_path = kwargs.get("colmap_path") or "fake-colmap"
            self.openmvs_dir = kwargs.get("openmvs_dir") or "fake-openmvs"

        def inspect(self):
            return SimpleNamespace(ready=True, tools={})

    class FakeCOLMAP:
        def __init__(self, executable):
            pass

        def feature_extractor(self, *args):
            return ["fake", "feature_extractor"]

        def match(self, *args):
            return ["fake", "matcher"]

        def mapper(self, *args):
            return ["fake", "mapper"]

        def image_undistorter(self, *args):
            return ["fake", "undistorter"]

    class FakeOpenMVS:
        def __init__(self, executable_dir):
            pass

        def interface_colmap(self, *args, **kwargs):
            return ["fake", "interface"]

        def densify(self, *args):
            return ["fake", "densify"]

        def reconstruct_mesh(self, *args):
            return ["fake", "mesh"]

        def refine_mesh(self, *args):
            return ["fake", "refine"]

        def texture_mesh(self, *args):
            return ["fake", "texture"]

    class FakeRunner:
        def __init__(self, log_dir, progress_callback=None, **kwargs):
            self.progress_callback = progress_callback

        def cancel(self):
            pass

        def run(self, command, stage, cwd, timeout=None):
            root = Path(cwd)

            if stage == missing_output:
                return SimpleNamespace(return_code=0, success=True)

            if stage == "feature_extraction":
                database = root / "database.db"
                with sqlite3.connect(database) as connection:
                    connection.execute(
                        "CREATE TABLE images (image_id INTEGER)"
                    )
                    connection.execute(
                        "CREATE TABLE keypoints "
                        "(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
                    )
                    connection.execute(
                        "CREATE TABLE matches "
                        "(pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
                    )
                    connection.execute(
                        "CREATE TABLE two_view_geometries "
                        "(pair_id INTEGER, rows INTEGER, cols INTEGER, "
                        "data BLOB, config INTEGER)"
                    )
                    connection.executemany(
                        "INSERT INTO images VALUES (?)",
                        [(1,), (2,)],
                    )
                    connection.execute(
                        "INSERT INTO keypoints "
                        "(image_id, rows, cols) VALUES (1, 100, 4)"
                    )
                    connection.execute(
                        "INSERT INTO matches "
                        "(pair_id, rows, cols) VALUES (1, 50, 2)"
                    )
                    connection.execute(
                        "INSERT INTO two_view_geometries "
                        "(pair_id, rows, cols, config) "
                        "VALUES (2147483649, 40, 2, 2)"
                    )

            elif stage == "sparse_mapping":
                model = root / "sparse" / "0"
                model.mkdir(parents=True, exist_ok=True)

                import struct

                (model / "cameras.bin").write_bytes(b"test-camera")

                with (model / "images.bin").open("wb") as f:
                    f.write(struct.pack("<Q", 2))
                    for image_id, name, tx in (
                        (1, "photo_01.jpg", 0.0),
                        (2, "photo_02.jpg", -2.0),
                    ):
                        f.write(struct.pack("<i", image_id))
                        f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
                        f.write(struct.pack("<3d", tx, 0.0, 0.0))
                        f.write(struct.pack("<i", 1))
                        f.write(name.encode("utf-8") + b"\x00")
                        f.write(struct.pack("<Q", 1))
                        f.write(struct.pack("<ddq", 100.0, 120.0, 1))

                with (model / "points3D.bin").open("wb") as f:
                    f.write(struct.pack("<Q", 1))
                    f.write(struct.pack("<Q", 1))
                    f.write(struct.pack("<3d", 1.0, 2.0, 3.0))
                    f.write(struct.pack("<3B", 255, 128, 64))
                    f.write(struct.pack("<d", 0.75))
                    f.write(struct.pack("<Q", 2))
                    f.write(struct.pack("<ii", 1, 0))
                    f.write(struct.pack("<ii", 2, 0))

            elif stage == "image_undistortion":
                (root / "undistorted" / "images").mkdir(
                    parents=True, exist_ok=True
                )
                (root / "undistorted" / "sparse").mkdir(
                    parents=True, exist_ok=True
                )

            elif stage == "openmvs_import":
                (root / "scene.mvs").write_bytes(b"scene")

            elif stage == "dense_reconstruction":
                (root / "scene_dense.mvs").write_bytes(b"dense-scene")

            elif stage == "mesh_reconstruction":
                (root / "mesh.ply").write_bytes(b"mesh")

            elif stage == "mesh_refinement":
                (root / "mesh_refined.ply").write_bytes(b"refined")

            elif stage == "mesh_texturing":
                (root / "mesh_textured.ply").write_bytes(b"textured")

            return SimpleNamespace(return_code=0, success=True)

    monkeypatch.setattr(
        pipeline_module, "Toolchain", FakeToolchain
    )
    monkeypatch.setattr(
        pipeline_module, "COLMAPCommands", FakeCOLMAP
    )
    monkeypatch.setattr(
        pipeline_module, "OpenMVSCommands", FakeOpenMVS
    )
    monkeypatch.setattr(
        pipeline_module, "ProcessRunner", FakeRunner
    )

    pipeline = pipeline_module.ReconstructionPipeline(
        config=ReconstructionConfig.from_profile("balanced")
    )

    return pipeline, images, workspace


def test_complete_simulated_reconstruction(monkeypatch, tmp_path):
    pipeline, images, workspace = build_test_pipeline(
        monkeypatch, tmp_path
    )

    result = pipeline.run(images, workspace)

    assert result.success
    assert result.final_model == workspace / "mesh_textured.ply"
    assert result.final_model.is_file()
    assert result.manifest_path.is_file()

    assert result.diagnostics is not None
    assert result.diagnostics["images"]["input"] == 2
    assert result.diagnostics["images"]["registered"] == 2
    assert result.diagnostics["images"]["registration_rate_percent"] == 100.0
    assert result.diagnostics["sparse_points"]["count"] == 1
    assert result.diagnostics["reprojection_error"]["mean"] == 0.75
    assert result.diagnostics["camera_coverage"]["baseline_span"] == 2.0
    assert result.diagnostics["tracks"]["mean_track_length"] == 2.0

    assert result.stages_completed == [
        "feature_extraction",
        "image_matching",
        "sparse_mapping",
        "image_undistortion",
        "openmvs_import",
        "dense_reconstruction",
        "mesh_reconstruction",
        "mesh_refinement",
        "mesh_texturing",
    ]

    assert pipeline_module.json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )["success"] is True


def test_pipeline_stops_when_required_output_is_missing(
    monkeypatch, tmp_path
):
    pipeline, images, workspace = build_test_pipeline(
        monkeypatch,
        tmp_path,
        missing_output="mesh_reconstruction",
    )

    with pytest.raises(
        pipeline_module.ReconstructionError,
        match="did not produce",
    ):
        pipeline.run(images, workspace)

    manifest = workspace / "reconstruction_manifest.json"
    assert manifest.is_file()

    data = pipeline_module.json.loads(
        manifest.read_text(encoding="utf-8")
    )

    assert data["success"] is False
    assert "mesh_reconstruction" not in data["stages_completed"]
    assert "did not produce" in data["error"]


def test_pipeline_rejects_nonempty_workspace(monkeypatch, tmp_path):
    pipeline, images, workspace = build_test_pipeline(
        monkeypatch, tmp_path
    )

    workspace.mkdir(parents=True, exist_ok=True)
    existing_file = workspace / "important_existing_file.txt"
    existing_file.write_text("preserve this data", encoding="utf-8")

    with pytest.raises(
        pipeline_module.ReconstructionError,
        match="workspace",
    ):
        pipeline.run(images, workspace)

    assert existing_file.read_text(encoding="utf-8") == "preserve this data"

def test_pipeline_rejects_stale_required_output(monkeypatch, tmp_path):
    pipeline, images, workspace = build_test_pipeline(
        monkeypatch, tmp_path
    )

    stale_output = workspace / "stale_output.ply"
    stale_output.parent.mkdir(parents=True, exist_ok=True)
    stale_output.write_bytes(b"old reconstruction")

    import time
    time.sleep(0.02)

    def fake_run(**kwargs):
        return None

    pipeline.runner = pipeline_module.ProcessRunner(log_dir=workspace / "logs")
    pipeline.runner.run = fake_run

    with pytest.raises(
        pipeline_module.ReconstructionError,
        match="did not produce",
    ):
        pipeline._run_stage(
            name="stale_output_test",
            command=["fake-command"],
            cwd=workspace,
            required_outputs=[stale_output],
        )


def test_sparse_model_selection_prefers_registered_images(
    monkeypatch, tmp_path
):
    pipeline, images, workspace = build_test_pipeline(
        monkeypatch, tmp_path
    )

    import struct

    sparse = workspace / "sparse"
    model_a = sparse / "0"
    model_b = sparse / "1"

    for model, registered_count, points_size in (
        (model_a, 2, 10000),
        (model_b, 5, 100),
    ):
        model.mkdir(parents=True, exist_ok=True)
        (model / "cameras.bin").write_bytes(b"camera")
        (model / "images.bin").write_bytes(
            struct.pack("<Q", registered_count)
        )
        (model / "points3D.bin").write_bytes(
            b"x" * points_size
        )

    pipeline.paths = SimpleNamespace(sparse=sparse)

    selected = pipeline._select_sparse_model()

    assert selected == model_b

def test_database_rejects_missing_geometric_verification(
    monkeypatch, tmp_path
):
    pipeline, images, workspace = build_test_pipeline(
        monkeypatch, tmp_path
    )

    workspace.mkdir(parents=True, exist_ok=True)
    database = workspace / "database.db"

    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE images (image_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE keypoints (image_id INTEGER, rows INTEGER)"
        )
        connection.execute(
            "CREATE TABLE matches (pair_id INTEGER, rows INTEGER)"
        )
        connection.execute(
            "CREATE TABLE two_view_geometries "
            "(pair_id INTEGER, rows INTEGER)"
        )

        connection.executemany(
            "INSERT INTO images VALUES (?)",
            [(1,), (2,)],
        )
        connection.execute(
            "INSERT INTO keypoints VALUES (1, 100)"
        )
        connection.execute(
            "INSERT INTO matches VALUES (1, 50)"
        )

    pipeline.paths = SimpleNamespace(database=database)

    with pytest.raises(
        pipeline_module.ReconstructionError,
        match="geometr",
    ):
        pipeline._validate_database()




