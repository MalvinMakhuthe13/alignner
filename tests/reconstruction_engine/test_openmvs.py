from pathlib import Path

import pytest

from src.reconstruction_engine.config import ReconstructionConfig
from src.reconstruction_engine.openmvs import OpenMVSCommands


@pytest.fixture
def openmvs(tmp_path):
    executable_names = [
        "InterfaceCOLMAP.exe",
        "DensifyPointCloud.exe",
        "ReconstructMesh.exe",
        "RefineMesh.exe",
        "TextureMesh.exe",
    ]

    for name in executable_names:
        (tmp_path / name).touch()

    return OpenMVSCommands(str(tmp_path))


@pytest.fixture
def config():
    return ReconstructionConfig.from_profile(
        "balanced",
        num_threads=4,
    )


def test_interface_colmap_command(openmvs):
    command = openmvs.interface_colmap(
        "undistorted",
        "scene.mvs",
        "images",
        4,
    )

    assert command[0].endswith("InterfaceCOLMAP.exe")
    assert command[1:3] == ["--input-file", "undistorted"]
    assert "--output-file" in command
    assert "--image-folder" in command
    assert "--max-threads" in command


def test_densify_command(openmvs, config):
    command = openmvs.densify("scene.mvs", config)

    assert command[0].endswith("DensifyPointCloud.exe")
    assert "scene.mvs" in command
    assert "--resolution-level" in command
    assert "--number-views" in command
    assert "--number-views-fuse" not in command
    assert "--estimate-colors" in command
    assert "--max-threads" in command


def test_reconstruct_mesh_command(openmvs, config):
    command = openmvs.reconstruct_mesh(
        "scene.mvs",
        "mesh.ply",
        config,
    )

    assert command[0].endswith("ReconstructMesh.exe")
    assert "--input-file" in command
    assert "--output-file" in command
    assert "--min-point-distance" in command


def test_refine_mesh_command(openmvs, config):
    command = openmvs.refine_mesh(
        "scene.mvs",
        "mesh.ply",
        "refined.ply",
        config,
    )

    assert command[0].endswith("RefineMesh.exe")
    assert "--input-file" in command
    assert "--mesh-file" in command
    assert "--output-file" in command
    assert "--resolution-level" in command


def test_texture_mesh_command(openmvs, config):
    command = openmvs.texture_mesh(
        "scene.mvs",
        "refined.ply",
        "textured.mvs",
        config,
    )

    assert command[0].endswith("TextureMesh.exe")
    assert "--input-file" in command
    assert "--mesh-file" in command
    assert "--output-file" in command


def test_missing_executable_raises_error(tmp_path):
    commands = OpenMVSCommands(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        commands.densify(
            "scene.mvs",
            ReconstructionConfig(),
        )

