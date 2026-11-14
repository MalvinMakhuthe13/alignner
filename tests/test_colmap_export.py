from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.aligner import AlignmentResult, ImageAligner, ImageFeatures


def _feature(path, points):
    keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return ImageFeatures(path, keypoints, np.empty((0, 32), dtype=np.uint8), image, 100, 100)


def test_colmap_export_writes_matching_image_and_point_tracks(tmp_path: Path):
    features = [
        _feature("a.jpg", [(10, 20), (30, 40)]),
        _feature("b.jpg", [(11, 21), (31, 41)]),
    ]
    poses = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (np.eye(3), np.array([[-1.0], [0.0], [0.0]])),
    }
    result = AlignmentResult(
        features=features,
        pairs=[],
        components=[[0, 1]],
        camera_poses=poses,
        sparse_points=np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 6.0]]),
        point_colors=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        point_tracks=[
            (0, 0, 0), (0, 1, 0),
            (1, 0, 1),  # Deliberately untracked in image 1: must not be exported.
        ],
    )
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])

    output = Path(ImageAligner().export_colmap(result, str(tmp_path / "model"), K))
    image_lines = [line for line in (output / "images.txt").read_text().splitlines()
                   if line and not line.startswith("#")]
    point_lines = [line for line in (output / "points3D.txt").read_text().splitlines()
                   if line and not line.startswith("#")]

    assert len(image_lines) == 4  # two image headers and two POINTS2D rows
    assert image_lines[1].split() == ["10.000000", "20.000000", "1"]
    assert image_lines[3].split() == ["11.000000", "21.000000", "1"]
    assert len(point_lines) == 1
    fields = point_lines[0].split()
    assert fields[0] == "1"
    assert fields[-4:] == ["1", "0", "2", "0"]
    assert (output / "images" / "a.jpg").exists()
    assert (output / "images" / "b.jpg").exists()


def test_colmap_export_requires_registered_camera(tmp_path: Path):
    result = AlignmentResult(
        features=[_feature("a.jpg", [(10, 20)])],
        pairs=[],
        components=[[0]],
    )
    K = np.eye(3)

    try:
        ImageAligner().export_colmap(result, str(tmp_path / "empty"), K)
    except ValueError as exc:
        assert "No registered camera poses" in str(exc)
    else:
        raise AssertionError("Expected export to reject an unregistered reconstruction")
