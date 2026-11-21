"""Tests for component-aware visualisation helpers and export filtering.

These tests verify conventions and filtering behaviour without changing
reconstruction geometry.
"""
import numpy as np
from pathlib import Path

from src.aligner import AlignmentResult, ImageAligner, ImageFeatures


def _feature(path, points=None, w=100, h=100):
    import cv2
    if points is None:
        points = [(10, 20), (30, 40)]
    keypoints = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points]
    return ImageFeatures(path, keypoints, np.empty((0, 32), dtype=np.uint8), None, w, h)


def test_camera_center_openCV_convention():
    """Camera centre must be -R.T @ t under the stored OpenCV pose convention."""
    R = np.eye(3)
    t = np.array([[0.0], [0.0], [5.0]])
    center = -R.T @ t.reshape(3)
    np.testing.assert_allclose(center, [0, 0, -5])


def test_viewing_direction_is_R_T_column_2():
    """Camera +Z in world is the third column of R.T."""
    # Identity: +Z camera looks along +Z world
    R = np.eye(3)
    forward_world = R.T[:, 2]
    np.testing.assert_allclose(forward_world, [0, 0, 1], atol=1e-8)


def test_component_export_filters_cameras_and_points(tmp_path):
    features = [_feature("a.jpg"), _feature("b.jpg"), _feature("c.jpg")]
    poses = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (np.eye(3), np.array([[-1.0], [0.0], [0.0]])),
        2: (np.eye(3), np.array([[0.0], [0.0], [-2.0]])),
    }
    result = AlignmentResult(
        features=features,
        pairs=[],
        components=[[0, 1], [2]],
        camera_poses=poses,
        sparse_points=np.array([[0.0, 0.0, 5.0], [1.0, 1.0, 6.0], [10.0, 0.0, 8.0]]),
        point_colors=np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8),
        point_tracks=[
            (0, 0, 0), (0, 1, 0),
            (1, 0, 1), (1, 1, 1),
            (2, 2, 0), (2, 2, 1),
        ],
    )
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    out = Path(ImageAligner().export_colmap_component(result, 0, str(tmp_path / "c0"), K))
    images_txt = (out / "images.txt").read_text()
    assert "a.jpg" in images_txt
    assert "b.jpg" in images_txt
    assert "c.jpg" not in images_txt

    points_txt = (out / "points3D.txt").read_text()
    point_lines = [l for l in points_txt.splitlines() if l and not l.startswith("#")]
    assert len(point_lines) >= 1
    # The third image (component 2) must not appear in this export
    assert "c.jpg" not in (out / "images.txt").read_text()


def test_min_support_ratio_validation():
    aligner = ImageAligner()
    # Production default is still accepted
    try:
        # Just exercise the validator path via a dummy call that will fail later
        # We only care that 0.70 and 0.40 do not raise on the range check.
        from src.aligner import ImageAligner as IA
        # The check lives inside triangulate_multiview; call it with empty data
        # to hit the range validation early.
        IA().triangulate_multiview([], [], {}, np.eye(3), min_support_ratio=0.70)
    except ValueError as e:
        # Empty data may raise other errors; range error must not occur for 0.70
        assert "min_support_ratio" not in str(e)
    # 0.40 is now allowed (experimental)
    try:
        IA().triangulate_multiview([], [], {}, np.eye(3), min_support_ratio=0.40)
    except ValueError as e:
        assert "min_support_ratio" not in str(e)
    # Below 0.40 must still raise
    try:
        IA().triangulate_multiview([], [], {}, np.eye(3), min_support_ratio=0.3)
        assert False, "should have raised"
    except ValueError as e:
        assert "min_support_ratio" in str(e)
