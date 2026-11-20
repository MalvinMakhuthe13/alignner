import cv2
import numpy as np

from src.aligner import ImageAligner, ImageFeatures, MatchPair


def make_scene(camera_count=3, corrupt_ids=None):
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    point_true = np.array([0.2, 0.1, 5.0])
    corrupt_ids = set(corrupt_ids or [])

    camera_centres = [
        np.array([float(i), 0.0, 0.0])
        for i in range(camera_count)
    ]

    poses = {}
    features = []

    for image_id, centre in enumerate(camera_centres):
        R = np.eye(3)
        t = -centre.reshape(3, 1)
        poses[image_id] = (R, t)

        camera_point = R @ point_true.reshape(3, 1) + t
        pixel = K @ camera_point
        uv = (pixel[:2] / pixel[2, 0]).ravel()

        if image_id in corrupt_ids:
            uv += np.array([20.0, 0.0])

        keypoint = cv2.KeyPoint(float(uv[0]), float(uv[1]), 1)

        features.append(
            ImageFeatures(
                f"camera_{image_id}.jpg",
                [keypoint],
                np.empty((0, 32), dtype=np.uint8),
                None,
                640,
                480,
            )
        )

    pairs = [
        MatchPair(i, j, [cv2.DMatch(0, 0, 0.0)], 1)
        for i in range(camera_count)
        for j in range(i + 1, camera_count)
    ]

    return features, pairs, poses, K, point_true


def test_triangulation_recovers_clean_three_camera_point():
    features, pairs, poses, K, point_true = make_scene()

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K
    )

    assert points is not None
    assert len(points) == 1
    assert np.linalg.norm(points[0] - point_true) < 1e-5
    assert len(tracks) == 3


def test_triangulation_recovers_point_with_one_bad_observation():
    features, pairs, poses, K, point_true = make_scene(
        camera_count=4,
        corrupt_ids={1},
    )

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K
    )

    assert points is not None
    assert len(points) == 1
    assert np.linalg.norm(points[0] - point_true) < 1e-4

    # Three reliable cameras should support the reconstructed point.
    assert len(tracks) == 3
    assert {image_id for _, image_id, _ in tracks} == {0, 2, 3}


def test_triangulation_rejects_two_bad_of_four_observations():
    features, pairs, poses, K, _ = make_scene(
        camera_count=4,
        corrupt_ids={1, 2},
    )

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K
    )

    assert points is None
    assert colors is None
    assert tracks is None


def make_custom_scene(camera_centres, pixel_offsets=None, point_true=None):
    """Build a synthetic scene with configurable camera geometry and noise."""
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    if point_true is None:
        point_true = np.array([0.2, 0.1, 5.0])

    if pixel_offsets is None:
        pixel_offsets = {}

    poses = {}
    features = []

    for image_id, centre in enumerate(camera_centres):
        centre = np.asarray(centre, dtype=np.float64)

        R = np.eye(3)
        t = -centre.reshape(3, 1)
        poses[image_id] = (R, t)

        camera_point = R @ point_true.reshape(3, 1) + t
        pixel = K @ camera_point
        uv = (pixel[:2] / pixel[2, 0]).ravel()

        uv += np.asarray(
            pixel_offsets.get(image_id, [0.0, 0.0]),
            dtype=np.float64,
        )

        keypoint = cv2.KeyPoint(float(uv[0]), float(uv[1]), 1)

        features.append(
            ImageFeatures(
                f"custom_camera_{image_id}.jpg",
                [keypoint],
                np.empty((0, 32), dtype=np.uint8),
                None,
                640,
                480,
            )
        )

    pairs = [
        MatchPair(i, j, [cv2.DMatch(0, 0, 0.0)], 1)
        for i in range(len(camera_centres))
        for j in range(i + 1, len(camera_centres))
    ]

    return features, pairs, poses, K, point_true


def test_triangulation_handles_small_pixel_noise():
    centres = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]

    offsets = {
        0: [0.2, -0.1],
        1: [-0.3, 0.2],
        2: [0.1, 0.25],
        3: [-0.2, -0.15],
    }

    features, pairs, poses, K, truth = make_custom_scene(
        centres,
        pixel_offsets=offsets,
    )

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K,
        reproj_thresh=2.5,
    )

    assert points is not None
    assert len(points) == 1
    assert np.linalg.norm(points[0] - truth) < 0.02
    assert len(tracks) == 4


def test_triangulation_rejects_insufficient_parallax():
    centres = [
        [0.000, 0.0, 0.0],
        [0.001, 0.0, 0.0],
        [0.002, 0.0, 0.0],
        [0.003, 0.0, 0.0],
    ]

    features, pairs, poses, K, _ = make_custom_scene(centres)

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K,
        min_triangulation_angle_deg=0.5,
    )

    assert points is None
    assert colors is None
    assert tracks is None


def test_triangulation_handles_non_collinear_camera_geometry():
    centres = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]

    features, pairs, poses, K, truth = make_custom_scene(centres)

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K,
    )

    assert points is not None
    assert len(points) == 1
    assert np.linalg.norm(points[0] - truth) < 1e-5
    assert len(tracks) == 4

import pytest


@pytest.mark.parametrize("angle", [-0.1, 180.0, float("nan"), float("inf"), float("-inf")])
def test_triangulation_rejects_invalid_angle_threshold(angle):
    features, pairs, poses, K, _ = make_scene()

    with pytest.raises(ValueError, match="min_triangulation_angle_deg"):
        ImageAligner().triangulate_multiview(
            features, pairs, poses, K,
            min_triangulation_angle_deg=angle,
        )


def test_triangulation_accepts_zero_angle_threshold():
    features, pairs, poses, K, truth = make_scene()

    points, colors, tracks = ImageAligner().triangulate_multiview(
        features, pairs, poses, K,
        min_triangulation_angle_deg=0.0,
    )

    assert points is not None
    assert len(points) == 1
    assert np.linalg.norm(points[0] - truth) < 1e-5
