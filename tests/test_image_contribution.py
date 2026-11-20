import numpy as np
import pytest
from types import SimpleNamespace

from src.image_contribution import (
    ContributionConfig,
    score_image_contributions,
)


def make_feature(points, width=100, height=100):
    return SimpleNamespace(
        keypoints=np.asarray(points, dtype=float).reshape(-1, 2),
        image_shape=(height, width),
    )


def make_scene():
    K = np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])

    points_3d = np.array([
        [-0.5, -0.3, 5.0],
        [0.0, 0.0, 5.0],
        [0.5, 0.3, 5.0],
        [-0.3, 0.4, 6.0],
    ])

    centres = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    ]

    poses = {}
    feature_points = [[] for _ in centres]
    tracks = []

    for image_index, centre in enumerate(centres):
        R = np.eye(3)
        t = -centre
        poses[image_index] = (R, t)

        for point_id, X in enumerate(points_3d):
            camera_point = R @ X + t
            projected = K @ camera_point
            uv = projected[:2] / projected[2]
            keypoint_index = len(feature_points[image_index])
            feature_points[image_index].append(uv)
            tracks.append((point_id, image_index, keypoint_index))

    features = [make_feature(p) for p in feature_points]

    pairs = [
        SimpleNamespace(
            i=0, j=1, raw_match_count=100,
            fundamental_inliers=80, inliers=70,
        ),
        SimpleNamespace(
            i=1, j=2, raw_match_count=100,
            fundamental_inliers=75, inliers=65,
        ),
    ]

    return SimpleNamespace(
        features=features,
        pairs=pairs,
        components=[[0, 1, 2]],
        camera_poses=poses,
        sparse_points=points_3d,
        point_tracks=tracks,
        camera_matrix=K,
    )


def test_scores_all_images_and_returns_finite_scores():
    report = score_image_contributions(make_scene())

    assert len(report.images) == 3
    for item in report.images:
        assert np.isfinite(item.contribution_score)
        assert 0.0 <= item.contribution_score <= 100.0
        assert item.point_track_count == 4


def test_pair_support_is_aggregated_for_both_images():
    report = score_image_contributions(make_scene())

    first = report.by_image_index(0)
    middle = report.by_image_index(1)
    last = report.by_image_index(2)

    assert first.pair_count == 1
    assert middle.pair_count == 2
    assert last.pair_count == 1

    assert first.raw_matches == 100
    assert middle.raw_matches == 200
    assert last.raw_matches == 100


def test_synthetic_reprojection_error_is_near_zero():
    report = score_image_contributions(make_scene())

    for item in report.images:
        assert item.reprojection_observations == 4
        assert item.mean_reprojection_error_px is not None
        assert item.mean_reprojection_error_px < 1e-8


def test_parallax_is_measured_from_world_rays():
    report = score_image_contributions(make_scene())

    for item in report.images:
        assert item.valid_parallax_tracks > 0
        assert item.mean_parallax_deg is not None
        assert item.mean_parallax_deg > 0


def test_disconnected_components_are_not_compared_for_coverage():
    scene = make_scene()
    scene.components = [[0, 1], [2]]

    report = score_image_contributions(scene)

    assert report.component_by_image[0] == report.component_by_image[1]
    assert report.component_by_image[2] != report.component_by_image[0]


def test_missing_geometry_returns_warning_without_crashing():
    scene = make_scene()
    scene.camera_matrix = None
    scene.camera_poses = {}
    scene.sparse_points = None

    report = score_image_contributions(scene)

    assert len(report.images) == 3
    assert report.warnings
    assert all(item.mean_reprojection_error_px is None for item in report.images)


def test_invalid_track_indices_are_ignored():
    scene = make_scene()
    scene.point_tracks.extend([
        (999, 0, 0),
        (0, 999, 0),
        (0, 0, 999),
        ("bad", 0, 0),
        (1, 1, -1),
    ])

    report = score_image_contributions(scene)

    assert all(item.point_track_count == 4 for item in report.images)


def test_empty_result_is_supported():
    scene = SimpleNamespace(
        features=[],
        pairs=[],
        components=[],
        camera_poses={},
        sparse_points=None,
        point_tracks=[],
        camera_matrix=None,
    )

    report = score_image_contributions(scene)

    assert report.images == []
    assert report.component_by_image == {}


def test_config_rejects_invalid_weights_and_grid_size():
    with pytest.raises(ValueError):
        ContributionConfig(pair_support_weight=-1)

    with pytest.raises(ValueError):
        ContributionConfig(
            pair_support_weight=0,
            track_support_weight=0,
            image_coverage_weight=0,
            reprojection_weight=0,
            parallax_weight=0,
            distinctive_coverage_weight=0,
        )

    with pytest.raises(ValueError):
        ContributionConfig(grid_size=0)


def test_same_component_images_get_coverage_metrics():
    report = score_image_contributions(make_scene())

    assert all(item.occupied_grid_cells > 0 for item in report.images)
    assert all(0.0 <= item.grid_coverage <= 1.0 for item in report.images)

def test_actual_aligner_classes_are_supported():
    import cv2
    from src.aligner import ImageFeatures, MatchPair

    features = [
        ImageFeatures(
            path="image_0.jpg",
            keypoints=[cv2.KeyPoint(10.0, 10.0, 1.0)],
            descriptors=np.zeros((1, 32), dtype=np.uint8),
            width=100,
            height=100,
        ),
        ImageFeatures(
            path="image_1.jpg",
            keypoints=[cv2.KeyPoint(20.0, 20.0, 1.0)],
            descriptors=np.zeros((1, 32), dtype=np.uint8),
            width=100,
            height=100,
        ),
    ]

    pair = MatchPair(
        idx1=0,
        idx2=1,
        matches=[],
        inliers=7,
        raw_match_count=10,
        fundamental_inliers=8,
    )

    result = SimpleNamespace(
        features=features,
        pairs=[pair],
        components=[[0, 1]],
        camera_poses={},
        sparse_points=None,
        point_tracks=[],
        camera_matrix=None,
    )

    report = score_image_contributions(result)

    first = report.by_image_index(0)
    second = report.by_image_index(1)

    assert first.pair_count == 1
    assert second.pair_count == 1
    assert first.raw_matches == 10
    assert first.fundamental_inliers == 8
    assert first.pose_inliers == 7
    assert first.geometric_inlier_ratio == pytest.approx(0.8)
    assert first.pose_retention_ratio == pytest.approx(0.875)
