import struct

import pytest

from src.reconstruction_engine.diagnostics import (
    read_images_binary,
    read_points3d_binary,
    calculate_reprojection_statistics,
    decode_colmap_pair_id,
)


def test_reads_registered_images(tmp_path):
    path = tmp_path / "images.bin"

    with path.open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
        f.write(struct.pack("<3d", 0.0, 0.0, 0.0))
        f.write(struct.pack("<i", 1))
        f.write(b"photo.jpg\x00")
        f.write(struct.pack("<Q", 0))

    images = read_images_binary(path)

    assert len(images) == 1
    assert images[0]["name"] == "photo.jpg"
    assert images[0]["registered"] is True


def test_reads_sparse_point_track_length(tmp_path):
    path = tmp_path / "points3D.bin"

    with path.open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<3d", 1.0, 2.0, 3.0))
        f.write(struct.pack("<3B", 255, 128, 64))
        f.write(struct.pack("<d", 0.5))
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<ii", 1, 0))
        f.write(struct.pack("<ii", 2, 1))

    points = read_points3d_binary(path)

    assert len(points) == 1
    assert points[0]["track_length"] == 2
    assert points[0]["error"] == pytest.approx(0.5)


def test_calculates_reprojection_statistics():
    stats = calculate_reprojection_statistics([1.0, 2.0, 3.0, 4.0])

    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["maximum"] == pytest.approx(4.0)


def test_empty_reprojection_errors_are_safe():
    stats = calculate_reprojection_statistics([])

    assert stats["mean"] is None
    assert stats["median"] is None
    assert stats["maximum"] is None


def test_rejects_truncated_image_binary(tmp_path):
    path = tmp_path / "images.bin"
    path.write_bytes(b"\x01")

    with pytest.raises((ValueError, EOFError)):
        read_images_binary(path)

def test_reads_image_pose_and_2d_observations(tmp_path):
    path = tmp_path / "images.bin"

    with path.open("wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<i", 7))
        f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
        f.write(struct.pack("<3d", 1.0, 2.0, 3.0))
        f.write(struct.pack("<i", 2))
        f.write(b"front.jpg\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<ddq", 120.5, 240.5, 99))

    images = read_images_binary(path)

    assert images[0]["image_id"] == 7
    assert images[0]["camera_id"] == 2
    assert images[0]["tvec"] == (1.0, 2.0, 3.0)
    assert images[0]["observations"][0]["point3d_id"] == 99


def test_reprojection_statistics_ignore_non_finite_values():
    stats = calculate_reprojection_statistics(
        [1.0, 2.0, float("nan"), float("inf")]
    )

    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(1.5)

def test_summarizes_sparse_reconstruction_quality():
    from src.reconstruction_engine.diagnostics import (
        summarize_sparse_reconstruction,
    )

    images = [
        {"image_id": 1, "name": "a.jpg", "registered": True},
        {"image_id": 2, "name": "b.jpg", "registered": True},
        {"image_id": 3, "name": "c.jpg", "registered": True},
        {"image_id": 4, "name": "d.jpg", "registered": True},
    ]

    points = [
        {"track_length": 3, "error": 0.5},
        {"track_length": 2, "error": 1.0},
        {"track_length": 4, "error": 1.5},
    ]

    report = summarize_sparse_reconstruction(
        images=images,
        points=points,
        input_image_count=5,
    )

    assert report["images"]["input"] == 5
    assert report["images"]["registered"] == 4
    assert report["images"]["registration_rate_percent"] == 80.0

    assert report["sparse_points"]["count"] == 3
    assert report["tracks"]["mean_length"] == 3.0

    assert report["reprojection_error"]["mean"] == pytest.approx(1.0)
    assert report["reprojection_error"]["median"] == pytest.approx(1.0)
    assert report["reprojection_error"]["maximum"] == pytest.approx(1.5)


def test_sparse_report_handles_no_registered_images_or_points():
    from src.reconstruction_engine.diagnostics import (
        summarize_sparse_reconstruction,
    )

    report = summarize_sparse_reconstruction(
        images=[],
        points=[],
        input_image_count=5,
    )

    assert report["images"]["registered"] == 0
    assert report["images"]["registration_rate_percent"] == 0.0
    assert report["sparse_points"]["count"] == 0
    assert report["tracks"]["mean_length"] is None
    assert report["reprojection_error"]["mean"] is None


def test_sparse_report_rejects_invalid_input_image_count():
    from src.reconstruction_engine.diagnostics import (
        summarize_sparse_reconstruction,
    )

    with pytest.raises(ValueError):
        summarize_sparse_reconstruction(
            images=[],
            points=[],
            input_image_count=-1,
        )

def test_calculates_camera_centres_from_identity_poses():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_coverage,
    )

    images = [
        {
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
        {
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (-2.0, 0.0, 0.0),
        },
    ]

    report = calculate_camera_coverage(images)

    assert report["camera_count"] == 2
    assert report["centres"][0] == (0.0, 0.0, 0.0)
    assert report["centres"][1] == (2.0, 0.0, 0.0)
    assert report["baseline_span"] == pytest.approx(2.0)


def test_camera_coverage_handles_empty_input():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_coverage,
    )

    report = calculate_camera_coverage([])

    assert report["camera_count"] == 0
    assert report["centres"] == []
    assert report["baseline_span"] is None


def test_camera_coverage_rejects_invalid_quaternion():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_coverage,
    )

    images = [
        {
            "qvec": (0.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
        }
    ]

    with pytest.raises(ValueError):
        calculate_camera_coverage(images)

def test_calculates_track_quality_distribution():
    from src.reconstruction_engine.diagnostics import (
        calculate_track_quality,
    )

    points = [
        {"track_length": 2},
        {"track_length": 3},
        {"track_length": 4},
        {"track_length": 6},
    ]

    report = calculate_track_quality(points)

    assert report["point_count"] == 4
    assert report["mean_track_length"] == pytest.approx(3.75)
    assert report["points_seen_by_3_or_more_cameras"] == 3
    assert report["percentage_seen_by_3_or_more_cameras"] == 75.0


def test_track_quality_handles_empty_cloud():
    from src.reconstruction_engine.diagnostics import (
        calculate_track_quality,
    )

    report = calculate_track_quality([])

    assert report["point_count"] == 0
    assert report["mean_track_length"] is None
    assert report["percentage_seen_by_3_or_more_cameras"] is None


def test_calculates_reprojection_error_thresholds():
    from src.reconstruction_engine.diagnostics import (
        calculate_reprojection_quality,
    )

    report = calculate_reprojection_quality(
        [0.5, 1.5, 2.5, 4.5]
    )

    assert report["count"] == 4
    assert report["above_2px"] == 2
    assert report["above_4px"] == 1
    assert report["percentage_above_2px"] == 50.0


def test_reprojection_quality_handles_empty_cloud():
    from src.reconstruction_engine.diagnostics import (
        calculate_reprojection_quality,
    )

    report = calculate_reprojection_quality([])

    assert report["count"] == 0
    assert report["above_2px"] == 0
    assert report["percentage_above_2px"] is None

def test_builds_complete_sparse_diagnostics_report():
    from src.reconstruction_engine.diagnostics import (
        build_sparse_diagnostics_report,
    )

    images = [
        {
            "image_id": 1,
            "name": "a.jpg",
            "registered": True,
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
        {
            "image_id": 2,
            "name": "b.jpg",
            "registered": True,
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (-2.0, 0.0, 0.0),
        },
    ]

    points = [
        {"track_length": 3, "error": 0.5},
        {"track_length": 2, "error": 1.0},
        {"track_length": 4, "error": 2.5},
    ]

    report = build_sparse_diagnostics_report(
        images=images,
        points=points,
        input_image_count=3,
    )

    assert report["images"]["registration_rate_percent"] == pytest.approx(
        200 / 3
    )
    assert report["sparse_points"]["count"] == 3
    assert report["tracks"]["points_seen_by_3_or_more_cameras"] == 2
    assert report["reprojection_quality"]["above_2px"] == 1
    assert report["camera_coverage"]["camera_count"] == 2
    assert report["camera_coverage"]["baseline_span"] == pytest.approx(2.0)



def test_camera_point_support_counts_unique_points_and_tracks():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_point_support,
    )

    images = [
        {
            "image_id": 1,
            "name": "front.jpg",
            "observations": [
                {"x": 10.0, "y": 20.0, "point3d_id": 101},
                {"x": 11.0, "y": 21.0, "point3d_id": 101},
                {"x": 12.0, "y": 22.0, "point3d_id": 102},
                {"x": 13.0, "y": 23.0, "point3d_id": -1},
                {"x": 14.0, "y": 24.0, "point3d_id": 999},
            ],
        },
        {
            "image_id": 2,
            "name": "side.jpg",
            "observations": [
                {"x": 30.0, "y": 40.0, "point3d_id": 101},
                {"x": 31.0, "y": 41.0, "point3d_id": 103},
            ],
        },
    ]

    points = [
        {"point3d_id": 101, "track_length": 4},
        {"point3d_id": 102, "track_length": 2},
        {"point3d_id": 103, "track_length": 3},
    ]

    report = calculate_camera_point_support(images, points)

    assert report["camera_count"] == 2
    assert report["total_observations"] == 7
    assert report["total_triangulated_observations"] == 5
    assert report["total_untriangulated_observations"] == 1
    assert report["total_unresolved_point_references"] == 1

    front = report["cameras"][0]
    assert front["unique_3d_points"] == 2
    assert front["triangulated_observation_count"] == 3
    assert front["unique_points_by_track_length"]["2"] == 1
    assert front["unique_points_by_track_length"]["4_plus"] == 1


def test_camera_point_support_handles_missing_observations():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_point_support,
    )

    report = calculate_camera_point_support(
        images=[{"image_id": 1, "name": "empty.jpg"}],
        points=[],
    )

    assert report["camera_count"] == 1
    assert report["cameras"][0]["observation_count"] == 0
    assert report["cameras"][0]["unique_3d_points"] == 0


def test_complete_sparse_report_includes_camera_point_support():
    from src.reconstruction_engine.diagnostics import (
        build_sparse_diagnostics_report,
    )

    report = build_sparse_diagnostics_report(
        images=[
            {
                "image_id": 1,
                "name": "a.jpg",
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "observations": [
                    {"x": 10.0, "y": 20.0, "point3d_id": 10},
                ],
            },
        ],
        points=[
            {"point3d_id": 10, "track_length": 2, "error": 0.5},
        ],
        input_image_count=1,
    )

    assert report["camera_point_support"]["camera_count"] == 1
    assert report["camera_point_support"]["cameras"][0]["unique_3d_points"] == 1


def test_complete_report_identifies_measurement_limitations():
    from src.reconstruction_engine.diagnostics import (
        build_sparse_diagnostics_report,
    )

    report = build_sparse_diagnostics_report(
        images=[],
        points=[],
        input_image_count=0,
    )

    assert report["limitations"]


def test_decodes_valid_colmap_pair_id():
    from src.reconstruction_engine.diagnostics import COLMAP_MAX_IMAGE_ID

    assert decode_colmap_pair_id(COLMAP_MAX_IMAGE_ID + 2) == (1, 2)
    assert decode_colmap_pair_id(10 * COLMAP_MAX_IMAGE_ID + 25) == (10, 25)


@pytest.mark.parametrize("pair_id", [0, -1, -100])
def test_rejects_nonpositive_colmap_pair_ids(pair_id):
    with pytest.raises(ValueError):
        decode_colmap_pair_id(pair_id)


@pytest.mark.parametrize("pair_id", [True, False, 1.5, "123", None])
def test_rejects_noninteger_colmap_pair_ids(pair_id):
    with pytest.raises(ValueError):
        decode_colmap_pair_id(pair_id)


@pytest.mark.parametrize(
    "pair_id",
    [
        2 * 2_147_483_647 + 1,
        2_147_483_647,
        2_147_483_647 + 1,
    ],
)
def test_rejects_invalid_colmap_pair_order_or_encoding(pair_id):
    with pytest.raises(ValueError):
        decode_colmap_pair_id(pair_id)

def test_camera_pose_quality_measures_position_and_orientation():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_pose_quality,
    )

    images = [
        {
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
        {
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (-2.0, 0.0, 0.0),
        },
    ]

    report = calculate_camera_pose_quality(images)

    assert report["camera_count"] == 2
    assert report["position_spread"]["x"] == pytest.approx(2.0)
    assert report["pairwise_position_distance"]["count"] == 1
    assert report["pairwise_position_distance"]["minimum"] == pytest.approx(2.0)
    assert report["pairwise_position_distance"]["maximum"] == pytest.approx(2.0)
    assert report["viewing_direction_diversity"]["mean_degrees"] == pytest.approx(0.0)
    assert report["relative_rotation_diversity"]["mean_degrees"] == pytest.approx(0.0)


def test_camera_pose_quality_detects_ninety_degree_rotation():
    import math

    from src.reconstruction_engine.diagnostics import (
        calculate_camera_pose_quality,
    )

    half_angle = math.radians(45.0)
    images = [
        {
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
        {
            "qvec": (math.cos(half_angle), 0.0, math.sin(half_angle), 0.0),
            "tvec": (0.0, 0.0, 0.0),
        },
    ]

    report = calculate_camera_pose_quality(images)

    assert report["viewing_direction_diversity"]["mean_degrees"] == pytest.approx(90.0)
    assert report["relative_rotation_diversity"]["mean_degrees"] == pytest.approx(90.0)


def test_camera_pose_quality_handles_empty_and_single_camera():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_pose_quality,
    )

    empty = calculate_camera_pose_quality([])
    assert empty["camera_count"] == 0
    assert empty["position_spread"] is None
    assert empty["pairwise_position_distance"]["count"] == 0
    assert empty["viewing_direction_diversity"]["mean_degrees"] is None

    single = calculate_camera_pose_quality([
        {
            "qvec": (1.0, 0.0, 0.0, 0.0),
            "tvec": (1.0, 2.0, 3.0),
        }
    ])

    assert single["camera_count"] == 1
    assert single["pairwise_position_distance"]["count"] == 0
    assert single["pairwise_position_distance"]["minimum"] is None
    assert single["relative_rotation_diversity"]["mean_degrees"] is None


def test_camera_pose_quality_rejects_non_finite_translation():
    from src.reconstruction_engine.diagnostics import (
        calculate_camera_pose_quality,
    )

    with pytest.raises(ValueError, match="non-finite"):
        calculate_camera_pose_quality([
            {
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, float("inf"), 0.0),
            }
        ])

