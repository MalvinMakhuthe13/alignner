import pytest

from src.reconstruction_engine.diagnostics import (
    COLMAP_MAX_IMAGE_ID,
    analyze_camera_pair_graph,
)


def pair_id(image_id_1, image_id_2):
    assert image_id_1 < image_id_2
    return image_id_1 * COLMAP_MAX_IMAGE_ID + image_id_2


def test_detects_disconnected_camera_groups():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2, 3, 4, 5],
        verified_pairs=[
            (pair_id(1, 2), 40),
            (pair_id(2, 3), 35),
            (pair_id(4, 5), 25),
        ],
    )

    assert report["component_count"] == 2
    assert report["components"] == [[1, 2, 3], [4, 5]]
    assert report["largest_component_size"] == 3
    assert report["largest_component_percentage"] == 60.0
    assert report["isolated_image_ids"] == []


def test_detects_isolated_cameras():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2, 3],
        verified_pairs=[
            (pair_id(1, 2), 20),
        ],
    )

    assert report["component_count"] == 2
    assert report["isolated_image_ids"] == [3]
    assert report["connected_image_count"] == 2


def test_calculates_camera_degrees_and_weak_connections():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2, 3, 4],
        verified_pairs=[
            (pair_id(1, 2), 20),
            (pair_id(2, 3), 30),
        ],
        weak_connection_threshold=2,
    )

    assert report["degree_by_image"] == {
        1: 1,
        2: 2,
        3: 1,
        4: 0,
    }
    assert report["weakly_connected_image_ids"] == [1, 3]
    assert report["isolated_image_ids"] == [4]


def test_calculates_verified_pair_coverage():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2, 3, 4],
        verified_pairs=[
            (pair_id(1, 2), 40),
            (pair_id(2, 3), 30),
            (pair_id(3, 4), 20),
        ],
    )

    assert report["possible_pair_count"] == 6
    assert report["verified_pair_count"] == 3
    assert report["pair_coverage_percent"] == 50.0
    assert report["verified_match_rows"] == 90


def test_ignores_zero_row_pairs():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2],
        verified_pairs=[
            (pair_id(1, 2), 0),
        ],
    )

    assert report["verified_pair_count"] == 0
    assert report["isolated_image_ids"] == [1, 2]


def test_reports_invalid_pair_ids():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2],
        verified_pairs=[
            (1, 20),
            (pair_id(1, 2), 30),
        ],
    )

    assert report["invalid_pair_count"] == 1
    assert report["invalid_pair_ids"] == [1]
    assert report["verified_pair_count"] == 1


def test_reports_pairs_referencing_missing_images():
    report = analyze_camera_pair_graph(
        image_ids=[1, 2],
        verified_pairs=[
            (pair_id(1, 3), 20),
        ],
    )

    assert report["dangling_pair_count"] == 1
    assert report["dangling_pairs"][0]["image_ids"] == [1, 3]
    assert report["verified_pair_count"] == 0


def test_rejects_duplicate_or_invalid_image_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_camera_pair_graph(
            image_ids=[1, 1],
            verified_pairs=[],
        )

    with pytest.raises(ValueError, match="valid positive"):
        analyze_camera_pair_graph(
            image_ids=[0, 1],
            verified_pairs=[],
        )


def test_rejects_invalid_threshold_and_match_counts():
    with pytest.raises(ValueError, match="threshold"):
        analyze_camera_pair_graph(
            image_ids=[1, 2],
            verified_pairs=[],
            weak_connection_threshold=0,
        )

    with pytest.raises(ValueError, match="row counts"):
        analyze_camera_pair_graph(
            image_ids=[1, 2],
            verified_pairs=[
                (pair_id(1, 2), -5),
            ],
        )
