import csv

import numpy as np
import pytest

from src.reconstruction_engine.diagnostic_export import (
    export_mesh_diagnostics,
)
from src.reconstruction_engine.quality import read_ply


def test_exports_coloured_components_and_diagnostic_files(tmp_path):
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0],
    ], dtype=float)

    faces = [[0, 1, 2], [3, 4, 5]]

    report = {
        "triangle_anomalies": [
            {"source_face_index": 1},
        ],
        "long_edge_anomalies": [
            {
                "length": 1.0,
                "vertex_indices": [3, 4],
                "midpoint": [5.5, 0, 0],
                "endpoint_a": [5, 0, 0],
                "endpoint_b": [6, 0, 0],
            },
        ],
        "component_details": [
            {
                "component": 0,
                "vertex_count": 3,
                "face_count": 1,
                "surface_area": 0.5,
                "centroid": [1 / 3, 1 / 3, 0],
                "bounding_box": {
                    "minimum": [0, 0, 0],
                    "maximum": [1, 1, 0],
                },
            },
            {
                "component": 1,
                "vertex_count": 3,
                "face_count": 1,
                "surface_area": 0.5,
                "centroid": [16 / 3, 1 / 3, 0],
                "bounding_box": {
                    "minimum": [5, 0, 0],
                    "maximum": [6, 1, 0],
                },
            },
        ],
    }

    result = export_mesh_diagnostics(
        vertices, faces, report, tmp_path
    )

    assert result["component_count"] == 2
    assert result["anomaly_face_count"] == 1
    assert result["long_edge_count"] == 1

    for key in (
        "components_mesh",
        "anomaly_mesh",
        "long_edges_csv",
        "components_csv",
    ):
        assert result[key].is_file()

    components = read_ply(result["components_mesh"])
    assert len(components["vertices"]) == 6
    assert len(components["faces"]) == 2

    anomalies = read_ply(result["anomaly_mesh"])
    assert len(anomalies["vertices"]) == 3
    assert len(anomalies["faces"]) == 1

    with result["long_edges_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert float(rows[0]["length"]) == pytest.approx(1.0)

    with result["components_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2


def test_rejects_non_triangular_faces(tmp_path):
    vertices = np.zeros((4, 3))
    faces = [[0, 1, 2, 3]]

    with pytest.raises(ValueError, match="triangular"):
        export_mesh_diagnostics(vertices, faces, {}, tmp_path)


def test_rejects_non_integer_face_indices(tmp_path):
    vertices = np.zeros((3, 3))
    faces = [[0.0, 1.0, 2.0]]

    with pytest.raises(ValueError, match="integers"):
        export_mesh_diagnostics(vertices, faces, {}, tmp_path)


def test_rejects_out_of_range_face_indices(tmp_path):
    vertices = np.zeros((3, 3))
    faces = [[0, 1, 5]]

    with pytest.raises(ValueError, match="out-of-range"):
        export_mesh_diagnostics(vertices, faces, {}, tmp_path)


def test_exports_empty_anomaly_mesh(tmp_path):
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ], dtype=float)

    faces = [[0, 1, 2]]

    result = export_mesh_diagnostics(
        vertices,
        faces,
        {
            "triangle_anomalies": [],
            "long_edge_anomalies": [],
            "component_details": [],
        },
        tmp_path,
    )

    anomaly_mesh = read_ply(result["anomaly_mesh"])

    assert len(anomaly_mesh["vertices"]) == 0
    assert len(anomaly_mesh["faces"]) == 0
