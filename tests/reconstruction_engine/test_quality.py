from pathlib import Path
import struct

import numpy as np
import pytest

from src.reconstruction_engine.quality import (
    PLYFormatError,
    calculate_point_cloud_quality,
    read_ply,
)


def test_reads_ascii_point_cloud(tmp_path):
    path = tmp_path / "cloud_ascii.ply"
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "1 2 3\n"
        "4 5 6\n",
        encoding="ascii",
    )

    result = read_ply(path)

    assert result["vertices"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert result["faces"] == []
    assert result["metadata"]["vertex_count"] == 2


def test_reads_openmvs_binary_variable_length_properties(tmp_path):
    path = tmp_path / "openmvs_cloud.ply"

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float32 x\n"
        b"property float32 y\n"
        b"property float32 z\n"
        b"property uint8 red\n"
        b"property uint8 green\n"
        b"property uint8 blue\n"
        b"property float32 nx\n"
        b"property float32 ny\n"
        b"property float32 nz\n"
        b"property list uint8 uint32 view_indices\n"
        b"property list uint8 float32 view_weights\n"
        b"end_header\n"
    )

    with path.open("wb") as handle:
        handle.write(header)

        for x, y, z in [(1, 2, 3), (4, 5, 6)]:
            handle.write(struct.pack("<3f", x, y, z))
            handle.write(struct.pack("<3B", 255, 128, 64))
            handle.write(struct.pack("<3f", 0, 0, 1))
            handle.write(struct.pack("<B2I", 2, 1, 3))
            handle.write(struct.pack("<B2f", 2, 0.5, 0.5))

    result = read_ply(path)

    assert result["vertices"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert result["metadata"]["vertex_count"] == 2
    assert result["metadata"]["encoding"] == "binary_little_endian"


def test_reads_binary_mesh_faces(tmp_path):
    path = tmp_path / "mesh.ply"

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 3\n"
        b"property float32 x\n"
        b"property float32 y\n"
        b"property float32 z\n"
        b"element face 1\n"
        b"property list uint8 uint32 vertex_indices\n"
        b"end_header\n"
    )

    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0))
        handle.write(struct.pack("<B3I", 3, 0, 1, 2))

    result = read_ply(path)

    assert len(result["vertices"]) == 3
    assert result["faces"] == [[0, 1, 2]]
    assert result["metadata"]["face_count"] == 1


def test_rejects_truncated_binary_vertex_data(tmp_path):
    path = tmp_path / "truncated.ply"

    path.write_bytes(
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"end_header\n"
        + struct.pack("<2f", 1, 2)
    )

    with pytest.raises(PLYFormatError, match="Unexpected end"):
        read_ply(path)


def test_point_cloud_quality_reports_bounds_and_nonfinite_points():
    points = np.array([
        [0, 1, 2],
        [4, 5, 6],
        [float("nan"), 0, 0],
    ])

    report = calculate_point_cloud_quality(points)

    assert report["point_count"] == 3
    assert report["finite_point_count"] == 2
    assert report["non_finite_point_count"] == 1
    assert report["bounding_box"]["dimensions"] == [4, 4, 4]


def test_point_cloud_quality_handles_empty_cloud():
    report = calculate_point_cloud_quality(np.empty((0, 3)))

    assert report["point_count"] == 0
    assert report["finite_point_count"] == 0
    assert report["bounding_box"] is None


def test_rejects_invalid_vertex_array_shape():
    with pytest.raises(ValueError, match="shape"):
        calculate_point_cloud_quality(np.array([1, 2, 3]))



def test_reads_variable_lists_with_different_lengths_and_property_order(tmp_path):
    path = tmp_path / "variable_lengths.ply"

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 3\n"
        b"property float32 x\n"
        b"property list uint8 uint32 view_indices\n"
        b"property float32 y\n"
        b"property float32 z\n"
        b"property list uint8 float32 view_weights\n"
        b"end_header\n"
    )

    with path.open("wb") as handle:
        handle.write(header)

        records = [
            (1, [2], 10, 100, [0.5]),
            (2, [3, 4, 5], 20, 200, [0.2, 0.3, 0.5]),
            (3, [], 30, 300, []),
        ]

        for x, indices, y, z, weights in records:
            handle.write(struct.pack("<f", x))
            handle.write(struct.pack("<B", len(indices)))
            if indices:
                handle.write(struct.pack(f"<{len(indices)}I", *indices))
            handle.write(struct.pack("<2f", y, z))
            handle.write(struct.pack("<B", len(weights)))
            if weights:
                handle.write(struct.pack(f"<{len(weights)}f", *weights))

    result = read_ply(path)

    assert result["vertices"].tolist() == [
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
    ]


def test_rejects_truncated_variable_length_list_payload(tmp_path):
    path = tmp_path / "truncated_list.ply"

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property list uint8 uint32 view_indices\n"
        b"property float y\n"
        b"property float z\n"
        b"end_header\n"
    )

    path.write_bytes(
        header
        + struct.pack("<fB", 1.0, 3)
        + struct.pack("<I", 10)
    )

    with pytest.raises(PLYFormatError, match="Truncated PLY list data"):
        read_ply(path)


def test_rejects_negative_variable_length_list_count(tmp_path):
    path = tmp_path / "negative_list_count.ply"

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property list char uint32 view_indices\n"
        b"property float y\n"
        b"property float z\n"
        b"end_header\n"
    )

    path.write_bytes(
        header
        + struct.pack("<fb", 1.0, -1)
        + struct.pack("<2f", 2.0, 3.0)
    )

    with pytest.raises(PLYFormatError, match="Invalid or unreasonable"):
        read_ply(path)


def test_rejects_truncated_scalar_after_variable_length_list(tmp_path):
    path = tmp_path / "truncated_scalar_after_list.ply"

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 1\n"
        b"property float x\n"
        b"property list uint8 uint32 view_indices\n"
        b"property float y\n"
        b"property float z\n"
        b"end_header\n"
    )

    path.write_bytes(
        header
        + struct.pack("<fBI", 1.0, 1, 7)
        + struct.pack("<f", 2.0)
    )

    with pytest.raises(PLYFormatError, match="Unexpected end"):
        read_ply(path)



def test_mesh_quality_identifies_closed_tetrahedron():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)

    faces = [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ]

    report = calculate_mesh_quality(vertices, faces)

    assert report["face_count"] == 4
    assert report["boundary_edge_count"] == 0
    assert report["non_manifold_edge_count"] == 0
    assert report["connected_component_count"] == 1
    assert report["isolated_vertex_count"] == 0
    assert report["surface_area"] > 0


def test_mesh_quality_detects_open_boundary_edges():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=float)

    report = calculate_mesh_quality(vertices, [[0, 1, 2], [0, 2, 3]])

    assert report["boundary_edge_count"] == 4
    assert report["non_manifold_edge_count"] == 0
    assert report["connected_component_count"] == 1


def test_mesh_quality_detects_disconnected_components():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0],
        [9, 9, 9],
    ], dtype=float)

    report = calculate_mesh_quality(
        vertices,
        [[0, 1, 2], [3, 4, 5]],
    )

    assert report["connected_component_count"] == 3
    assert report["component_sizes"] == [3, 3, 1]
    assert report["isolated_vertex_count"] == 1


def test_mesh_quality_detects_non_manifold_edges():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, -1, 0], [0, 0, 1],
    ], dtype=float)

    faces = [
        [0, 1, 2],
        [1, 0, 3],
        [0, 1, 4],
    ]

    report = calculate_mesh_quality(vertices, faces)

    assert report["non_manifold_edge_count"] == 1


def test_mesh_quality_detects_degenerate_triangles():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ], dtype=float)

    report = calculate_mesh_quality(vertices, [[0, 1, 2]])

    assert report["degenerate_triangle_count"] == 1
    assert report["surface_area"] == 0


def test_mesh_quality_rejects_invalid_face_indices():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.zeros((3, 3))

    with pytest.raises(ValueError, match="out-of-range"):
        calculate_mesh_quality(vertices, [[0, 1, 5]])


def test_mesh_quality_calculates_polygon_surface_area():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=float)

    report = calculate_mesh_quality(vertices, [[0, 1, 2, 3]])

    assert report["triangle_count"] == 2
    assert report["surface_area"] == pytest.approx(1.0)
    assert report["boundary_edge_count"] == 4



def test_mesh_quality_reports_triangle_aspect_ratios():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
    ])

    report = calculate_mesh_quality(vertices, [[0, 1, 2]])

    assert report["triangle_aspect_ratio"]["minimum"] == pytest.approx(1.0)
    assert report["triangle_aspect_ratio"]["maximum"] == pytest.approx(1.0)


def test_mesh_quality_reports_edge_length_statistics():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])

    report = calculate_mesh_quality(vertices, [[0, 1, 2], [0, 2, 3]])

    assert report["edge_length"]["minimum"] == pytest.approx(1.0)
    assert report["edge_length"]["maximum"] == pytest.approx(np.sqrt(2))


def test_mesh_quality_reports_component_face_counts_and_areas():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0],
    ])

    report = calculate_mesh_quality(
        vertices,
        [[0, 1, 2], [3, 4, 5]],
    )

    assert len(report["component_details"]) == 2
    assert [item["face_count"] for item in report["component_details"]] == [1, 1]
    assert all(item["surface_area"] == pytest.approx(0.5)
               for item in report["component_details"])


def test_mesh_quality_identifies_closed_boundary_loop():
    from src.reconstruction_engine.quality import calculate_mesh_quality

    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])

    report = calculate_mesh_quality(vertices, [[0, 1, 2], [0, 2, 3]])

    assert report["boundary_edge_count"] == 4
    assert report["boundary_component_count"] == 1
    assert report["boundary_loop_count"] == 1
    assert report["boundary_chain_count"] == 0
    assert report["boundary_branched_count"] == 0
    assert report["boundary_component_edge_counts"] == [4]
