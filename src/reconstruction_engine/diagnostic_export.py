from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _validate_mesh(vertices, faces):
    points = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3).")

    if not np.isfinite(points).all():
        raise ValueError("Vertices must contain finite coordinates.")

    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("Diagnostic export currently requires triangular faces.")

    if not np.issubdtype(face_array.dtype, np.integer):
        raise ValueError("Face indices must be integers.")

    face_array = face_array.astype(np.int64, copy=False)

    if np.any(face_array < 0) or np.any(face_array >= len(points)):
        raise ValueError("Face contains an out-of-range vertex index.")

    return points, face_array


def _component_labels(vertex_count, faces):
    if vertex_count == 0:
        return np.empty(0, dtype=np.int32), 0

    if len(faces) == 0:
        return np.arange(vertex_count, dtype=np.int32), vertex_count

    edges = np.concatenate((
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ))
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)

    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))

    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(vertex_count, vertex_count),
    ).tocsr()

    component_count, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    return labels, component_count


def _write_coloured_ply(path, vertices, faces, colours):
    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertex_data[name] = vertices[:, axis]

    vertex_data["red"] = colours[:, 0]
    vertex_data["green"] = colours[:, 1]
    vertex_data["blue"] = colours[:, 2]

    face_dtype = np.dtype([
        ("count", "u1"),
        ("indices", "<u4", (3,)),
    ])

    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data["count"] = 3
    face_data["indices"] = faces

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    )

    with Path(path).open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertex_data.tofile(handle)
        face_data.tofile(handle)


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_mesh_diagnostics(vertices, faces, report, output_dir):
    """
    Export coloured connected components, anomalous triangles,
    long-edge details and component spatial summaries.

    The source mesh is never modified.
    """
    points, face_array = _validate_mesh(vertices, faces)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, component_count = _component_labels(
        len(points), face_array
    )

    counts = np.bincount(labels, minlength=component_count)
    ordered_labels = np.argsort(counts)[::-1]

    # Main component is grey; detached components receive distinct colours.
    palette = [
        (230, 150, 45),
        (60, 170, 220),
        (190, 90, 190),
        (80, 190, 120),
        (220, 90, 90),
        (210, 190, 70),
        (100, 130, 220),
    ]

    label_to_rank = np.empty(component_count, dtype=np.int32)
    label_to_rank[ordered_labels] = np.arange(component_count)

    colours = np.empty((len(points), 3), dtype=np.uint8)
    colours[:] = (150, 150, 150)

    for rank, label in enumerate(ordered_labels[1:], start=0):
        colours[labels == label] = palette[rank % len(palette)]

    component_mesh_path = output_dir / "components_coloured.ply"
    _write_coloured_ply(
        component_mesh_path, points, face_array, colours
    )

    # Export the anomalous triangles as a compact, separate mesh.
    anomalies = report.get("triangle_anomalies", [])
    anomaly_face_ids = sorted({
        int(item["source_face_index"])
        for item in anomalies
        if 0 <= int(item["source_face_index"]) < len(face_array)
    })

    anomaly_mesh_path = output_dir / "triangle_anomalies.ply"

    if anomaly_face_ids:
        selected_faces = face_array[anomaly_face_ids]
        selected_vertices = np.unique(selected_faces)
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[selected_vertices] = np.arange(len(selected_vertices))

        compact_faces = remap[selected_faces]
        anomaly_colours = np.tile(
            np.array([[255, 40, 40]], dtype=np.uint8),
            (len(selected_vertices), 1),
        )

        _write_coloured_ply(
            anomaly_mesh_path,
            points[selected_vertices],
            compact_faces,
            anomaly_colours,
        )
    else:
        _write_coloured_ply(
            anomaly_mesh_path,
            np.empty((0, 3)),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.uint8),
        )

    long_edge_rows = []
    for item in report.get("long_edge_anomalies", []):
        row = {"length": item["length"]}
        for axis, name in enumerate(("x", "y", "z")):
            row[f"midpoint_{name}"] = item["midpoint"][axis]
            row[f"endpoint_a_{name}"] = item["endpoint_a"][axis]
            row[f"endpoint_b_{name}"] = item["endpoint_b"][axis]
        row["vertex_a"] = item["vertex_indices"][0]
        row["vertex_b"] = item["vertex_indices"][1]
        long_edge_rows.append(row)

    long_edges_path = output_dir / "long_edges.csv"
    edge_fields = [
        "length", "vertex_a", "vertex_b",
        "midpoint_x", "midpoint_y", "midpoint_z",
        "endpoint_a_x", "endpoint_a_y", "endpoint_a_z",
        "endpoint_b_x", "endpoint_b_y", "endpoint_b_z",
    ]
    _write_csv(long_edges_path, long_edge_rows, edge_fields)

    component_rows = []
    for item in report.get("component_details", []):
        row = {
            "component": item["component"],
            "vertex_count": item["vertex_count"],
            "face_count": item["face_count"],
            "surface_area": item["surface_area"],
        }

        for axis, name in enumerate(("x", "y", "z")):
            row[f"centroid_{name}"] = item["centroid"][axis]
            row[f"minimum_{name}"] = item["bounding_box"]["minimum"][axis]
            row[f"maximum_{name}"] = item["bounding_box"]["maximum"][axis]

        component_rows.append(row)

    components_path = output_dir / "components.csv"
    component_fields = [
        "component", "vertex_count", "face_count", "surface_area",
        "centroid_x", "centroid_y", "centroid_z",
        "minimum_x", "minimum_y", "minimum_z",
        "maximum_x", "maximum_y", "maximum_z",
    ]
    _write_csv(components_path, component_rows, component_fields)

    return {
        "components_mesh": component_mesh_path,
        "anomaly_mesh": anomaly_mesh_path,
        "long_edges_csv": long_edges_path,
        "components_csv": components_path,
        "component_count": int(component_count),
        "anomaly_face_count": len(anomaly_face_ids),
        "long_edge_count": len(long_edge_rows),
    }
