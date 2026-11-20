from __future__ import annotations

import math
import mmap
import struct
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "b", "int8": "b",
    "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h",
    "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i",
    "uint": "I", "uint32": "I",
    "float": "f", "float32": "f",
    "double": "d", "float64": "d",
}

PLY_TYPE_SIZES = {
    name: struct.calcsize("<" + fmt)
    for name, fmt in PLY_TYPES.items()
}


class PLYFormatError(ValueError):
    """Raised when a PLY file is malformed or unsupported."""


def _read_header(handle):
    first = handle.readline()
    if first.strip() != b"ply":
        raise PLYFormatError("File does not start with a PLY signature.")

    format_name = None
    elements = []
    current = None

    while True:
        line = handle.readline()
        if not line:
            raise PLYFormatError("PLY header is missing end_header.")

        try:
            parts = line.decode("ascii").strip().split()
        except UnicodeDecodeError as exc:
            raise PLYFormatError("PLY header must be ASCII.") from exc

        if not parts or parts[0] in ("comment", "obj_info"):
            continue

        if parts[0] == "format":
            if len(parts) != 3 or parts[2] != "1.0":
                raise PLYFormatError("Unsupported PLY format declaration.")
            format_name = parts[1]

        elif parts[0] == "element":
            if len(parts) != 3:
                raise PLYFormatError("Invalid element declaration.")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise PLYFormatError("Invalid element count.") from exc

            if count < 0 or count > 100_000_000:
                raise PLYFormatError("Unreasonable PLY element count.")

            current = {"name": parts[1], "count": count, "properties": []}
            elements.append(current)

        elif parts[0] == "property":
            if current is None:
                raise PLYFormatError("Property appears before an element.")

            if len(parts) == 3:
                _, type_name, name = parts
                if type_name not in PLY_TYPES:
                    raise PLYFormatError(
                        f"Unsupported PLY property type: {type_name}"
                    )
                current["properties"].append({
                    "kind": "scalar",
                    "type": type_name,
                    "name": name,
                })

            elif len(parts) == 5 and parts[1] == "list":
                _, _, count_type, item_type, name = parts
                if count_type not in PLY_TYPES or item_type not in PLY_TYPES:
                    raise PLYFormatError("Unsupported PLY list property type.")
                current["properties"].append({
                    "kind": "list",
                    "count_type": count_type,
                    "item_type": item_type,
                    "name": name,
                })

            else:
                raise PLYFormatError("Invalid PLY property declaration.")

        elif parts[0] == "end_header":
            break

    if format_name not in ("ascii", "binary_little_endian"):
        raise PLYFormatError(
            f"Unsupported PLY encoding: {format_name}"
        )

    if not elements:
        raise PLYFormatError("PLY file contains no elements.")

    return format_name, elements


def _read_binary_scalar(handle, type_name):
    fmt = "<" + PLY_TYPES[type_name]
    size = PLY_TYPE_SIZES[type_name]
    data = handle.read(size)

    if len(data) != size:
        raise PLYFormatError("Unexpected end of PLY binary data.")

    return struct.unpack(fmt, data)[0]


def _read_binary_list(handle, count_type, item_type):
    count = _read_binary_scalar(handle, count_type)

    if isinstance(count, bool) or count < 0 or count > 10_000_000:
        raise PLYFormatError("Invalid or unreasonable PLY list length.")

    size = count * PLY_TYPE_SIZES[item_type]
    data = handle.read(size)

    if len(data) != size:
        raise PLYFormatError("Truncated PLY list data.")

    return data, count


def _read_binary_scalar_element(handle, properties, count):
    """Vectorised reader for PLY elements containing scalar properties only."""
    if any(prop["kind"] != "scalar" for prop in properties):
        return None

    names = [prop["name"] for prop in properties]
    if len(names) != len(set(names)):
        raise PLYFormatError("Duplicate PLY property names are unsupported.")

    dtype = np.dtype([
        (prop["name"], "<" + PLY_TYPES[prop["type"]])
        for prop in properties
    ])

    records = np.fromfile(handle, dtype=dtype, count=count)

    if len(records) != count:
        raise PLYFormatError("Unexpected end of PLY binary element data.")

    return records


def _read_binary_vertex_element(handle, properties, count):
    """Read binary vertex XYZ into preallocated storage, skipping list payloads."""
    names = [prop["name"] for prop in properties]
    if len(names) != len(set(names)):
        raise PLYFormatError("Duplicate PLY vertex property names are unsupported.")

    scalar_names = {
        prop["name"] for prop in properties if prop["kind"] == "scalar"
    }
    if not {"x", "y", "z"}.issubset(scalar_names):
        raise PLYFormatError("Vertex element must contain scalar x, y and z.")

    groups = []
    current_scalars = []

    for prop in properties:
        if prop["kind"] == "scalar":
            current_scalars.append(prop)
        else:
            if current_scalars:
                fmt = "<" + "".join(
                    PLY_TYPES[item["type"]] for item in current_scalars
                )
                groups.append((
                    "scalars", current_scalars, struct.Struct(fmt)
                ))
                current_scalars = []
            groups.append(("list", prop, None))

    if current_scalars:
        fmt = "<" + "".join(
            PLY_TYPES[item["type"]] for item in current_scalars
        )
        groups.append(("scalars", current_scalars, struct.Struct(fmt)))

    xyz = np.empty((count, 3), dtype=np.float64)
    xyz_columns = {"x": 0, "y": 1, "z": 2}

    data_start = handle.tell()
    handle.seek(0, 2)
    file_size = handle.tell()
    handle.seek(data_start)

    for row in range(count):
        for kind, group, parser in groups:
            if kind == "scalars":
                raw = handle.read(parser.size)
                if len(raw) != parser.size:
                    raise PLYFormatError(
                        "Unexpected end of PLY binary vertex data."
                    )

                values = parser.unpack(raw)
                for prop, value in zip(group, values):
                    column = xyz_columns.get(prop["name"])
                    if column is not None:
                        xyz[row, column] = value
            else:
                list_count = _read_binary_scalar(
                    handle, group["count_type"]
                )
                if list_count < 0 or list_count > 10_000_000:
                    raise PLYFormatError(
                        "Invalid or unreasonable PLY list length."
                    )

                byte_count = (
                    list_count * PLY_TYPE_SIZES[group["item_type"]]
                )
                next_position = handle.tell() + byte_count

                if next_position > file_size:
                    raise PLYFormatError("Truncated PLY list data.")

                handle.seek(byte_count, 1)

    return xyz


def _read_binary_vertex_element_mmap(handle, properties, count):
    """Read variable-length binary vertex records through a memory map."""
    names = [prop["name"] for prop in properties]
    if len(names) != len(set(names)):
        raise PLYFormatError("Duplicate PLY vertex property names are unsupported.")

    scalar_names = {
        prop["name"] for prop in properties if prop["kind"] == "scalar"
    }
    if not {"x", "y", "z"}.issubset(scalar_names):
        raise PLYFormatError("Vertex element must contain scalar x, y and z.")

    groups = []
    scalar_group = []

    for prop in properties:
        if prop["kind"] == "scalar":
            scalar_group.append(prop)
        else:
            if scalar_group:
                parser = struct.Struct(
                    "<" + "".join(PLY_TYPES[item["type"]] for item in scalar_group)
                )
                groups.append(("scalar", scalar_group, parser))
                scalar_group = []
            groups.append(("list", prop, None))

    if scalar_group:
        parser = struct.Struct(
            "<" + "".join(PLY_TYPES[item["type"]] for item in scalar_group)
        )
        groups.append(("scalar", scalar_group, parser))

    xyz = np.empty((count, 3), dtype=np.float64)
    columns = {"x": 0, "y": 1, "z": 2}
    data_start = handle.tell()

    with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        offset = data_start
        file_size = len(mapped)

        for row in range(count):
            for kind, group, parser in groups:
                if kind == "scalar":
                    end = offset + parser.size
                    if end > file_size:
                        raise PLYFormatError(
                            "Unexpected end of PLY binary vertex data."
                        )

                    values = parser.unpack_from(mapped, offset)
                    offset = end

                    for prop, value in zip(group, values):
                        column = columns.get(prop["name"])
                        if column is not None:
                            xyz[row, column] = value
                else:
                    count_fmt = "<" + PLY_TYPES[group["count_type"]]
                    count_size = PLY_TYPE_SIZES[group["count_type"]]

                    if offset + count_size > file_size:
                        raise PLYFormatError("Truncated PLY list count.")

                    list_count = struct.unpack_from(
                        count_fmt, mapped, offset
                    )[0]
                    offset += count_size

                    if list_count < 0 or list_count > 10_000_000:
                        raise PLYFormatError(
                            "Invalid or unreasonable PLY list length."
                        )

                    payload_size = (
                        list_count * PLY_TYPE_SIZES[group["item_type"]]
                    )
                    if offset + payload_size > file_size:
                        raise PLYFormatError("Truncated PLY list data.")

                    offset += payload_size

    handle.seek(offset)
    return xyz


def read_ply(path):
    """
    Read PLY geometry.

    Returns:
        {
            "vertices": float64 NumPy array shaped (N, 3),
            "faces": list of integer index lists,
            "metadata": format and element counts
        }

    Non-geometric properties, including OpenMVS view metadata,
    are consumed but not retained.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    vertices = []
    fast_vertex_array = None
    faces = []
    vertex_count = 0
    face_count = 0

    with path.open("rb") as handle:
        encoding, elements = _read_header(handle)

        for element in elements:
            name = element["name"]
            properties = element["properties"]

            if name == "vertex":
                vertex_count = element["count"]
                property_names = {
                    prop["name"]
                    for prop in properties
                    if prop["kind"] == "scalar"
                }

                if not {"x", "y", "z"}.issubset(property_names):
                    raise PLYFormatError(
                        "Vertex element must contain scalar x, y and z."
                    )

            if name == "face":
                face_count = element["count"]

            if encoding == "binary_little_endian" and name == "vertex":
                records = _read_binary_scalar_element(
                    handle, properties, element["count"]
                )

                if records is not None:
                    fast_vertex_array = np.column_stack((
                        records["x"],
                        records["y"],
                        records["z"],
                    )).astype(np.float64, copy=False)
                else:
                    fast_vertex_array = _read_binary_vertex_element_mmap(
                        handle, properties, element["count"]
                    )

                continue

            for _ in range(element["count"]):
                values = {}

                if encoding == "ascii":
                    line = handle.readline()
                    if not line:
                        raise PLYFormatError("Truncated ASCII PLY data.")

                    try:
                        tokens = line.decode("ascii").split()
                    except UnicodeDecodeError as exc:
                        raise PLYFormatError(
                            "Invalid ASCII PLY data."
                        ) from exc

                    position = 0

                    for prop in properties:
                        if prop["kind"] == "scalar":
                            if position >= len(tokens):
                                raise PLYFormatError(
                                    "Missing scalar property value."
                                )
                            try:
                                values[prop["name"]] = float(
                                    tokens[position]
                                )
                            except ValueError as exc:
                                raise PLYFormatError(
                                    "Invalid scalar property value."
                                ) from exc
                            position += 1

                        else:
                            if position >= len(tokens):
                                raise PLYFormatError(
                                    "Missing list property count."
                                )
                            count = int(tokens[position])
                            position += 1

                            if count < 0 or count > 10_000_000:
                                raise PLYFormatError(
                                    "Invalid PLY list length."
                                )

                            end = position + count
                            if end > len(tokens):
                                raise PLYFormatError(
                                    "Truncated PLY list property."
                                )

                            if (
                                name == "face"
                                and prop["name"] == "vertex_indices"
                            ):
                                values[prop["name"]] = [
                                    int(v) for v in tokens[position:end]
                                ]

                            position = end

                    if position != len(tokens):
                        raise PLYFormatError(
                            "Unexpected extra values in PLY element."
                        )

                else:
                    for prop in properties:
                        if prop["kind"] == "scalar":
                            values[prop["name"]] = _read_binary_scalar(
                                handle, prop["type"]
                            )
                        else:
                            data, count = _read_binary_list(
                                handle,
                                prop["count_type"],
                                prop["item_type"],
                            )

                            if (
                                name == "face"
                                and prop["name"] == "vertex_indices"
                            ):
                                fmt = "<" + PLY_TYPES[prop["item_type"]]
                                size = PLY_TYPE_SIZES[prop["item_type"]]
                                values[prop["name"]] = [
                                    struct.unpack_from(
                                        fmt, data, i * size
                                    )[0]
                                    for i in range(count)
                                ]

                if name == "vertex":
                    vertices.append((
                        values["x"],
                        values["y"],
                        values["z"],
                    ))

                elif name == "face":
                    indices = values.get("vertex_indices")
                    if indices is not None:
                        faces.append(indices)

    if fast_vertex_array is not None:
        vertex_array = fast_vertex_array
    else:
        vertex_array = np.asarray(vertices, dtype=np.float64)

        if vertex_array.size == 0:
            vertex_array = np.empty((0, 3), dtype=np.float64)
        else:
            vertex_array = vertex_array.reshape((-1, 3))

    if len(vertex_array) != vertex_count:
        raise PLYFormatError("Parsed vertex count does not match header.")

    if len(faces) != face_count and face_count:
        raise PLYFormatError(
            "Not all PLY faces contained vertex_indices."
        )

    return {
        "vertices": vertex_array,
        "faces": faces,
        "metadata": {
            "encoding": encoding,
            "vertex_count": vertex_count,
            "face_count": face_count,
            "file_size_bytes": path.stat().st_size,
        },
    }


def calculate_mesh_quality(vertices, faces):
    """
    Analyse mesh topology and geometry.

    Faces may be triangles or polygons with a consistent vertex count.
    Polygon surface area is calculated using fan triangulation.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    points = np.asarray(vertices, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3).")

    if not np.isfinite(points).all():
        raise ValueError("Mesh vertices must contain only finite coordinates.")

    if not faces:
        return {
            "vertex_count": int(len(points)),
            "face_count": 0,
            "triangle_count": 0,
            "surface_area": 0.0,
            "degenerate_triangle_count": 0,
            "boundary_edge_count": 0,
            "non_manifold_edge_count": 0,
            "connected_component_count": int(len(points)),
            "component_sizes": [1] * len(points),
            "isolated_vertex_count": int(len(points)),
            "bounding_box": calculate_point_cloud_quality(points)["bounding_box"],
        }

    try:
        face_array = np.asarray(faces, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Faces must form a rectangular array of indices.") from exc

    if face_array.ndim != 2 or face_array.shape[1] < 3:
        raise ValueError("Faces must contain at least three vertex indices.")

    if np.any(face_array < 0) or np.any(face_array >= len(points)):
        raise ValueError("Face contains an out-of-range vertex index.")

    face_count, vertices_per_face = face_array.shape

    # Build each polygon's boundary edges.
    edge_parts = []
    for column in range(vertices_per_face):
        next_column = (column + 1) % vertices_per_face
        edge_parts.append(face_array[:, [column, next_column]])

    edges = np.concatenate(edge_parts, axis=0)
    edges.sort(axis=1)

    unique_edges, edge_incidence = np.unique(
        edges, axis=0, return_counts=True
    )

    boundary_edge_count = int(np.count_nonzero(edge_incidence == 1))
    non_manifold_edge_count = int(np.count_nonzero(edge_incidence > 2))

    # Fan-triangulate polygons for surface-area and degeneracy measurements.
    triangles = np.concatenate([
        face_array[:, [0, column, column + 1]]
        for column in range(1, vertices_per_face - 1)
    ], axis=0)

    triangle_points = points[triangles]
    cross_products = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)

    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    area_tolerance = max(
        scale * scale * 1e-12,
        np.finfo(np.float64).eps,
    )

    repeated_indices = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    degenerate_mask = repeated_indices | (triangle_areas <= area_tolerance)

    # Build an undirected sparse graph for vertex-connected components.
    graph_rows = np.concatenate((unique_edges[:, 0], unique_edges[:, 1]))
    graph_cols = np.concatenate((unique_edges[:, 1], unique_edges[:, 0]))

    adjacency = coo_matrix(
        (
            np.ones(len(graph_rows), dtype=np.uint8),
            (graph_rows, graph_cols),
        ),
        shape=(len(points), len(points)),
    ).tocsr()

    component_count, component_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )

    component_sizes = np.bincount(
        component_labels, minlength=component_count
    )
    component_sizes = sorted(
        (int(size) for size in component_sizes),
        reverse=True,
    )

    used_vertices = np.unique(face_array)

    # Triangle shape: 1.0 is equilateral; larger values are less regular.
    triangle_edge_lengths = np.stack((
        np.linalg.norm(
            triangle_points[:, 1] - triangle_points[:, 0], axis=1
        ),
        np.linalg.norm(
            triangle_points[:, 2] - triangle_points[:, 1], axis=1
        ),
        np.linalg.norm(
            triangle_points[:, 0] - triangle_points[:, 2], axis=1
        ),
    ), axis=1)

    longest_triangle_edge = triangle_edge_lengths.max(axis=1)
    triangle_perimeters = triangle_edge_lengths.sum(axis=1)
    triangle_inradii = np.divide(
        2.0 * triangle_areas,
        triangle_perimeters,
        out=np.zeros_like(triangle_areas),
        where=triangle_perimeters > 0,
    )

    triangle_aspect_ratio = np.divide(
        longest_triangle_edge,
        2.0 * np.sqrt(3.0) * triangle_inradii,
        out=np.full_like(triangle_areas, np.inf),
        where=triangle_inradii > 0,
    )

    valid_aspect_ratios = triangle_aspect_ratio[
        np.isfinite(triangle_aspect_ratio)
    ]

    if len(valid_aspect_ratios):
        triangle_aspect_report = {
            "minimum": float(valid_aspect_ratios.min()),
            "median": float(np.median(valid_aspect_ratios)),
            "p95": float(np.percentile(valid_aspect_ratios, 95)),
            "maximum": float(valid_aspect_ratios.max()),
            "above_2_count": int(np.count_nonzero(valid_aspect_ratios > 2)),
            "above_5_count": int(np.count_nonzero(valid_aspect_ratios > 5)),
            "above_10_count": int(np.count_nonzero(valid_aspect_ratios > 10)),
        }
    else:
        triangle_aspect_report = None

    # Edge-length distribution over unique mesh edges.
    unique_edge_lengths = np.linalg.norm(
        points[unique_edges[:, 0]] - points[unique_edges[:, 1]],
        axis=1,
    )

    if len(unique_edge_lengths):
        edge_length_report = {
            "minimum": float(unique_edge_lengths.min()),
            "median": float(np.median(unique_edge_lengths)),
            "p95": float(np.percentile(unique_edge_lengths, 95)),
            "maximum": float(unique_edge_lengths.max()),
            "mean": float(unique_edge_lengths.mean()),
        }
    else:
        edge_length_report = None

    # Per-component face counts and surface areas.
    face_component_labels = component_labels[face_array[:, 0]]
    component_face_counts = np.bincount(
        face_component_labels, minlength=component_count
    )

    triangle_face_ids = np.tile(
        np.arange(face_count), vertices_per_face - 2
    )
    triangle_component_labels = face_component_labels[triangle_face_ids]

    component_surface_areas = np.bincount(
        triangle_component_labels,
        weights=triangle_areas,
        minlength=component_count,
    )

    component_report = [
        {
            "component": int(index),
            "vertex_count": int(component_sizes[index])
            if False else int(np.count_nonzero(component_labels == index)),
            "face_count": int(component_face_counts[index]),
            "surface_area": float(component_surface_areas[index]),
        }
        for index in range(component_count)
    ]
    component_report.sort(
        key=lambda item: item["vertex_count"], reverse=True
    )

    # Add spatial bounds and centroids for each connected component.
    for item in component_report:
        component_vertices = points[
            component_labels == item["component"]
        ]
        item["centroid"] = component_vertices.mean(axis=0).tolist()
        item["bounding_box"] = {
            "minimum": component_vertices.min(axis=0).tolist(),
            "maximum": component_vertices.max(axis=0).tolist(),
        }

    # Rank the most distorted triangles by aspect ratio.
    triangle_centroids = triangle_points.mean(axis=1)
    worst_triangle_indices = np.argsort(
        triangle_aspect_ratio
    )[::-1][:25]

    triangle_anomalies = [
        {
            "triangle_index": int(index),
            "source_face_index": int(triangle_face_ids[index]),
            "aspect_ratio": float(triangle_aspect_ratio[index]),
            "longest_edge": float(longest_triangle_edge[index]),
            "centroid": triangle_centroids[index].tolist(),
        }
        for index in worst_triangle_indices
        if np.isfinite(triangle_aspect_ratio[index])
    ]

    # Rank the longest unique mesh edges.
    longest_edge_indices = np.argsort(
        unique_edge_lengths
    )[::-1][:25]

    long_edge_anomalies = [
        {
            "length": float(unique_edge_lengths[index]),
            "vertex_indices": unique_edges[index].tolist(),
            "midpoint": (
                (
                    points[unique_edges[index, 0]]
                    + points[unique_edges[index, 1]]
                ) / 2.0
            ).tolist(),
            "endpoint_a": points[unique_edges[index, 0]].tolist(),
            "endpoint_b": points[unique_edges[index, 1]].tolist(),
        }
        for index in longest_edge_indices
    ]

    # Analyse connected boundary regions and identify closed loops.
    boundary_edges = unique_edges[edge_incidence == 1]

    boundary_loop_count = 0
    boundary_chain_count = 0
    boundary_branched_count = 0
    boundary_component_edge_counts = []

    if len(boundary_edges):
        boundary_vertices, boundary_inverse = np.unique(
            boundary_edges, return_inverse=True
        )
        boundary_inverse = boundary_inverse.reshape((-1, 2))

        boundary_degree = np.bincount(
            boundary_inverse.ravel(), minlength=len(boundary_vertices)
        )

        boundary_rows = np.concatenate((
            boundary_inverse[:, 0], boundary_inverse[:, 1]
        ))
        boundary_cols = np.concatenate((
            boundary_inverse[:, 1], boundary_inverse[:, 0]
        ))

        boundary_adjacency = coo_matrix(
            (
                np.ones(len(boundary_rows), dtype=np.uint8),
                (boundary_rows, boundary_cols),
            ),
            shape=(len(boundary_vertices), len(boundary_vertices)),
        ).tocsr()

        boundary_component_count, boundary_labels = connected_components(
            boundary_adjacency, directed=False, return_labels=True
        )

        boundary_edge_labels = boundary_labels[boundary_inverse[:, 0]]
        boundary_component_edge_counts = np.bincount(
            boundary_edge_labels, minlength=boundary_component_count
        )

        for boundary_component in range(boundary_component_count):
            member_degrees = boundary_degree[
                boundary_labels == boundary_component
            ]

            if np.all(member_degrees == 2):
                boundary_loop_count += 1
            elif np.any(member_degrees > 2):
                boundary_branched_count += 1
            else:
                boundary_chain_count += 1

        boundary_component_edge_counts = sorted(
            (int(count) for count in boundary_component_edge_counts),
            reverse=True,
        )

    return {
        "vertex_count": int(len(points)),
        "face_count": int(face_count),
        "triangle_count": int(len(triangles)),
        "surface_area": float(triangle_areas.sum()),
        "degenerate_triangle_count": int(degenerate_mask.sum()),
        "boundary_edge_count": boundary_edge_count,
        "non_manifold_edge_count": non_manifold_edge_count,
        "connected_component_count": int(component_count),
        "component_sizes": component_sizes,
        "isolated_vertex_count": int(len(points) - len(used_vertices)),
        "triangle_aspect_ratio": triangle_aspect_report,
        "edge_length": edge_length_report,
        "component_details": component_report,
        "triangle_anomalies": triangle_anomalies,
        "long_edge_anomalies": long_edge_anomalies,
        "boundary_component_count": int(
            boundary_loop_count + boundary_chain_count + boundary_branched_count
        ),
        "boundary_loop_count": int(boundary_loop_count),
        "boundary_chain_count": int(boundary_chain_count),
        "boundary_branched_count": int(boundary_branched_count),
        "boundary_component_edge_counts": boundary_component_edge_counts,
        "bounding_box": calculate_point_cloud_quality(points)["bounding_box"],
    }


def calculate_point_cloud_quality(vertices):
    points = np.asarray(vertices, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3).")

    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]

    if len(finite_points):
        minimum = finite_points.min(axis=0)
        maximum = finite_points.max(axis=0)
        spread = maximum - minimum
    else:
        minimum = maximum = spread = None

    return {
        "point_count": int(len(points)),
        "finite_point_count": int(finite_mask.sum()),
        "non_finite_point_count": int((~finite_mask).sum()),
        "bounding_box": (
            None if minimum is None else {
                "minimum": minimum.tolist(),
                "maximum": maximum.tolist(),
                "dimensions": spread.tolist(),
            }
        ),
    }
