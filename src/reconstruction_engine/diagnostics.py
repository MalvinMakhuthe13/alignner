import math
import statistics
import struct
from pathlib import Path
COLMAP_MAX_IMAGE_ID = 2_147_483_647


def decode_colmap_pair_id(pair_id):
    if isinstance(pair_id, bool) or not isinstance(pair_id, int):
        raise ValueError("COLMAP pair ID must be an integer.")

    if pair_id <= 0:
        raise ValueError("COLMAP pair ID must be positive.")

    image_id_1 = pair_id // COLMAP_MAX_IMAGE_ID
    image_id_2 = pair_id % COLMAP_MAX_IMAGE_ID

    if (
        image_id_1 <= 0
        or image_id_2 <= 0
        or image_id_1 >= image_id_2
        or image_id_2 >= COLMAP_MAX_IMAGE_ID
    ):
        raise ValueError(f"Invalid COLMAP pair ID: {pair_id}")

    return image_id_1, image_id_2



def _read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(
            f"Unexpected end of COLMAP binary file: "
            f"needed {size} bytes, received {len(data)}."
        )
    return data


def _read_struct(handle, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(handle, size))


def read_images_binary(path):
    images = []

    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")

        if count > 10_000_000:
            raise ValueError("Unreasonable image count in COLMAP binary.")

        for _ in range(count):
            (image_id,) = _read_struct(handle, "<i")
            qvec = _read_struct(handle, "<4d")
            tvec = _read_struct(handle, "<3d")
            (camera_id,) = _read_struct(handle, "<i")

            name_bytes = bytearray()
            while True:
                char = _read_exact(handle, 1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)

                if len(name_bytes) > 1_000_000:
                    raise ValueError("Unreasonable image name length.")

            name = name_bytes.decode("utf-8")

            (num_points2d,) = _read_struct(handle, "<Q")
            if num_points2d > 100_000_000:
                raise ValueError("Unreasonable 2D point count.")

            observations = []
            for _ in range(num_points2d):
                x, y, point3d_id = _read_struct(handle, "<ddq")
                observations.append({
                    "x": x,
                    "y": y,
                    "point3d_id": point3d_id,
                })

            images.append({
                "image_id": image_id,
                "name": name,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "observations": observations,
                "registered": True,
            })

    return images


def read_points3d_binary(path):
    points = []

    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")

        if count > 100_000_000:
            raise ValueError("Unreasonable 3D point count.")

        for _ in range(count):
            (point_id,) = _read_struct(handle, "<Q")
            xyz = _read_struct(handle, "<3d")
            rgb = _read_struct(handle, "<3B")
            (error,) = _read_struct(handle, "<d")
            (track_length,) = _read_struct(handle, "<Q")

            if track_length > 100_000_000:
                raise ValueError("Unreasonable point track length.")

            track = []
            for _ in range(track_length):
                image_id, point2d_idx = _read_struct(handle, "<ii")
                track.append({
                    "image_id": image_id,
                    "point2d_idx": point2d_idx,
                })

            points.append({
                "point3d_id": point_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track_length": track_length,
                "track": track,
            })

    return points


def calculate_reprojection_statistics(errors):
    values = [
        float(value)
        for value in errors
        if value is not None and math.isfinite(float(value))
    ]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "maximum": None,
        }

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def summarize_sparse_reconstruction(
    images,
    points,
    input_image_count,
):
    if (
        isinstance(input_image_count, bool)
        or not isinstance(input_image_count, int)
        or input_image_count < 0
    ):
        raise ValueError("input_image_count must be a non-negative integer.")

    registered_count = sum(
        1 for image in images if image.get("registered", True)
    )

    if registered_count > input_image_count:
        raise ValueError(
            "Registered image count cannot exceed input image count."
        )

    registration_rate = (
        registered_count / input_image_count * 100.0
        if input_image_count
        else 0.0
    )

    track_lengths = [
        int(point["track_length"])
        for point in points
        if point.get("track_length") is not None
    ]

    track_lengths = [
        length for length in track_lengths if length >= 0
    ]

    reprojection = calculate_reprojection_statistics(
        point.get("error") for point in points
    )

    return {
        "images": {
            "input": input_image_count,
            "registered": registered_count,
            "unregistered": input_image_count - registered_count,
            "registration_rate_percent": registration_rate,
        },
        "sparse_points": {
            "count": len(points),
        },
        "tracks": {
            "count": len(track_lengths),
            "mean_length": (
                statistics.fmean(track_lengths)
                if track_lengths else None
            ),
            "median_length": (
                statistics.median(track_lengths)
                if track_lengths else None
            ),
        },
        "reprojection_error": reprojection,
    }


def _quaternion_to_rotation_matrix(qvec):
    if len(qvec) != 4:
        raise ValueError("COLMAP quaternion must contain four values.")

    qw, qx, qy, qz = (float(value) for value in qvec)

    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)

    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid COLMAP quaternion.")

    qw, qx, qy, qz = (
        value / norm for value in (qw, qx, qy, qz)
    )

    return (
        (
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ),
        (
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
        ),
        (
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ),
    )


def calculate_camera_coverage(images):
    centres = []

    for image in images:
        rotation = _quaternion_to_rotation_matrix(image["qvec"])
        translation = tuple(float(v) for v in image["tvec"])

        if len(translation) != 3:
            raise ValueError("COLMAP translation must contain three values.")

        if not all(math.isfinite(v) for v in translation):
            raise ValueError("Invalid COLMAP translation.")

        centre = tuple(
            -sum(rotation[row][column] * translation[row]
                 for row in range(3))
            for column in range(3)
        )

        centres.append(centre)

    if not centres:
        return {
            "camera_count": 0,
            "centres": [],
            "baseline_span": None,
        }

    maximum_distance = 0.0

    for index, first in enumerate(centres):
        for second in centres[index + 1:]:
            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(first, second))
            )
            maximum_distance = max(maximum_distance, distance)

    return {
        "camera_count": len(centres),
        "centres": centres,
        "baseline_span": maximum_distance,
    }


def calculate_track_quality(points):
    lengths = [
        int(point["track_length"])
        for point in points
        if point.get("track_length") is not None
    ]

    if any(length < 0 for length in lengths):
        raise ValueError("Track lengths cannot be negative.")

    count = len(lengths)

    if count == 0:
        return {
            "point_count": 0,
            "mean_track_length": None,
            "points_seen_by_3_or_more_cameras": 0,
            "percentage_seen_by_3_or_more_cameras": None,
        }

    supported = sum(length >= 3 for length in lengths)

    return {
        "point_count": count,
        "mean_track_length": statistics.fmean(lengths),
        "points_seen_by_3_or_more_cameras": supported,
        "percentage_seen_by_3_or_more_cameras": (
            supported / count * 100.0
        ),
    }


def calculate_reprojection_quality(errors):
    stats = calculate_reprojection_statistics(errors)

    values = [
        float(value)
        for value in errors
        if value is not None and math.isfinite(float(value))
    ]

    count = len(values)
    above_2 = sum(value > 2.0 for value in values)
    above_4 = sum(value > 4.0 for value in values)

    return {
        **stats,
        "above_2px": above_2,
        "above_4px": above_4,
        "percentage_above_2px": (
            above_2 / count * 100.0 if count else None
        ),
        "percentage_above_4px": (
            above_4 / count * 100.0 if count else None
        ),
    }



def calculate_camera_pose_quality(images):
    """Summarize camera position and orientation diversity from COLMAP poses."""
    centres = []
    directions = []
    rotations = []

    for image in images:
        rotation = _quaternion_to_rotation_matrix(image["qvec"])
        translation = tuple(float(v) for v in image["tvec"])

        if len(translation) != 3:
            raise ValueError("COLMAP translation must contain three values.")

        if not all(math.isfinite(v) for v in translation):
            raise ValueError("Camera translation contains non-finite values.")

        centre = tuple(
            -sum(rotation[row][column] * translation[row]
                 for row in range(3))
            for column in range(3)
        )

        # COLMAP cameras look along their positive camera-space Z axis.
        direction = tuple(rotation[2][column] for column in range(3))

        centres.append(centre)
        directions.append(direction)
        rotations.append(rotation)

    def angle_statistics(values):
        if not values:
            return {
                "count": 0,
                "mean_degrees": None,
                "median_degrees": None,
                "maximum_degrees": None,
            }

        return {
            "count": len(values),
            "mean_degrees": statistics.fmean(values),
            "median_degrees": statistics.median(values),
            "maximum_degrees": max(values),
        }

    if not centres:
        return {
            "camera_count": 0,
            "position_spread": None,
            "pairwise_position_distance": angle_statistics([]),
            "viewing_direction_diversity": angle_statistics([]),
            "relative_rotation_diversity": angle_statistics([]),
        }

    axis_names = ("x", "y", "z")
    position_spread = {}

    for axis, name in enumerate(axis_names):
        values = [centre[axis] for centre in centres]
        position_spread[name] = max(values) - min(values)

    distances = []
    viewing_angles = []
    relative_rotations = []

    for first in range(len(centres)):
        for second in range(first + 1, len(centres)):
            distance = math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(centres[first], centres[second])
            ))
            distances.append(distance)

            dot = sum(
                a * b for a, b in zip(
                    directions[first], directions[second]
                )
            )
            dot = max(-1.0, min(1.0, dot))
            viewing_angles.append(math.degrees(math.acos(dot)))

            relative_trace = sum(
                rotations[first][row][column]
                * rotations[second][row][column]
                for row in range(3)
                for column in range(3)
            )
            cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
            relative_rotations.append(
                math.degrees(math.acos(cosine))
            )

    return {
        "camera_count": len(centres),
        "position_spread": position_spread,
        "pairwise_position_distance": {
            "count": len(distances),
            "minimum": min(distances) if distances else None,
            "mean": statistics.fmean(distances) if distances else None,
            "median": statistics.median(distances) if distances else None,
            "maximum": max(distances) if distances else None,
        },
        "viewing_direction_diversity": angle_statistics(viewing_angles),
        "relative_rotation_diversity": angle_statistics(relative_rotations),
    }


def calculate_camera_point_support(images, points):
    """Measure triangulated 3D-point support observed by each registered image."""
    point_track_lengths = {}

    for point in points:
        point_id = point.get("point3d_id")
        if point_id is None:
            continue

        track_length = point.get("track_length")
        if track_length is None:
            track_length = len(point.get("track", []))

        if (
            isinstance(track_length, bool)
            or not isinstance(track_length, int)
            or track_length < 0
        ):
            raise ValueError("Point track lengths must be non-negative integers.")

        point_track_lengths[point_id] = track_length

    camera_reports = []
    total_observations = 0
    total_triangulated_observations = 0
    total_untriangulated_observations = 0
    total_unresolved_references = 0

    for image in images:
        image_id = image.get("image_id")
        observations = image.get("observations", [])

        if not isinstance(observations, (list, tuple)):
            raise ValueError("Image observations must be a list or tuple.")

        unique_points = set()
        track_counts = {"2": 0, "3": 0, "4_plus": 0, "under_2": 0}
        untriangulated = 0
        unresolved = 0
        triangulated_observations = 0

        for observation in observations:
            point_id = observation.get("point3d_id")

            if point_id is None or point_id == -1:
                untriangulated += 1
                continue

            if isinstance(point_id, bool) or not isinstance(point_id, int):
                raise ValueError("Observation point3d_id must be an integer.")

            track_length = point_track_lengths.get(point_id)
            if track_length is None:
                unresolved += 1
                continue

            triangulated_observations += 1
            unique_points.add(point_id)

        for point_id in unique_points:
            track_length = point_track_lengths[point_id]
            if track_length == 2:
                track_counts["2"] += 1
            elif track_length == 3:
                track_counts["3"] += 1
            elif track_length >= 4:
                track_counts["4_plus"] += 1
            else:
                track_counts["under_2"] += 1

        observation_count = len(observations)
        total_observations += observation_count
        total_triangulated_observations += triangulated_observations
        total_untriangulated_observations += untriangulated
        total_unresolved_references += unresolved

        camera_reports.append({
            "image_id": image_id,
            "name": image.get("name"),
            "observation_count": observation_count,
            "triangulated_observation_count": triangulated_observations,
            "unique_3d_points": len(unique_points),
            "untriangulated_observations": untriangulated,
            "unresolved_point_references": unresolved,
            "unique_points_by_track_length": track_counts,
        })

    return {
        "camera_count": len(camera_reports),
        "total_observations": total_observations,
        "total_triangulated_observations": total_triangulated_observations,
        "total_untriangulated_observations": total_untriangulated_observations,
        "total_unresolved_point_references": total_unresolved_references,
        "cameras": camera_reports,
    }


def build_sparse_diagnostics_report(
    images,
    points,
    input_image_count,
):
    reprojection_errors = [
        point.get("error")
        for point in points
    ]

    return {
        "images": summarize_sparse_reconstruction(
            images=images,
            points=points,
            input_image_count=input_image_count,
        )["images"],

        "sparse_points": {
            "count": len(points),
        },

        "tracks": calculate_track_quality(points),

        "reprojection_error": calculate_reprojection_statistics(
            reprojection_errors
        ),

        "reprojection_quality": calculate_reprojection_quality(
            reprojection_errors
        ),

        "camera_coverage": calculate_camera_coverage(images),
        "camera_point_support": calculate_camera_point_support(images, points),
        "camera_pose_quality": calculate_camera_pose_quality(images),

        "limitations": [
            "Camera baseline measures spatial spread, not complete object coverage.",
            "Reprojection error measures image-space consistency, not dimensional accuracy.",
            "Sparse point count alone does not establish surface completeness.",
            "Track length does not independently establish point accuracy.",
            "Absolute scale requires independent scale information or calibration.",
        ],
    }

def analyze_camera_pair_graph(
    image_ids,
    verified_pairs,
    weak_connection_threshold=2,
):
    """Analyse connectivity of geometrically verified COLMAP image pairs."""

    if (
        isinstance(weak_connection_threshold, bool)
        or not isinstance(weak_connection_threshold, int)
        or weak_connection_threshold < 1
    ):
        raise ValueError("Weak connection threshold must be a positive integer.")

    image_ids = list(image_ids)

    if any(
        isinstance(image_id, bool)
        or not isinstance(image_id, int)
        or image_id <= 0
        or image_id >= COLMAP_MAX_IMAGE_ID
        for image_id in image_ids
    ):
        raise ValueError("Image IDs must be valid positive COLMAP image IDs.")

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image IDs are not allowed.")

    image_set = set(image_ids)
    adjacency = {image_id: set() for image_id in image_ids}

    invalid_pair_ids = []
    dangling_pairs = []
    verified_match_rows = 0

    for pair_id, rows in verified_pairs:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError("Verified match row counts must be nonnegative integers.")

        if rows == 0:
            continue

        try:
            image_id_1, image_id_2 = decode_colmap_pair_id(pair_id)
        except ValueError:
            invalid_pair_ids.append(pair_id)
            continue

        if image_id_1 not in image_set or image_id_2 not in image_set:
            dangling_pairs.append({
                "pair_id": pair_id,
                "image_ids": [image_id_1, image_id_2],
            })
            continue

        adjacency[image_id_1].add(image_id_2)
        adjacency[image_id_2].add(image_id_1)
        verified_match_rows += rows

    degree_by_image = {
        image_id: len(neighbours)
        for image_id, neighbours in adjacency.items()
    }

    isolated_image_ids = sorted(
        image_id
        for image_id, degree in degree_by_image.items()
        if degree == 0
    )

    weakly_connected_image_ids = sorted(
        image_id
        for image_id, degree in degree_by_image.items()
        if 0 < degree < weak_connection_threshold
    )

    components = []
    visited = set()

    for start_id in sorted(image_ids):
        if start_id in visited:
            continue

        stack = [start_id]
        visited.add(start_id)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)

        components.append(sorted(component))

    components.sort(key=lambda component: (-len(component), component))

    image_count = len(image_ids)
    possible_pairs = image_count * (image_count - 1) // 2
    verified_pair_count = sum(
        len(neighbours) for neighbours in adjacency.values()
    ) // 2

    connected_image_count = image_count - len(isolated_image_ids)
    largest_component_size = (
        len(components[0]) if components else 0
    )

    return {
        "image_count": image_count,
        "verified_pair_count": verified_pair_count,
        "verified_match_rows": verified_match_rows,
        "possible_pair_count": possible_pairs,
        "pair_coverage_percent": (
            verified_pair_count / possible_pairs * 100.0
            if possible_pairs else 0.0
        ),
        "component_count": len(components),
        "components": components,
        "largest_component_size": largest_component_size,
        "largest_component_percentage": (
            largest_component_size / image_count * 100.0
            if image_count else 0.0
        ),
        "connected_image_count": connected_image_count,
        "isolated_image_ids": isolated_image_ids,
        "weak_connection_threshold": weak_connection_threshold,
        "weakly_connected_image_ids": weakly_connected_image_ids,
        "degree_by_image": degree_by_image,
        "invalid_pair_ids": invalid_pair_ids,
        "invalid_pair_count": len(invalid_pair_ids),
        "dangling_pairs": dangling_pairs,
        "dangling_pair_count": len(dangling_pairs),
    }

