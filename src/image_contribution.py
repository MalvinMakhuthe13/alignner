"""Explainable per-image contribution scoring for Alignner.

This module is deliberately independent of the alignment pipeline.
It consumes an AlignmentResult-like object using duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ContributionConfig:
    """Explicit scoring weights. Weights are normalized before use."""

    pair_support_weight: float = 0.20
    track_support_weight: float = 0.25
    image_coverage_weight: float = 0.15
    reprojection_weight: float = 0.15
    parallax_weight: float = 0.10
    distinctive_coverage_weight: float = 0.15  # Deprecated; excluded from scoring.

    grid_size: int = 8
    min_parallax_deg: float = 0.5
    good_reprojection_px: float = 3.0

    def __post_init__(self):
        weights = [
            self.pair_support_weight,
            self.track_support_weight,
            self.image_coverage_weight,
            self.reprojection_weight,
            self.parallax_weight,
            self.distinctive_coverage_weight,
        ]
        if any(not np.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Scoring weights must be finite and non-negative.")
        if sum(weights) <= 0:
            raise ValueError("At least one scoring weight must be positive.")
        if self.grid_size < 1:
            raise ValueError("grid_size must be at least 1.")
        if self.min_parallax_deg < 0:
            raise ValueError("min_parallax_deg cannot be negative.")
        if self.good_reprojection_px <= 0:
            raise ValueError("good_reprojection_px must be positive.")


@dataclass
class ImageContributionMetrics:
    image_index: int
    pair_count: int = 0
    raw_matches: int = 0
    fundamental_inliers: int = 0
    pose_inliers: int = 0
    geometric_inlier_ratio: float = 0.0
    pose_retention_ratio: float = 0.0

    point_track_count: int = 0
    shared_point_count: int = 0
    mean_track_length: float = 0.0

    occupied_grid_cells: int = 0
    grid_coverage: float = 0.0
    coverage_entropy: float = 0.0
    distinctive_grid_cells: int = 0
    distinctive_coverage: float = 0.0

    reprojection_observations: int = 0
    mean_reprojection_error_px: Optional[float] = None
    median_reprojection_error_px: Optional[float] = None

    mean_parallax_deg: Optional[float] = None
    valid_parallax_tracks: int = 0

    normalized_metrics: Dict[str, float] = field(default_factory=dict)
    contribution_score: float = 0.0
    assessment: str = "needs review"
    notes: List[str] = field(default_factory=list)


@dataclass
class ContributionReport:
    images: List[ImageContributionMetrics]
    component_by_image: Dict[int, int]
    config: ContributionConfig
    warnings: List[str] = field(default_factory=list)

    def by_image_index(self, image_index: int) -> Optional[ImageContributionMetrics]:
        for item in self.images:
            if item.image_index == image_index:
                return item
        return None


def _get(obj: Any, names: Sequence[str], default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _finite_float(value, default=0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _pair_endpoints(pair) -> Optional[Tuple[int, int]]:
    """Read common endpoint naming conventions without requiring MatchPair."""
    aliases = (
        ("i", "j"),
        ("idx1", "idx2"),
        ("image1", "image2"),
        ("image_idx1", "image_idx2"),
        ("img1", "img2"),
        ("source", "target"),
        ("query_idx", "train_idx"),
    )
    for first, second in aliases:
        a = _get(pair, (first,))
        b = _get(pair, (second,))
        if a is not None and b is not None:
            try:
                return int(a), int(b)
            except (TypeError, ValueError):
                pass
    return None


def _keypoints_xy(feature) -> np.ndarray:
    raw = _get(feature, ("keypoints", "points", "keypoint_xy"))
    if raw is None:
        return np.empty((0, 2), dtype=np.float64)

    if isinstance(raw, np.ndarray):
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return np.asarray(arr[:, :2], dtype=np.float64)
        return np.empty((0, 2), dtype=np.float64)

    rows = []
    try:
        for item in raw:
            if hasattr(item, "pt"):
                rows.append(item.pt)
            else:
                rows.append(item[:2])
    except (TypeError, ValueError, IndexError):
        return np.empty((0, 2), dtype=np.float64)

    try:
        arr = np.asarray(rows, dtype=np.float64)
        return arr.reshape(-1, 2) if arr.size else np.empty((0, 2))
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)


def _image_dimensions(feature) -> Optional[Tuple[float, float]]:
    """Return width, height when the feature object exposes image dimensions."""
    shape = _get(feature, ("image_shape", "shape"))
    if shape is not None:
        try:
            values = tuple(shape)
            if len(values) >= 2:
                height, width = float(values[0]), float(values[1])
                if width > 0 and height > 0:
                    return width, height
        except (TypeError, ValueError):
            pass

    size = _get(feature, ("image_size", "size"))
    if size is not None:
        try:
            values = tuple(size)
            if len(values) >= 2:
                width, height = float(values[0]), float(values[1])
                if width > 0 and height > 0:
                    return width, height
        except (TypeError, ValueError):
            pass

    width = _get(feature, ("width",))
    height = _get(feature, ("height",))
    try:
        width, height = float(width), float(height)
        if width > 0 and height > 0:
            return width, height
    except (TypeError, ValueError):
        pass

    image = _get(feature, ("image", "original_image"))
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        height, width = image.shape[:2]
        if width > 0 and height > 0:
            return float(width), float(height)

    return None


def _valid_pose(pose):
    try:
        R, t = pose
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            return None
        return R, t
    except (TypeError, ValueError):
        return None


def _safe_unit(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def _angle_degrees(a, b) -> Optional[float]:
    ua, ub = _safe_unit(a), _safe_unit(b)
    if ua is None or ub is None:
        return None
    cosine = np.clip(np.dot(ua, ub), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _component_map(components, image_count):
    mapping = {}
    for component_id, component in enumerate(components or []):
        for image_index in component:
            try:
                index = int(image_index)
                if 0 <= index < image_count:
                    mapping[index] = component_id
            except (TypeError, ValueError):
                continue

    next_id = len(components or [])
    for index in range(image_count):
        if index not in mapping:
            mapping[index] = next_id
            next_id += 1
    return mapping


def _coverage(points: List[np.ndarray], dimensions, grid_size: int):
    if dimensions is None or not points:
        return 0, 0.0, 0.0, set()

    width, height = dimensions
    cells = set()
    counts: Dict[Tuple[int, int], int] = {}

    for point in points:
        x, y = float(point[0]), float(point[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < 0 or y < 0 or x >= width or y >= height:
            continue

        gx = min(grid_size - 1, int(x / width * grid_size))
        gy = min(grid_size - 1, int(y / height * grid_size))
        cell = (gx, gy)
        cells.add(cell)
        counts[cell] = counts.get(cell, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0, 0.0, 0.0, cells

    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    max_entropy = float(np.log(grid_size * grid_size))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return len(cells), len(cells) / float(grid_size * grid_size), normalized_entropy, cells


def _normalize_within_components(values, component_map):
    """Min-max normalize each metric inside each connected component."""
    output = {index: 0.0 for index in values}
    groups: Dict[int, List[int]] = {}

    for index, component_id in component_map.items():
        groups.setdefault(component_id, []).append(index)

    for indices in groups.values():
        valid = [values[i] for i in indices if values[i] is not None and np.isfinite(values[i])]
        if not valid:
            continue

        low, high = min(valid), max(valid)
        if high - low <= 1e-12:
            # Equal evidence is neutral, not an artificial ranking.
            for index in indices:
                output[index] = 0.5 if values[index] is not None else 0.0
        else:
            for index in indices:
                value = values[index]
                output[index] = (
                    float(np.clip((value - low) / (high - low), 0.0, 1.0))
                    if value is not None and np.isfinite(value)
                    else 0.0
                )
    return output


def score_image_contributions(result, config: Optional[ContributionConfig] = None) -> ContributionReport:
    """Score images in an AlignmentResult-like object.

    Expected point_tracks entries are (point_id, image_index, keypoint_index).
    Camera poses use world-to-camera extrinsics: X_camera = R @ X_world + t.
    Geometry is evaluated only within connected components.
    """
    config = config or ContributionConfig()

    features = list(_get(result, ("features",), []) or [])
    pairs = list(_get(result, ("pairs",), []) or [])
    components = _get(result, ("components",), []) or []
    poses = _get(result, ("camera_poses",), {}) or {}
    sparse_points = _get(result, ("sparse_points",), None)
    point_tracks = _get(result, ("point_tracks",), None) or []
    K = _get(result, ("camera_matrix",), None)

    image_count = len(features)
    metrics = [ImageContributionMetrics(image_index=i) for i in range(image_count)]
    warnings = []

    component_map = _component_map(components, image_count)
    component_members: Dict[int, set] = {}
    for index, component_id in component_map.items():
        component_members.setdefault(component_id, set()).add(index)

    # Pair-level evidence. Support several possible MatchPair field names.
    for pair in pairs:
        endpoints = _pair_endpoints(pair)
        if endpoints is None:
            continue
        a, b = endpoints
        if a == b or not (0 <= a < image_count and 0 <= b < image_count):
            continue
        if component_map[a] != component_map[b]:
            continue

        raw = max(0, int(_finite_float(_get(pair, ("raw_match_count",), 0))))
        fundamental = max(0, int(_finite_float(_get(pair, ("fundamental_inliers",), 0))))
        pose_inliers = max(0, int(_finite_float(_get(pair, ("inliers", "pose_inliers"), 0))))

        for index in (a, b):
            item = metrics[index]
            item.pair_count += 1
            item.raw_matches += raw
            item.fundamental_inliers += fundamental
            item.pose_inliers += pose_inliers

    for item in metrics:
        if item.raw_matches:
            item.geometric_inlier_ratio = item.fundamental_inliers / item.raw_matches
            item.pose_retention_ratio = (
                item.pose_inliers / item.fundamental_inliers
                if item.fundamental_inliers else 0.0
            )

    # Parse valid track observations.
    tracks_by_point: Dict[int, List[Tuple[int, int]]] = {}
    for track in point_tracks:
        try:
            point_id, image_index, keypoint_index = track[:3]
            point_id, image_index, keypoint_index = (
                int(point_id), int(image_index), int(keypoint_index)
            )
        except (TypeError, ValueError, IndexError):
            continue

        if not (0 <= image_index < image_count):
            continue
        if keypoint_index < 0:
            continue
        if point_id < 0:
            continue

        tracks_by_point.setdefault(point_id, []).append((image_index, keypoint_index))

    # Remove duplicate observations within a point track.
    for point_id in list(tracks_by_point):
        tracks_by_point[point_id] = sorted(set(tracks_by_point[point_id]))

    image_points: Dict[int, List[np.ndarray]] = {i: [] for i in range(image_count)}
    grid_cells: Dict[int, set] = {i: set() for i in range(image_count)}
    reprojection_errors: Dict[int, List[float]] = {i: [] for i in range(image_count)}
    parallax_angles: Dict[int, List[float]] = {i: [] for i in range(image_count)}
    track_lengths: Dict[int, List[int]] = {i: [] for i in range(image_count)}
    valid_track_ids: Dict[int, set] = {i: set() for i in range(image_count)}

    keypoints = [_keypoints_xy(feature) for feature in features]
    dimensions = [_image_dimensions(feature) for feature in features]

    # Track support and image-plane coverage.
    for point_id, observations in tracks_by_point.items():
        usable = [
            (image_index, keypoint_index)
            for image_index, keypoint_index in observations
            if keypoint_index < len(keypoints[image_index])
            and component_map[image_index] == component_map[observations[0][0]]
        ]
        if len(usable) < 2:
            continue

        for image_index, keypoint_index in usable:
            valid_track_ids[image_index].add(point_id)
            track_lengths[image_index].append(len(usable))
            uv = keypoints[image_index][keypoint_index]
            if np.all(np.isfinite(uv)):
                image_points[image_index].append(uv)

    for i, item in enumerate(metrics):
        item.point_track_count = len(valid_track_ids[i])
        item.shared_point_count = item.point_track_count
        item.mean_track_length = (
            float(np.mean(track_lengths[i])) if track_lengths[i] else 0.0
        )

        occupied, coverage, entropy, cells = _coverage(
            image_points[i], dimensions[i], config.grid_size
        )
        item.occupied_grid_cells = occupied
        item.grid_coverage = coverage
        item.coverage_entropy = entropy
        grid_cells[i] = cells

    # Reprojection and parallax require calibrated intrinsics, poses and 3D points.
    K_array = None
    try:
        K_array = np.asarray(K, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(K_array)) or abs(np.linalg.det(K_array)) < 1e-12:
            K_array = None
    except (TypeError, ValueError):
        K_array = None

    points_array = None
    try:
        points_array = np.asarray(sparse_points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            points_array = None
    except (TypeError, ValueError):
        points_array = None

    valid_poses = {}
    for index, pose in poses.items():
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < image_count:
            parsed = _valid_pose(pose)
            if parsed is not None:
                valid_poses[index] = parsed

    if K_array is not None and points_array is not None:
        try:
            K_inverse = np.linalg.inv(K_array)
        except np.linalg.LinAlgError:
            K_inverse = None

        if K_inverse is not None:
            for point_id, observations in tracks_by_point.items():
                if point_id >= len(points_array):
                    continue
                X = points_array[point_id]
                if not np.all(np.isfinite(X)):
                    continue

                usable = [
                    (image_index, keypoint_index)
                    for image_index, keypoint_index in observations
                    if image_index in valid_poses
                    and keypoint_index < len(keypoints[image_index])
                ]
                if len(usable) < 2:
                    continue

                # Do not compare camera geometry across disconnected components.
                component_ids = {component_map[image_index] for image_index, _ in usable}
                if len(component_ids) != 1:
                    continue

                camera_rays = {}
                for image_index, keypoint_index in usable:
                    R, t = valid_poses[image_index]
                    X_camera = R @ X + t
                    if not np.all(np.isfinite(X_camera)) or X_camera[2] <= 1e-9:
                        continue

                    uv_h = K_array @ X_camera
                    if abs(uv_h[2]) <= 1e-12:
                        continue
                    predicted = uv_h[:2] / uv_h[2]
                    observed = keypoints[image_index][keypoint_index]
                    if not np.all(np.isfinite(observed)):
                        continue

                    error = float(np.linalg.norm(predicted - observed))
                    if np.isfinite(error):
                        reprojection_errors[image_index].append(error)

                    pixel_ray = K_inverse @ np.array(
                        [observed[0], observed[1], 1.0], dtype=np.float64
                    )
                    world_ray = _safe_unit(R.T @ pixel_ray)
                    if world_ray is not None:
                        camera_rays[image_index] = world_ray

                for image_index in camera_rays:
                    angles = [
                        _angle_degrees(camera_rays[image_index], other_ray)
                        for other_index, other_ray in camera_rays.items()
                        if other_index != image_index
                    ]
                    angles = [a for a in angles if a is not None]
                    if angles:
                        parallax_angles[image_index].append(max(angles))

    else:
        warnings.append(
            "Reprojection and parallax metrics unavailable: valid camera_matrix, "
            "camera_poses and sparse_points are required."
        )

    for i, item in enumerate(metrics):
        errors = reprojection_errors[i]
        item.reprojection_observations = len(errors)
        if errors:
            item.mean_reprojection_error_px = float(np.mean(errors))
            item.median_reprojection_error_px = float(np.median(errors))

        angles = parallax_angles[i]
        item.valid_parallax_tracks = len(angles)
        if angles:
            item.mean_parallax_deg = float(np.mean(angles))

    # Normalize each metric within its own connected component.
    raw_metrics = {
        "pair_support": {
            i: float(metrics[i].pose_inliers) for i in range(image_count)
        },
        "track_support": {
            i: float(metrics[i].point_track_count) for i in range(image_count)
        },
        "coverage": {
            i: float(metrics[i].grid_coverage) for i in range(image_count)
        },
        "reprojection_quality": {
            i: (
                1.0 / (1.0 + metrics[i].mean_reprojection_error_px)
                if metrics[i].mean_reprojection_error_px is not None else None
            )
            for i in range(image_count)
        },
        "parallax": {
            i: (
                float(metrics[i].mean_parallax_deg)
                if metrics[i].mean_parallax_deg is not None else None
            )
            for i in range(image_count)
        },
    }

    normalized = {
        name: _normalize_within_components(values, component_map)
        for name, values in raw_metrics.items()
    }

    weights = {
        "pair_support": config.pair_support_weight,
        "track_support": config.track_support_weight,
        "coverage": config.image_coverage_weight,
        "reprojection_quality": config.reprojection_weight,
        "parallax": config.parallax_weight,
    }
    total_weight = sum(weights.values())

    for i, item in enumerate(metrics):
        available = [
            name for name in weights
            if raw_metrics[name][i] is not None and weights[name] > 0
        ]
        available_weight = sum(weights[name] for name in available)

        if available_weight > 0:
            item.normalized_metrics = {
                name: normalized[name][i] for name in weights
            }
            item.contribution_score = 100.0 * sum(
                normalized[name][i] * weights[name] for name in available
            ) / available_weight

        # Assessments are review guidance, not automatic keep/delete decisions.
        if item.point_track_count == 0 and item.pose_inliers == 0:
            item.assessment = "needs review"
            item.notes.append("No verified pair or multi-view point support was found.")
        elif item.contribution_score >= 70:
            item.assessment = "high contribution"
        elif item.contribution_score >= 35:
            item.assessment = "supporting"
        else:
            item.assessment = "possible redundancy — review before excluding"

        if item.mean_reprojection_error_px is not None:
            if item.mean_reprojection_error_px > config.good_reprojection_px:
                item.notes.append("Reprojection error is above the configured target.")
        else:
            item.notes.append("Reprojection quality could not be evaluated.")

        if item.mean_parallax_deg is not None:
            if item.mean_parallax_deg < config.min_parallax_deg:
                item.notes.append("Observed parallax is below the configured threshold.")
        else:
            item.notes.append("Parallax could not be evaluated.")


    return ContributionReport(
        images=metrics,
        component_by_image=component_map,
        config=config,
        warnings=warnings,
    )
