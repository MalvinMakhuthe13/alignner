from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QualityProfile = Literal["fast", "balanced", "high_quality"]
MatcherType = Literal["exhaustive", "sequential", "spatial"]


@dataclass(frozen=True)
class ReconstructionConfig:
    profile: QualityProfile = "balanced"

    # Resource limits
    num_threads: int = 4
    max_image_size: int = 2400

    # COLMAP feature extraction and matching
    max_num_features: int = 8192
    matcher: MatcherType = "exhaustive"
    guided_matching: bool = True

    # Sequential matching
    sequential_overlap: int = 10
    sequential_loop_detection: bool = False

    # Spatial matching (requires usable image position priors)
    spatial_ignore_z: bool = True
    spatial_max_num_neighbors: int = 50
    spatial_min_num_neighbors: int = 0
    spatial_max_distance: float = 100.0

    # Sparse reconstruction
    min_num_matches: int = 15
    multiple_models: bool = False

    # Dense reconstruction
    resolution_level: int = 2
    densify_num_views: int = 5
    estimate_colors: bool = True

    # Mesh reconstruction
    mesh_resolution_level: int = 2
    mesh_min_point_distance: float = 2.5

    # Optional finishing stages
    refine_mesh: bool = True
    texture_mesh: bool = True

    def __post_init__(self):
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        if self.matcher not in ("exhaustive", "sequential", "spatial"):
            raise ValueError(
                f"Unsupported matcher: {self.matcher}. "
                "Choose exhaustive, sequential, or spatial."
            )

        if self.sequential_overlap < 1:
            raise ValueError("sequential_overlap must be at least 1")

        if self.spatial_min_num_neighbors < 0:
            raise ValueError("spatial_min_num_neighbors cannot be negative")

        if self.spatial_max_num_neighbors < 1:
            raise ValueError("spatial_max_num_neighbors must be at least 1")

        if self.spatial_min_num_neighbors > self.spatial_max_num_neighbors:
            raise ValueError(
                "spatial_min_num_neighbors cannot exceed "
                "spatial_max_num_neighbors"
            )

        if self.spatial_max_distance <= 0:
            raise ValueError("spatial_max_distance must be positive")

    @classmethod
    def from_profile(
        cls,
        profile: QualityProfile = "balanced",
        num_threads: int = 4,
    ) -> "ReconstructionConfig":

        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        profiles = {
            "fast": {
                "max_image_size": 1600,
                "max_num_features": 4096,
                "resolution_level": 3,
                "densify_num_views": 3,
                "mesh_resolution_level": 3,
                "refine_mesh": False,
                "texture_mesh": False,
            },
            "balanced": {
                "max_image_size": 2400,
                "max_num_features": 8192,
                "resolution_level": 2,
                "densify_num_views": 5,
                "mesh_resolution_level": 2,
                "refine_mesh": True,
                "texture_mesh": True,
            },
            "high_quality": {
                "max_image_size": 3200,
                "max_num_features": 12000,
                "resolution_level": 1,
                "densify_num_views": 8,
                "mesh_resolution_level": 1,
                "refine_mesh": True,
                "texture_mesh": True,
            },
        }

        if profile not in profiles:
            raise ValueError(f"Unknown reconstruction profile: {profile}")

        return cls(
            profile=profile,
            num_threads=num_threads,
            **profiles[profile],
        )
