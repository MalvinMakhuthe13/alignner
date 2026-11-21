
import cv2
import numpy as np

from src.aligner import MatchPair
from src.pose_diagnostics import pose_constraint_diagnostics


def make_pair(R, t):
    return MatchPair(
        idx1=0,
        idx2=1,
        matches=[],
        inliers=20,
        R=R,
        t=np.asarray(t, dtype=np.float64).reshape(3, 1),
    )


def test_consistent_poses_have_zero_residual():
    R = np.eye(3)
    t = np.array([1.0, 0.0, 0.0])

    poses = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (R, t.reshape(3, 1)),
    }

    result = pose_constraint_diagnostics(
        [make_pair(R, t)],
        poses,
        [[0, 1]],
    )

    assert result["edges_evaluated"] == 1
    assert result["rotation"]["median_deg"] == 0.0
    assert result["translation_direction"]["median_deg"] == 0.0


def test_rotation_perturbation_is_detected():
    angle = np.radians(10.0)

    R_measured = np.eye(3)
    R_perturbed = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])

    poses = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (R_perturbed, np.array([[1.0], [0.0], [0.0]])),
    }

    result = pose_constraint_diagnostics(
        [make_pair(R_measured, [1.0, 0.0, 0.0])],
        poses,
        [[0, 1]],
    )

    assert abs(result["rotation"]["median_deg"] - 10.0) < 1e-5


def test_translation_magnitude_does_not_change_direction_residual():
    R = np.eye(3)

    poses = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (R, np.array([[5.0], [0.0], [0.0]])),
    }

    pair_small = make_pair(R, [1.0, 0.0, 0.0])
    pair_large = make_pair(R, [100.0, 0.0, 0.0])

    small = pose_constraint_diagnostics(
        [pair_small], poses, [[0, 1]]
    )
    large = pose_constraint_diagnostics(
        [pair_large], poses, [[0, 1]]
    )

    assert small["translation_direction"]["median_deg"] == 0.0
    assert large["translation_direction"]["median_deg"] == 0.0
