import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.aligner import ImageAligner, MatchPair


def _pair(i, j, rotations, translations):
    R = rotations[j] @ rotations[i].T
    t = translations[j] - R @ translations[i]
    t = np.asarray(t, dtype=float).reshape(3, 1)
    if np.linalg.norm(t) < 1e-8:
        t = np.array([[1.0], [0.0], [0.0]])
    else:
        t = t / np.linalg.norm(t)
    return MatchPair(i, j, [], 20, R=R, t=t)


def _rotation_error(pairs, poses):
    values = []
    for pair in pairs:
        predicted = poses[pair.idx2][0] @ poses[pair.idx1][0].T
        values.append(np.linalg.norm(
            Rotation.from_matrix(pair.R.T @ predicted).as_rotvec()
        ))
    return float(np.mean(values))


def test_pose_graph_reduces_global_rotation_inconsistency():
    rotations_true = {
        0: np.eye(3),
        1: Rotation.from_euler("z", 0.12).as_matrix(),
        2: Rotation.from_euler("z", 0.24).as_matrix(),
    }
    translations = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([-1.0, 0.0, 0.0]),
        2: np.array([-2.0, 0.0, 0.0]),
    }
    pairs = [
        _pair(0, 1, rotations_true, translations),
        _pair(1, 2, rotations_true, translations),
        _pair(0, 2, rotations_true, translations),
    ]
    initial = {
        0: (np.eye(3), np.zeros((3, 1))),
        1: (rotations_true[1], translations[1].reshape(3, 1)),
        2: (Rotation.from_euler("z", 0.34).as_matrix(),
            np.array([[-2.2], [0.1], [0.0]])),
    }
    before = _rotation_error(pairs, initial)

    optimized = ImageAligner().optimize_pose_graph(pairs, initial, [[0, 1, 2]])

    assert _rotation_error(pairs, optimized) < before
    np.testing.assert_allclose(optimized[0][0], np.eye(3), atol=1e-8)
    assert all(np.all(np.isfinite(R)) and np.all(np.isfinite(t))
               for R, t in optimized.values())
