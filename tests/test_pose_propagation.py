from types import SimpleNamespace

import numpy as np

from src.aligner import ImageAligner


def _pair(idx1, idx2, rotation=None, translation=None):
    return SimpleNamespace(
        idx1=idx1,
        idx2=idx2,
        R=np.eye(3) if rotation is None else rotation,
        t=np.array([[1.0], [0.0], [0.0]]) if translation is None else translation,
    )


def test_propagate_poses_covers_disconnected_components():
    aligner = ImageAligner()
    pairs = [
        _pair(0, 1),
        _pair(2, 3, translation=np.array([[0.0], [2.0], [0.0]])),
    ]

    poses = aligner.propagate_poses(pairs, 5)

    assert set(poses) == {0, 1, 2, 3}
    np.testing.assert_allclose(poses[0][0], np.eye(3))
    np.testing.assert_allclose(poses[2][0], np.eye(3))
    np.testing.assert_allclose(poses[1][1].ravel(), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(poses[3][1].ravel(), [0.0, 2.0, 0.0])


def test_propagate_poses_handles_empty_graph():
    poses = ImageAligner().propagate_poses([], 3)

    assert poses == {}
