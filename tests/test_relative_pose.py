import cv2
import numpy as np

from src.aligner import ImageAligner


def test_relative_pose_returns_the_same_cheirality_verified_matches():
    rng = np.random.default_rng(17)
    K = np.array(
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    R_true, _ = cv2.Rodrigues(np.array([0.025, -0.04, 0.015]))
    t_true = np.array([0.8, 0.03, 0.02], dtype=np.float64)
    points = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, 100),
            rng.uniform(-1.0, 1.0, 100),
            rng.uniform(5.0, 10.0, 100),
        ]
    )

    points_2 = (R_true @ points.T).T + t_true
    projected_1 = (K @ points.T).T
    projected_1 = projected_1[:, :2] / projected_1[:, 2:3]
    projected_2 = (K @ points_2.T).T
    projected_2 = projected_2[:, :2] / projected_2[:, 2:3]

    # Replace the final 20 second-view observations with unrelated outliers.
    projected_2[80:] = np.column_stack(
        [rng.uniform(0, 1280, 20), rng.uniform(0, 720, 20)]
    )

    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in projected_1]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in projected_2]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(len(kp1))]

    aligner = ImageAligner(min_inliers=20, ransac_thresh=1.0)
    R, t, inlier_count, pose_matches, E = aligner.estimate_relative_pose(
        kp1, kp2, matches, K
    )

    assert R is not None
    assert t is not None
    assert E is not None
    assert inlier_count == len(pose_matches)
    assert 20 <= inlier_count < len(matches)
    assert all(match.queryIdx < 80 for match in pose_matches)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-3)
    assert np.all(np.isfinite(t))
