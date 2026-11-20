import cv2
import numpy as np


def robust_triangulate(observations, poses, K, threshold=2.5):
    """
    Experimental pair-hypothesis triangulation.
    Returns a point and its inlier camera IDs, or (None, []).
    """
    camera_ids = sorted(observations)
    candidates = []

    for i in range(len(camera_ids)):
        for j in range(i + 1, len(camera_ids)):
            ia, ib = camera_ids[i], camera_ids[j]

            Ra, ta = poses[ia]
            Rb, tb = poses[ib]

            P1 = K @ np.hstack((Ra, ta.reshape(3, 1)))
            P2 = K @ np.hstack((Rb, tb.reshape(3, 1)))

            uv1 = np.asarray(observations[ia], dtype=float).reshape(2, 1)
            uv2 = np.asarray(observations[ib], dtype=float).reshape(2, 1)

            h = cv2.triangulatePoints(P1, P2, uv1, uv2).reshape(4)

            if abs(h[3]) < 1e-10:
                continue

            point = h[:3] / h[3]

            if not np.all(np.isfinite(point)):
                continue

            inliers = []
            errors = []

            for image_id in camera_ids:
                R, t = poses[image_id]
                camera_point = R @ point.reshape(3, 1) + t.reshape(3, 1)

                if camera_point[2, 0] <= 1e-8:
                    continue

                projected = K @ camera_point
                projected = (projected[:2] / projected[2, 0]).ravel()

                error = float(np.linalg.norm(
                    projected - np.asarray(observations[image_id])
                ))

                if error <= threshold:
                    inliers.append(image_id)
                    errors.append(error)

            if len(inliers) >= 3:
                candidates.append((
                    len(inliers),
                    -float(np.median(errors)),
                    point,
                    inliers,
                ))

    if not candidates:
        return None, []

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    _, _, point, inliers = candidates[0]
    return point, inliers


def make_scene(corrupt_ids=()):
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    point_true = np.array([0.2, 0.1, 5.0])

    centres = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ]

    poses = {}
    observations = {}

    for image_id, centre in enumerate(centres):
        R = np.eye(3)
        t = -centre.reshape(3, 1)
        poses[image_id] = (R, t)

        camera_point = R @ point_true.reshape(3, 1) + t
        pixel = K @ camera_point
        uv = (pixel[:2] / pixel[2, 0]).ravel()

        if image_id in corrupt_ids:
            uv += np.array([20.0, 0.0])

        observations[image_id] = uv

    return observations, poses, K, point_true


def test_experiment_recovers_clean_point():
    observations, poses, K, truth = make_scene()

    point, inliers = robust_triangulate(observations, poses, K)

    assert point is not None
    assert len(inliers) == 4
    assert np.linalg.norm(point - truth) < 1e-5


def test_experiment_recovers_point_with_one_bad_camera():
    observations, poses, K, truth = make_scene(corrupt_ids=(1,))

    point, inliers = robust_triangulate(observations, poses, K)

    assert point is not None
    assert len(inliers) == 3
    assert 1 not in inliers
    assert np.linalg.norm(point - truth) < 1e-5


def test_experiment_rejects_two_bad_cameras():
    observations, poses, K, _ = make_scene(corrupt_ids=(1, 2))

    point, inliers = robust_triangulate(observations, poses, K)

    assert point is None
    assert inliers == []

