import cv2
import numpy as np

from src.aligner import ImageAligner, ImageFeatures


def test_bundle_adjust_reduces_reprojection_error():
    rng = np.random.default_rng(12)
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points_true = np.column_stack([
        rng.uniform(-1.0, 1.0, 20),
        rng.uniform(-0.7, 0.7, 20),
        rng.uniform(4.0, 7.0, 20),
    ])
    R0, t0 = np.eye(3), np.zeros((3, 1))
    R1, t1_true = np.eye(3), np.array([[-1.0], [0.0], [0.0]])

    def project(R, t, point):
        camera = R @ point.reshape(3, 1) + t
        pixel = K @ camera
        return (pixel[:2] / pixel[2, 0]).ravel()

    observations0 = [project(R0, t0, p) for p in points_true]
    observations1 = [project(R1, t1_true, p) for p in points_true]
    features = [
        ImageFeatures("a.jpg", [cv2.KeyPoint(float(u), float(v), 1) for u, v in observations0],
                      np.empty((0, 32)), None, 640, 480),
        ImageFeatures("b.jpg", [cv2.KeyPoint(float(u), float(v), 1) for u, v in observations1],
                      np.empty((0, 32)), None, 640, 480),
    ]
    poses = {
        0: (R0, t0),
        1: (R1, t1_true + np.array([[0.12], [0.05], [0.03]])),
    }
    points_initial = points_true + rng.normal(0.0, 0.035, points_true.shape)
    tracks = [(pid, 0, pid) for pid in range(len(points_true))]
    tracks += [(pid, 1, pid) for pid in range(len(points_true))]

    def mean_error(current_poses, current_points):
        errors = []
        for pid, image_idx, kp_idx in tracks:
            R, t = current_poses[image_idx]
            predicted = project(R, t, current_points[pid])
            observed = np.asarray(features[image_idx].keypoints[kp_idx].pt)
            errors.append(np.linalg.norm(predicted - observed))
        return float(np.mean(errors))

    before = mean_error(poses, points_initial)
    optimized_poses, optimized_points = ImageAligner().bundle_adjust(
        features, poses, points_initial, tracks, K, components=[[0, 1]], max_nfev=80
    )
    after = mean_error(optimized_poses, optimized_points)

    assert after < before
    np.testing.assert_allclose(optimized_poses[0][1], t0)
