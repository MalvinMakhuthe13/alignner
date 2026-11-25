"""End-to-end SfM checks on a synthetic scene with known cameras."""
import sys, tempfile, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sfm import IncrementalSfM, export_colmap, read_colmap_bin, camera_center, project  # noqa: E402
from tests.synthetic_scene import make_scene, make_flyover  # noqa: E402


def umeyama(src, dst):
    """Similarity transform (s, R, t) minimising |s R src + t − dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def run_case(n=24, k1=0.0, focal_hint=None):
    with tempfile.TemporaryDirectory() as td:
        paths, gt_centers, K = make_scene(td + "/imgs", n=n, k1=k1)
        t0 = time.time()
        sfm = IncrementalSfM(focal_px=focal_hint)
        rec = sfm.run(paths)
        dt = time.time() - t0
        st = rec.stats()
        reg = rec.registered_views
        est = np.array([camera_center(v) for v in reg])
        gt = gt_centers[[v.id for v in reg]]
        s, R, t = umeyama(est, gt)
        err = np.linalg.norm((s * est @ R.T + t) - gt, axis=1)
        out = export_colmap(rec, td + "/colmap")
        back = read_colmap_bin(td + "/colmap/sparse/0")
        return st, err, dt, out, back, rec


def test_orbit_recovers_cameras_and_points():
    st, err, dt, out, back, rec = run_case(24)
    print("stats", st, f"pose err mean {err.mean():.4f} max {err.max():.4f} (orbit radius 5)", f"{dt:.1f}s")
    assert st["registered"] == 24
    assert st["points"] > 2000
    assert st["mean_reprojection_error_px"] < 1.0
    assert err.max() < 0.05, err          # < 1 % of the orbit radius
    assert st["mean_track_length"] > 2.5   # real multi-view tracks, not pairwise
    # focal recovered from a wrong 1.2×max(w,h) guess (true 1100, guess 1536)
    f = list(rec.cameras.values())[0].f
    assert abs(f - 1100) / 1100 < 0.02, f
    # COLMAP binary round trip is complete and consistent
    assert len(back["images"]) == 24 and len(back["points"]) == st["points"]
    some = next(iter(back["points"].values()))
    assert len(some[3]) >= 2, "points carry tracks"
    img = next(iter(back["images"].values()))
    assert len(img[4]) > 100 and all(p[2] > 0 for p in img[4]), "images carry 2D↔3D links"
    # independent check: re-project every exported 2D↔3D link with the exported cameras
    errs = exported_reprojection(back)
    assert errs.mean() < 0.5 and errs.max() <= 4.0 + 1e-6, (errs.mean(), errs.max())
    for pid, (_, _, _, tr) in back["points"].items():
        for iid, idx in tr:
            assert back["images"][iid][4][idx][2] == pid


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def exported_reprojection(back):
    errs = []
    for q, t, cid, name, p2d in back["images"].values():
        fx, fy, cx, cy = back["cameras"][cid][3]
        R = q2R(q)
        for x, y, pid in p2d:
            Xc = R @ back["points"][pid][0] + t
            errs.append(np.hypot(fx * Xc[0] / Xc[2] + cx - x, fy * Xc[1] / Xc[2] + cy - y))
    return np.array(errs)


def test_lens_distortion_is_estimated_and_undistorted():
    st, err, dt, out, back, rec = run_case(20, k1=-0.08, focal_hint=1100)
    k = list(rec.cameras.values())[0].k
    print("distortion", k, st, f"pose err max {err.max():.4f}")
    assert st["registered"] == 20
    assert abs(k - (-0.08)) < 0.02, k
    assert out["undistorted"] is True
    assert st["mean_reprojection_error_px"] < 1.0
    assert err.max() < 0.06


def _check_alignment(paths, gt_centers, order=None, **kw):
    sfm = IncrementalSfM(max_features=5000, **kw)
    rec = sfm.run([paths[i] for i in order] if order is not None else paths)
    st = rec.stats()
    reg = rec.registered_views
    idx = [order[v.id] if order is not None else v.id for v in reg]
    est = np.array([camera_center(v) for v in reg])
    s, R, t = umeyama(est, gt_centers[idx])
    err = np.linalg.norm((s * est @ R.T + t) - gt_centers[idx], axis=1)
    return st, err


def test_shuffled_large_set_uses_retrieval():
    with tempfile.TemporaryDirectory() as td:
        paths, gt, _ = make_scene(td, n=30)
        order = list(np.random.default_rng(3).permutation(30))
        st, err = _check_alignment(paths, gt, order=order, exhaustive_limit=10, sequential_window=3)
        print("shuffled", st, f"pose err max {err.max():.4f}")
        assert st["registered"] >= 28
        assert err.max() < 0.05
        assert st["mean_reprojection_error_px"] < 1.0


def test_flyover_grid():
    with tempfile.TemporaryDirectory() as td:
        paths, gt, _ = make_flyover(td)
        # Flat nadir flights can't tell focal length from altitude on their own,
        # so (like real drone photos) the focal comes from EXIF / the camera.
        st, err = _check_alignment(paths, gt, focal_px=1000.0)
        print("flyover", st, f"pose err max {err.max():.4f} (spacing 0.9)")
        assert st["registered"] == len(paths)
        assert err.max() < 0.005
        assert st["mean_reprojection_error_px"] < 1.0





def test_exif_focal_is_used():
    from PIL import Image
    from src.sfm import exif_focal_px
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.jpg"
        im = Image.new("RGB", (4000, 3000))
        ex = Image.Exif()
        ex[0x8769] = {0xA405: 28}           # 28 mm equivalent
        im.save(p, exif=ex)
        f = exif_focal_px(str(p), 4000, 3000)
        assert f is not None and abs(f - 28 / 36 * 4000) < 1e-6, f


if __name__ == "__main__":
    test_exif_focal_is_used()
    test_shuffled_large_set_uses_retrieval()
    test_flyover_grid()
    test_orbit_recovers_cameras_and_points()
    test_lens_distortion_is_estimated_and_undistorted()
    print("sfm: all tests passed")
