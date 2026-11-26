"""Dense cloud on a rendered scene: points must lie on the real surfaces."""
import sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sfm import IncrementalSfM, camera_center  # noqa: E402
from src.dense import densify, write_ply  # noqa: E402
from tests.synthetic_scene import make_scene  # noqa: E402
from tests.test_sfm import umeyama  # noqa: E402


def surface_distance(X):
    """Distance to the ground (z=0, |x|,|y|≤3) or the box (|x|,|y|≤0.8, 0≤z≤1.2)."""
    g = np.abs(X[:, 2])
    b, h = 0.8, 1.2
    dx = np.maximum(np.abs(X[:, 0]) - b, 0); dy = np.maximum(np.abs(X[:, 1]) - b, 0)
    inside = (np.abs(X[:, 0]) <= b) & (np.abs(X[:, 1]) <= b) & (X[:, 2] >= 0) & (X[:, 2] <= h)
    outside_d = np.sqrt(dx ** 2 + dy ** 2 + np.maximum(X[:, 2] - h, 0) ** 2 + np.maximum(-X[:, 2], 0) ** 2)
    inside_d = np.minimum.reduce([b - np.abs(X[:, 0]), b - np.abs(X[:, 1]), h - X[:, 2], X[:, 2]])
    box = np.where(inside, inside_d, outside_d)
    return np.minimum(g, box)


def test_dense_cloud_hugs_the_surfaces():
    with tempfile.TemporaryDirectory() as td:
        paths, gt, K = make_scene(td + "/i", n=24)
        rec = IncrementalSfM(max_features=4000).run(paths)
        reg = rec.registered_views
        s, R, t = umeyama(np.array([camera_center(v) for v in reg]), gt[[v.id for v in reg]])
        t0 = time.time()
        cloud = densify(rec)
        dt = time.time() - t0
        X = (s * cloud.points.astype(np.float64) @ R.T) + t
        d = surface_distance(X)
        sparse_n = len(rec.points)
        print(f"dense: {len(cloud):,} points from {cloud.pairs_used} pairs in {dt:.1f}s (sparse {sparse_n:,}); "
              f"surface error median {np.median(d):.4f}, p90 {np.percentile(d, 90):.4f} (scene ~6 units)")
        assert len(cloud) > 10 * sparse_n, "dense cloud is far denser than the sparse one"
        assert np.median(d) < 0.02 and np.percentile(d, 90) < 0.08
        on_box = np.mean((np.abs(X[:, 0]) < 0.85) & (np.abs(X[:, 1]) < 0.85) & (X[:, 2] > 0.05))
        assert on_box > 0.03, "the box (the object) is well represented"
        write_ply(td + "/dense.ply", cloud)
        assert Path(td + "/dense.ply").stat().st_size > len(cloud) * 15


if __name__ == "__main__":
    test_dense_cloud_hugs_the_surfaces()
    print("dense: all tests passed")


def test_brush_cloud_even_and_on_target(tmp_path):
    import numpy as np
    from src.dense import even_cloud, fill_small_gaps, remove_outliers
    rng = np.random.default_rng(0)
    # a plane with a dense clump, a sparse half, pin-holes and a few strays
    P = np.concatenate([rng.random((40000, 2)) * [1, 1], rng.random((20000, 2)) * [0.2, 0.2]])
    P = np.c_[P, np.zeros(len(P))].astype(np.float32)
    holes = (np.hypot(P[:, 0] - 0.7, P[:, 1] - 0.7) < 0.01)
    P = P[~holes]
    stray = (rng.random((50, 3)) * 5 + 3).astype(np.float32)
    P = np.concatenate([P, stray]); C = np.full((len(P), 3), 128, np.uint8)
    keep = remove_outliers(P, k=10, std=2.0)
    assert keep[-50:].sum() == 0 and keep[:-50].mean() > 0.98
    E, EC, vox = even_cloud(P[keep], C[keep], 5000)
    assert 4000 < len(E) < 6000 and vox > 0
    # even: the clump no longer has many times the density
    clump = ((E[:, 0] < 0.2) & (E[:, 1] < 0.2)).mean()
    assert clump < 0.07, clump
    F, FC, added = fill_small_gaps(E, EC, vox)
    assert added > 0 and len(F) == len(E) + added and np.all(np.abs(F[:, 2]) < 1e-5)
