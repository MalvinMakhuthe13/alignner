"""Scene tools on a real (synthetic) reconstruction: level, scale, GPS, cleanup, quality."""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sfm import IncrementalSfM, camera_center, export_colmap, read_colmap_bin  # noqa: E402
from src import scene_tools as T  # noqa: E402
from tests.synthetic_scene import make_scene  # noqa: E402
from tests.test_sfm import exported_reprojection  # noqa: E402

LAT0, LON0, ALT0 = -26.2041, 28.0473, 1753.0          # Johannesburg


def _write_gps(path, xyz):
    """Treat synthetic metres (x east, y north, z up) as GPS around LAT0/LON0."""
    R = 6378137.0
    lat = LAT0 + math.degrees(xyz[1] / R)
    lon = LON0 + math.degrees(xyz[0] / (R * math.cos(math.radians(LAT0))))
    alt = ALT0 + xyz[2]

    def dms(v):
        v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
        return (d, m, s)
    im = Image.open(path)
    ex = Image.Exif()
    ex[0x8825] = {1: "S" if lat < 0 else "N", 2: dms(lat), 3: "W" if lon < 0 else "E", 4: dms(lon), 5: 0, 6: alt}
    im.save(path, exif=ex, quality=95)


def build(n=16, gps=False):
    td = tempfile.mkdtemp()
    paths, gt, K = make_scene(td + "/i", n=n)
    if gps:
        for p, c in zip(paths, gt):
            _write_gps(p, c)
    rec = IncrementalSfM(max_features=3500, focal_px=1100).run(paths)
    return td, rec, gt


def test_all():
    td, rec, gt = build(16, gps=True)
    assert len(rec.registered_views) == 16

    # quality analysis
    q = T.image_quality(rec)
    assert all(x.registered and x.mean_error_px < 1.0 and x.observations > 100 for x in q), q[:2]
    assert len(T.components(rec)) == 1

    # level: Y-up; the true up (+Z in the synthetic world) must become +Y
    est = np.array([camera_center(v) for v in rec.registered_views])
    s, R, t = T.umeyama(est, gt[[v.id for v in rec.registered_views]])
    up_est = T.estimate_up(rec)
    assert np.dot(R @ up_est, [0, 0, 1]) > 0.99, R @ up_est
    T.level_scene(rec, up_axis="y")
    # ground-plane points (true z≈0) now share one height
    est2 = np.array([camera_center(v) for v in rec.registered_views])
    ys = est2[:, 1]
    # camera heights vary only by the scene's ±0.4 bob relative to radius 5
    assert ys.std() / np.linalg.norm(est2[:, [0, 2]], axis=1).mean() < 0.12
    X = np.array([p.xyz for p in rec.points])
    assert np.abs(np.median(X, axis=0)).max() < 1e-6, 'subject centred'

    # projections unchanged after transforms (export round trip)
    export_colmap(rec, td + "/c1")
    assert exported_reprojection(read_colmap_bin(td + "/c1/sparse/0")).mean() < 0.5

    # define distance
    a, b = rec.registered_views[0].id, rec.registered_views[5].id
    T.define_distance(rec, a, b, 2.5)
    assert abs(np.linalg.norm(camera_center(rec.views[a]) - camera_center(rec.views[b])) - 2.5) < 1e-9

    # GPS georeference: metres, east-north-up
    info = T.georeference_from_gps(rec)
    print("gps", info)
    assert info["images_with_gps"] == 16 and info["rms_m"] < 0.02
    est3 = np.array([camera_center(v) for v in rec.registered_views])
    d_true = np.linalg.norm(gt[0] - gt[8])
    d_est = np.linalg.norm(est3[[v.id for v in rec.registered_views].index(0)] - est3[[v.id for v in rec.registered_views].index(8)])
    assert abs(d_est - d_true) / d_true < 0.01, (d_est, d_true)

    # cleanup
    before = len(rec.points)
    removed = T.clean_points(rec, min_track=3, outlier_k=8, outlier_std=2.5)
    assert 0 < removed < before and len(rec.points) == before - removed
    lo, hi = T.robust_bounds(rec)
    T.crop_to_region(rec, lo, hi)
    export_colmap(rec, td + "/c2")
    assert exported_reprojection(read_colmap_bin(td + "/c2/sparse/0")).mean() < 0.5


if __name__ == "__main__":
    test_all()
    print("scene tools: all tests passed")
