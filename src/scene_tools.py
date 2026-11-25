"""
Scene tools that act on an aligned reconstruction (RealityScan-style).

- quality analysis: per-image reprojection error, observations, coverage
- point cleanup: short / high-error tracks and statistical outliers
- level scene: put "up" where viewers expect it (Y-up for Fumoca / three.js,
  Z-up for RealityScan / Blender) and centre the subject at the origin
- georeference: scale + orient from photo GPS (drone / phone captures) so the
  model is in metres, east-north-up
- define distance: scale so two cameras are a known distance apart

All of these change the model consistently (cameras and points together),
so the exported COLMAP stays valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .sfm import Reconstruction, camera_center, project


# ─────────────────────────────────────────── similarity transforms

def apply_similarity(rec: Reconstruction, s: float, R: np.ndarray, t: np.ndarray) -> None:
    """X' = s·R·X + t for every point; cameras updated so projections are unchanged."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    for tr in rec.tracks:
        if tr.xyz is not None:
            tr.xyz = s * (R @ tr.xyz) + t
    for v in rec.views:
        if not v.registered:
            continue
        Rc, _ = cv2.Rodrigues(v.rvec)
        Rn = Rc @ R.T
        tn = s * v.tvec.reshape(3) - Rn @ t
        v.rvec = cv2.Rodrigues(Rn)[0].reshape(3)
        v.tvec = tn


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:                       # opposite: rotate 180° about any perpendicular axis
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return cv2.Rodrigues(axis * math.pi)[0]
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


# ─────────────────────────────────────────── quality analysis

@dataclass
class ImageQuality:
    view_id: int
    name: str
    registered: bool
    features: int
    observations: int
    mean_error_px: Optional[float]
    coverage: float                     # fraction of a 4×4 grid with observations


def image_quality(rec: Reconstruction) -> List[ImageQuality]:
    from pathlib import Path
    obs: Dict[int, List[Tuple[int, int]]] = {}
    for tr in rec.points:
        for vid, kp in tr.obs.items():
            obs.setdefault(vid, []).append((kp, id(tr)))
    by_id = {id(t): t for t in rec.points}
    out = []
    for v in rec.views:
        cam = rec.cameras[v.camera_id]
        lst = obs.get(v.id, []) if v.registered else []
        err = None
        cover = 0.0
        if lst:
            X = np.array([by_id[p].xyz for _, p in lst])
            px, _ = project(cam, v.rvec, v.tvec, X)
            uv = v.keypoints[[k for k, _ in lst]]
            err = float(np.mean(np.linalg.norm(px - uv, axis=1)))
            gx = np.clip((uv[:, 0] / cam.width * 4).astype(int), 0, 3)
            gy = np.clip((uv[:, 1] / cam.height * 4).astype(int), 0, 3)
            cover = len(set(zip(gx.tolist(), gy.tolist()))) / 16.0
        out.append(ImageQuality(v.id, Path(v.path).name, v.registered, len(v.keypoints), len(lst), err, cover))
    return out


def components(rec: Reconstruction) -> List[List[int]]:
    """Groups of images linked by verified matches (RealityScan 'components')."""
    parent = list(range(len(rec.views)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for p in rec.pairs:
        ra, rb = find(p.i), find(p.j)
        if ra != rb:
            parent[ra] = rb
    groups: Dict[int, List[int]] = {}
    for i in range(len(rec.views)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


# ─────────────────────────────────────────── cleanup

def clean_points(rec: Reconstruction, *, min_track: int = 2, max_error_px: Optional[float] = None,
                 outlier_k: int = 8, outlier_std: float = 2.5) -> int:
    """Remove weak points; returns how many were removed."""
    from scipy.spatial import cKDTree
    removed = 0
    for t in rec.points:
        n = sum(1 for v in t.obs if rec.views[v].registered)
        if n < min_track or (max_error_px is not None and t.error > max_error_px):
            t.xyz = None
            removed += 1
    pts = rec.points
    if outlier_k and len(pts) > outlier_k + 1:
        X = np.array([t.xyz for t in pts])
        d, _ = cKDTree(X).query(X, k=outlier_k + 1)
        md = d[:, 1:].mean(1)
        thr = md.mean() + outlier_std * md.std()
        for t, m in zip(pts, md):
            if m > thr:
                t.xyz = None
                removed += 1
    return removed


def robust_bounds(rec: Reconstruction, lo: float = 2, hi: float = 98) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([t.xyz for t in rec.points])
    return np.percentile(X, lo, axis=0), np.percentile(X, hi, axis=0)


def crop_to_region(rec: Reconstruction, bmin: np.ndarray, bmax: np.ndarray) -> int:
    """RealityScan 'reconstruction region': drop points outside the box."""
    removed = 0
    for t in rec.points:
        if np.any(t.xyz < bmin) or np.any(t.xyz > bmax):
            t.xyz = None
            removed += 1
    return removed


# ─────────────────────────────────────────── level / centre

def estimate_up(rec: Reconstruction) -> np.ndarray:
    """World 'up': photographers hold cameras upright, so −(camera y axis) on average.
    For orbits the ring of cameras is a strong cue too — use its normal, signed
    to agree with the cameras' own up."""
    reg = rec.registered_views
    ups = []
    for v in reg:
        Rc, _ = cv2.Rodrigues(v.rvec)
        ups.append(-Rc.T @ np.array([0.0, 1.0, 0.0]))   # camera −y in world
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up) + 1e-12
    C = np.array([camera_center(v) for v in reg])
    if len(C) >= 6:
        Cc = C - C.mean(0)
        _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
        n = Vt[2]
        flat = S[2] / (S[0] + 1e-12)
        if flat < 0.25 and abs(np.dot(n, up)) > 0.5:     # cameras lie roughly on a plane around the subject
            up = n if np.dot(n, up) > 0 else -n
    return up / np.linalg.norm(up)


def level_scene(rec: Reconstruction, up_axis: str = "y", center: bool = True) -> dict:
    """Rotate so the estimated up becomes +Y (Fumoca/three.js) or +Z (RealityScan/Blender);
    optionally move the subject (median of points) to the origin."""
    up = estimate_up(rec)
    target = np.array([0, 1.0, 0]) if up_axis == "y" else np.array([0, 0, 1.0])
    R = _rotation_between(up, target)
    t = np.zeros(3)
    if center and rec.points:
        X = np.array([p.xyz for p in rec.points])
        t = -np.median(X @ R.T, axis=0)
    apply_similarity(rec, 1.0, R, t)
    return {"up_before": up.tolist(), "up_axis": up_axis, "centered": bool(center)}


# ─────────────────────────────────────────── scale

def define_distance(rec: Reconstruction, view_a: int, view_b: int, metres: float) -> float:
    """Scale so the two cameras are `metres` apart. Returns the scale applied."""
    ca, cb = camera_center(rec.views[view_a]), camera_center(rec.views[view_b])
    d = float(np.linalg.norm(ca - cb))
    if d < 1e-9 or metres <= 0:
        raise ValueError("Pick two different, placed cameras and a positive distance.")
    s = metres / d
    apply_similarity(rec, s, np.eye(3), np.zeros(3))
    return s


def _gps_from_exif(path: str) -> Optional[Tuple[float, float, float]]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def dms(v):
            d, m, s = [float(x) for x in v]
            return d + m / 60 + s / 3600
        lat = dms(gps[2]) * (-1 if gps.get(1) in ("S", b"S") else 1)
        lon = dms(gps[4]) * (-1 if gps.get(3) in ("W", b"W") else 1)
        alt = float(gps.get(6, 0.0)) * (-1 if gps.get(5) in (1, b"\x01") else 1)
        return lat, lon, alt
    except Exception:  # noqa: BLE001
        return None


def _enu(lat, lon, alt, lat0, lon0, alt0):
    """Small-area east/north/up in metres (equirectangular; fine for a capture site)."""
    R = 6378137.0
    e = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * R
    return np.array([e, n, alt - alt0])


def georeference_from_gps(rec: Reconstruction, min_images: int = 3) -> dict:
    """Fit scale + rotation + offset from camera centres to photo GPS (ENU metres)."""
    pairs = []
    for v in rec.registered_views:
        g = _gps_from_exif(v.path)
        if g:
            pairs.append((v, g))
    if len(pairs) < min_images:
        raise ValueError(f"Only {len(pairs)} placed photo(s) carry GPS — need at least {min_images}.")
    lat0, lon0, alt0 = pairs[0][1]
    src = np.array([camera_center(v) for v, _ in pairs])
    dst = np.array([_enu(*g, lat0, lon0, alt0) for _, g in pairs])
    if np.linalg.norm(dst - dst.mean(0), axis=1).max() < 0.5:
        raise ValueError("The GPS positions are all within half a metre — too close together to scale from.")
    s, R, t = umeyama(src, dst)
    apply_similarity(rec, s, R, t)
    resid = np.linalg.norm(np.array([camera_center(v) for v, _ in pairs]) - dst, axis=1)
    return {"images_with_gps": len(pairs), "scale": s, "origin": [lat0, lon0, alt0],
            "rms_m": float(np.sqrt(np.mean(resid ** 2))), "max_m": float(resid.max())}
