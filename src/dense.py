"""
Dense cloud: turn an aligned capture into a point cloud that already looks
like the object (OpenMVS / RealityScan "dense point cloud" style), on the CPU
with no AI.

For every placed photo we pick its best stereo partner (shares many points,
5–25° apart), rectify the pair, run semi-global matching (OpenCV SGBM) with a
disparity range taken from the sparse points, and lift the depth to 3D. All
pairs are fused in a voxel grid that keeps only surfaces seen consistently
by at least two pairs, averaging position and colour — noise and floaters
drop out, the object stays.

The result can be exported as a PLY to look at, and added to the COLMAP
points3D so Brush starts training from a much richer, object-shaped cloud.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .sfm import Reconstruction, View, camera_center


@dataclass
class DenseCloud:
    points: np.ndarray        # (N, 3) float32
    colors: np.ndarray        # (N, 3) uint8
    support: np.ndarray       # (N,) how many stereo pairs saw it
    pairs_used: int

    def __len__(self):
        return len(self.points)


def _pose(v: View) -> Tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(v.rvec)
    return R, v.tvec.reshape(3).astype(np.float64)


def _shared_points(rec: Reconstruction) -> Dict[Tuple[int, int], List[np.ndarray]]:
    shared: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for t in rec.points:
        regs = [v for v in t.obs if rec.views[v].registered]
        for a in range(len(regs)):
            for b in range(a + 1, len(regs)):
                key = (min(regs[a], regs[b]), max(regs[a], regs[b]))
                shared.setdefault(key, []).append(t.xyz)
    return shared


def choose_pairs(rec: Reconstruction, per_image: int = 2, min_angle: float = 4.0, max_angle: float = 28.0,
                 min_shared: int = 40) -> List[Tuple[int, int]]:
    """Best stereo partners per image: many shared points and a useful baseline angle."""
    shared = _shared_points(rec)
    centers = {v.id: camera_center(v) for v in rec.registered_views}
    scored: Dict[int, List[Tuple[float, int]]] = {}
    for (a, b), pts in shared.items():
        if len(pts) < min_shared:
            continue
        P = np.array(pts)
        ra, rb = P - centers[a], P - centers[b]
        cos = np.sum(ra * rb, 1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
        ang = math.degrees(float(np.median(np.arccos(np.clip(cos, -1, 1)))))
        if not (min_angle <= ang <= max_angle):
            continue
        # prefer ~12° and many shared points
        score = len(pts) * math.exp(-((ang - 12.0) / 10.0) ** 2)
        scored.setdefault(a, []).append((score, b))
        scored.setdefault(b, []).append((score, a))
    pairs = set()
    for a, lst in scored.items():
        for _, b in sorted(lst, reverse=True)[:per_image]:
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def _stereo_pair(rec: Reconstruction, ia: int, ib: int, max_side: int, step: int):
    """Rectify + SGBM one pair. Returns world points, colours (or None)."""
    va, vb = rec.views[ia], rec.views[ib]
    ca, cb = rec.cameras[va.camera_id], rec.cameras[vb.camera_id]
    if (ca.width, ca.height) != (cb.width, cb.height):
        return None
    w, h = ca.width, ca.height
    s = min(1.0, max_side / max(w, h))
    size = (int(round(w * s)), int(round(h * s)))

    def K(c):
        return np.array([[c.f * s, 0, c.cx * s], [0, c.f * s, c.cy * s], [0, 0, 1]], np.float64)
    Ka, Kb = K(ca), K(cb)
    Da, Db = np.array([ca.k, 0, 0, 0, 0.0]), np.array([cb.k, 0, 0, 0, 0.0])
    Ra, ta = _pose(va)
    Rb, tb = _pose(vb)
    R = Rb @ Ra.T
    T = (tb - R @ ta).reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Ka, Da, Kb, Db, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        return "vertical"                          # vertical stereo: let the caller try another order/partner
    if P2[0, 3] > 0:
        return "swap"                              # partner is on the left — run the pair the other way round
    ia_img = cv2.imread(va.path, cv2.IMREAD_COLOR)
    ib_img = cv2.imread(vb.path, cv2.IMREAD_COLOR)
    if ia_img is None or ib_img is None:
        return None
    if s < 1:
        ia_img = cv2.resize(ia_img, size, interpolation=cv2.INTER_AREA)
        ib_img = cv2.resize(ib_img, size, interpolation=cv2.INTER_AREA)
    m1 = cv2.initUndistortRectifyMap(Ka, Da, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Kb, Db, R2, P2, size, cv2.CV_32FC1)
    L = cv2.remap(ia_img, m1[0], m1[1], cv2.INTER_LINEAR)
    Rimg = cv2.remap(ib_img, m2[0], m2[1], cv2.INTER_LINEAR)

    # disparity range from the sparse points this view sees
    f, B = P1[0, 0], abs(P2[0, 3] / P2[0, 0])
    pts = np.array([t.xyz for t in rec.points if ia in t.obs])
    if len(pts) < 20:
        return None
    Xc = (pts @ Ra.T + ta) @ R1.T                 # rectified camera-1 frame
    z = Xc[:, 2]
    z = z[z > 1e-6]
    if len(z) < 20:
        return None
    zlo, zhi = np.percentile(z, 2) / 1.35, np.percentile(z, 98) * 1.35
    dmin, dmax = f * B / zhi, f * B / zlo
    min_disp = int(math.floor(dmin)) - 2
    num = int(math.ceil((dmax - min_disp + 2) / 16.0)) * 16
    num = max(16, min(num, 16 * 24))
    bs = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num, blockSize=bs,
                                 P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs, disp12MaxDiff=1,
                                 uniquenessRatio=12, speckleWindowSize=80, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    gl, gr = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Rimg, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    valid = disp > (min_disp + 0.5)
    # rectification leaves black borders: ignore them
    valid &= (gl > 0) & (cv2.erode((gl > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    # low texture gives unreliable matches
    tex = cv2.Laplacian(gl, cv2.CV_32F, ksize=3)
    valid &= cv2.blur(np.abs(tex), (5, 5)) > 2.0
    xyz = cv2.reprojectImageTo3D(disp, Q)
    sub = np.zeros_like(valid); sub[::step, ::step] = True
    valid &= sub
    P = xyz[valid]
    C = L[valid][:, ::-1]                           # BGR → RGB
    zc = P[:, 2]
    keep = np.isfinite(P).all(1) & (zc > zlo) & (zc < zhi)
    P, C = P[keep], C[keep]
    # rectified cam1 → cam1 → world
    Xcam = P @ R1                                  # R1ᵀ applied to row vectors
    Xw = (Xcam - ta) @ Ra                          # Raᵀ (X − t)
    return Xw.astype(np.float32), C.astype(np.uint8)


def densify(rec: Reconstruction, *, max_side: int = 1000, per_image: int = 2, step: int = 2,
            voxel: Optional[float] = None, min_support: int = 2,
            progress: Optional[Callable[[float, str], None]] = None) -> DenseCloud:
    pairs = choose_pairs(rec, per_image=per_image)
    if not pairs:
        raise ValueError("No photo pairs with enough overlap and parallax for a dense cloud.")
    chunks, colors, pair_ids = [], [], []
    used = 0
    for n, (a, b) in enumerate(pairs):
        out = _stereo_pair(rec, a, b, max_side, step)
        if isinstance(out, str) and out == "swap":
            out = _stereo_pair(rec, b, a, max_side, step)
        if out is None or isinstance(out, str):
            continue
        P, C = out
        if len(P):
            chunks.append(P); colors.append(C); pair_ids.append(np.full(len(P), used, np.int32))
            used += 1
        if progress:
            progress((n + 1) / len(pairs), f"Dense matching {n + 1}/{len(pairs)} pairs")
    if not chunks:
        raise ValueError("Dense matching found no reliable surfaces (try more texture / overlap).")
    P = np.concatenate(chunks); C = np.concatenate(colors).astype(np.float32); pid = np.concatenate(pair_ids)

    # voxel fusion with multi-pair consistency
    sparse = np.array([t.xyz for t in rec.points])
    lo, hi = np.percentile(sparse, 2, axis=0), np.percentile(sparse, 98, axis=0)
    ext = float(np.max(hi - lo)) or 1.0
    vox = voxel or ext / 500.0
    pad = (hi - lo) * 0.3
    inside = np.all((P > lo - pad) & (P < hi + pad), axis=1)
    P, C, pid = P[inside], C[inside], pid[inside]
    keys = np.floor((P - (lo - pad)) / vox).astype(np.int64)
    dims = np.max(keys, axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    # distinct pairs per voxel
    pk = np.unique(np.stack([inv, pid], 1), axis=0)
    support = np.bincount(pk[:, 0], minlength=len(uniq))
    sumP = np.zeros((len(uniq), 3)); sumC = np.zeros((len(uniq), 3))
    np.add.at(sumP, inv, P); np.add.at(sumC, inv, C)
    need = min_support if used >= 3 else 1
    keep = support >= need
    pts = (sumP[keep] / counts[keep, None]).astype(np.float32)
    cols = np.clip(sumC[keep] / counts[keep, None], 0, 255).astype(np.uint8)
    return DenseCloud(points=pts, colors=cols, support=support[keep].astype(np.int32), pairs_used=used)


def write_ply(path: str, cloud: DenseCloud) -> None:
    n = len(cloud)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    rec = np.zeros(n, dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"] = cloud.points; rec["c"] = cloud.colors
    with open(path, "wb") as f:
        f.write(header); f.write(rec.tobytes())
