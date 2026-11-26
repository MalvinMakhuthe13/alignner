"""
Alignner incremental Structure-from-Motion (CPU only, no AI).

Pipeline (the same stages COLMAP / RealityScan use):

  1. features     SIFT (default) or ORB, computed on a down-scaled copy and
                  mapped back to full-resolution pixel coordinates
  2. intrinsics   per-image focal from EXIF (35 mm equivalent) when present,
                  otherwise 1.2 × the longer side; images that share size and
                  focal share one camera
  3. matching     ratio test + mutual check, then two-view geometric
                  verification (MAGSAC fundamental + essential matrix)
  4. tracks       verified matches are joined across all views into feature
                  tracks (union-find); tracks that hit one image twice are
                  dropped
  5. init pair    many inliers, low homography ratio and a real triangulation
                  angle, so the first 3D points are well conditioned
  6. register     every next image is placed with PnP (RANSAC + refinement)
                  against the points already in the model
  7. triangulate  multi-view DLT for tracks seen by ≥ 2 registered images,
                  with cheirality, angle and reprojection checks
  8. bundle adj.  sparse Levenberg–Marquardt (scipy) over poses, focal,
                  radial distortion and points, Huber loss; local after each
                  registration, global periodically and at the end, followed
                  by outlier filtering and re-triangulation

Output is a real COLMAP model (cameras / images / points3D, text and
binary) with full 2D↔3D tracks, which Brush, gsplat, Nerfstudio and COLMAP's
own tools read directly.
"""
from __future__ import annotations

import logging
import math
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.sparse import coo_matrix

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────── data types


@dataclass
class Camera:
    id: int
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k: float = 0.0                     # SIMPLE_RADIAL k1

    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class View:
    id: int
    path: str
    camera_id: int
    keypoints: np.ndarray               # (N, 2) full-resolution pixels
    descriptors: np.ndarray
    colors: np.ndarray                  # (N, 3) uint8 RGB sampled at keypoints
    rvec: Optional[np.ndarray] = None   # world → camera (Rodrigues)
    tvec: Optional[np.ndarray] = None

    @property
    def registered(self) -> bool:
        return self.rvec is not None


@dataclass
class TwoViewGeometry:
    i: int
    j: int
    matches: np.ndarray                 # (M, 2) keypoint indices (in i, in j)
    inliers: int
    h_ratio: float = 1.0                # homography inliers / essential inliers


@dataclass
class Track:
    obs: Dict[int, int]                 # view id → keypoint index
    xyz: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    error: float = 0.0


@dataclass
class Reconstruction:
    cameras: Dict[int, Camera]
    views: List[View]
    tracks: List[Track]
    pairs: List[TwoViewGeometry] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    # convenience
    @property
    def registered_views(self) -> List[View]:
        return [v for v in self.views if v.registered]

    @property
    def points(self) -> List[Track]:
        return [t for t in self.tracks if t.xyz is not None]

    def stats(self) -> dict:
        pts = self.points
        errs = [t.error for t in pts]
        obs = [len([v for v in t.obs if self.views[v].registered]) for t in pts]
        return {
            "images": len(self.views),
            "registered": len(self.registered_views),
            "points": len(pts),
            "mean_reprojection_error_px": float(np.mean(errs)) if errs else None,
            "mean_track_length": float(np.mean(obs)) if obs else None,
            "verified_pairs": len(self.pairs),
        }


# ─────────────────────────────────────────────────────────────── geometry


def project(cam: Camera, rvec, tvec, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points; returns pixels (N,2) and camera-space depth (N,)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    Xc = X @ R.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    u, v = Xc[:, 0] / zs, Xc[:, 1] / zs
    d = 1.0 + cam.k * (u * u + v * v)
    return np.stack([cam.f * u * d + cam.cx, cam.f * v * d + cam.cy], axis=1), z


def undistort_points(cam: Camera, px: np.ndarray) -> np.ndarray:
    """Pixels → normalised image coordinates (removes SIMPLE_RADIAL)."""
    u = (px[:, 0] - cam.cx) / cam.f
    v = (px[:, 1] - cam.cy) / cam.f
    if abs(cam.k) < 1e-12:
        return np.stack([u, v], axis=1)
    uu, vv = u.copy(), v.copy()
    for _ in range(8):  # fixed-point inversion
        d = 1.0 + cam.k * (uu * uu + vv * vv)
        uu, vv = u / d, v / d
    return np.stack([uu, vv], axis=1)


def camera_center(view: View) -> np.ndarray:
    R, _ = cv2.Rodrigues(view.rvec)
    return -R.T @ view.tvec.reshape(3)


def triangulate_track(Ps: List[np.ndarray], xs: List[np.ndarray]) -> np.ndarray:
    """Multi-view linear (DLT) triangulation from normalised coordinates."""
    A = []
    for P, x in zip(Ps, xs):
        A.append(x[0] * P[2] - P[0])
        A.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / (X[3] if abs(X[3]) > 1e-12 else 1e-12)


# ─────────────────────────────────────────────────────────────── EXIF


def exif_focal_px(path: str, width: int, height: int) -> Optional[float]:
    """Focal length in pixels from EXIF, if the photo carries it."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            data = {**dict(exif), **dict(sub)}
        f35 = data.get(0xA405)                    # FocalLengthIn35mmFilm
        if f35:
            return float(f35) / 36.0 * max(width, height)
        f_mm = data.get(0x920A)                   # FocalLength
        fpx_res = data.get(0xA20E)                # FocalPlaneXResolution
        unit = data.get(0xA210, 2)                # 2 = inch, 3 = cm
        if f_mm and fpx_res:
            per_mm = float(fpx_res) / (25.4 if int(unit) == 2 else 10.0)
            return float(f_mm) * per_mm
    except Exception:  # noqa: BLE001 - EXIF is best effort
        return None
    return None


# ─────────────────────────────────────────────────────────────── the pipeline


class IncrementalSfM:
    def __init__(
        self,
        detector: str = "sift",
        max_features: int = 8000,
        feature_max_side: int = 2400,
        ratio: float = 0.8,
        min_inliers: int = 30,
        max_reproj_px: float = 4.0,
        min_tri_angle_deg: float = 1.5,
        exhaustive_limit: int = 60,
        sequential_window: int = 12,
        focal_px: Optional[float] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ):
        self.detector_name = detector.lower()
        self.max_features = max_features
        self.feature_max_side = feature_max_side
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.max_reproj_px = max_reproj_px
        self.min_tri_angle = math.radians(min_tri_angle_deg)
        self.exhaustive_limit = exhaustive_limit
        self.sequential_window = sequential_window
        self.focal_override = focal_px
        self._progress = progress
        self.log: List[str] = []
        self.ba_x_scale = "jac"
        self.ba_tr_options = {}

    # -- helpers ------------------------------------------------------------
    def _p(self, frac: float, msg: str):
        logger.info(msg)
        if self._progress:
            self._progress(max(0.0, min(1.0, frac)), msg)

    def _note(self, msg: str):
        self.log.append(msg)
        logger.info(msg)

    def _detector(self):
        if self.detector_name == "orb":
            return cv2.ORB_create(nfeatures=self.max_features * 3, scaleFactor=1.2, nlevels=8)
        # detect generously (incl. low-contrast detail), then spread evenly below
        return cv2.SIFT_create(nfeatures=self.max_features * 3, contrastThreshold=0.03)

    def _spread(self, kps, desc, w: int, h: int):
        """Keep `max_features`, spread evenly over the photo: each cell of a grid
        gets an equal share (half the budget) of its strongest features, the rest go to the
        strongest overall. Plain "strongest N" clusters on busy texture and
        leaves plain areas empty — uneven, gappy sparse clouds."""
        n = len(kps)
        if n <= self.max_features:
            return kps, desc
        gx, gy = 16, max(4, round(16 * h / max(1, w)))
        resp = np.array([k.response for k in kps]); pts = np.array([k.pt for k in kps])
        cell = (np.minimum(gy - 1, (pts[:, 1] * gy / h).astype(int)) * gx + np.minimum(gx - 1, (pts[:, 0] * gx / w).astype(int)))
        quota = max(1, int(self.max_features * 0.5) // (gx * gy))
        order = np.lexsort((-resp, cell))                  # by cell, strongest first
        rank = np.empty(n, int); start = 0
        sc = cell[order]
        for i in range(n):
            if i == 0 or sc[i] != sc[i - 1]: start = i
            rank[order[i]] = i - start
        keep = rank < quota
        rest = np.where(~keep)[0]
        extra = self.max_features - int(keep.sum())
        if extra > 0 and len(rest):
            keep[rest[np.argsort(-resp[rest])[:extra]]] = True
        idx = np.where(keep)[0]
        return [kps[i] for i in idx], desc[idx]

    # -- 1+2 features & cameras ---------------------------------------------
    def extract(self, paths: List[str]) -> Tuple[Dict[int, Camera], List[View]]:
        det = self._detector()
        cameras: Dict[Tuple, Camera] = {}
        views: List[View] = []
        for n, path in enumerate(paths):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                self._note(f"skipped unreadable image {Path(path).name}")
                continue
            h, w = img.shape[:2]
            s = min(1.0, self.feature_max_side / max(h, w))
            small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            kps, desc = det.detectAndCompute(gray, None)
            if desc is not None and len(kps):
                kps, desc = self._spread(kps, desc, gray.shape[1], gray.shape[0])
            if desc is None or len(kps) < 50:
                self._note(f"skipped {Path(path).name}: only {0 if kps is None else len(kps)} features")
                continue
            if self.detector_name != "orb":
                # RootSIFT: noticeably better matching for the same cost
                desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
                desc = np.sqrt(desc).astype(np.float32)
            pts = np.array([kp.pt for kp in kps], dtype=np.float64) / s
            ix = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
            iy = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
            colors = img[iy, ix][:, ::-1].copy()  # BGR → RGB

            f = self.focal_override or exif_focal_px(path, w, h) or 1.2 * max(w, h)
            key = (w, h, round(f, 1))
            if key not in cameras:
                cameras[key] = Camera(id=len(cameras) + 1, width=w, height=h, f=float(f), cx=w / 2.0, cy=h / 2.0)
            views.append(View(id=len(views), path=str(path), camera_id=cameras[key].id,
                              keypoints=pts, descriptors=desc, colors=colors))
            self._p(0.15 * (n + 1) / len(paths), f"Features {n + 1}/{len(paths)}: {len(pts)} in {Path(path).name}")
        cams = {c.id: c for c in cameras.values()}
        for c in cams.values():
            self._note(f"camera {c.id}: {c.width}×{c.height}, focal {c.f:.0f}px")
        return cams, views

    # -- 3 matching ----------------------------------------------------------
    def _matcher(self):
        if self.detector_name == "orb":
            return cv2.BFMatcher(cv2.NORM_HAMMING)
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))

    def _ratio_matches(self, matcher, d1, d2) -> np.ndarray:
        """Lowe ratio test one way, then a mutual-nearest check on the survivors only."""
        fwd = {}
        for m in matcher.knnMatch(d1, d2, k=2):
            if len(m) == 2 and m[0].distance < self.ratio * m[1].distance:
                fwd[m[0].queryIdx] = m[0].trainIdx
        if not fwd:
            return np.zeros((0, 2), dtype=np.int64)
        q = np.fromiter(fwd.keys(), dtype=np.int64)
        t = np.fromiter(fwd.values(), dtype=np.int64)
        back = matcher.knnMatch(d2[t], d1, k=1)
        best = np.array([b[0].trainIdx if b else -1 for b in back], dtype=np.int64)
        keep = best == q
        return np.stack([q[keep], t[keep]], axis=1)

    def candidate_pairs(self, n: int, views: Optional[List[View]] = None) -> List[Tuple[int, int]]:
        """Exhaustive for small sets; otherwise sequential neighbours + visual retrieval."""
        if n <= self.exhaustive_limit:
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        pairs = set()
        for i in range(n):
            for d in range(1, self.sequential_window + 1):
                if i + d < n:
                    pairs.add((i, i + d))
        if views is not None:
            for i, j in self.retrieval_pairs(views, k=self.retrieval_k):
                pairs.add((min(i, j), max(i, j)))
        return sorted(pairs)

    retrieval_k = 20

    def retrieval_pairs(self, views: List[View], k: int = 20, words: int = 512) -> List[Tuple[int, int]]:
        """
        Bag-of-visual-words image retrieval (TF-IDF, cosine) so photos taken
        out of order, revisits and flyover strips still get matched.
        """
        rng = np.random.default_rng(0)
        sample = np.concatenate([v.descriptors[rng.choice(len(v.descriptors), min(300, len(v.descriptors)), replace=False)]
                                 for v in views]).astype(np.float32)
        words = int(min(words, max(16, len(sample) // 20)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
        _, _, centers = cv2.kmeans(sample, words, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=16))
        hist = np.zeros((len(views), words), np.float32)
        for n, v in enumerate(views):
            ms = index.match(v.descriptors.astype(np.float32), centers)
            np.add.at(hist[n], [m.trainIdx for m in ms], 1)
        df = (hist > 0).sum(0)
        idf = np.log(len(views) / np.maximum(df, 1))
        tf = hist / np.maximum(hist.sum(1, keepdims=True), 1)
        vec = tf * idf
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        sim = vec @ vec.T
        np.fill_diagonal(sim, -1)
        out = []
        for i in range(len(views)):
            for j in np.argsort(-sim[i])[:k]:
                out.append((i, int(j)))
        return out

    def match(self, cameras: Dict[int, Camera], views: List[View]) -> List[TwoViewGeometry]:
        matcher = self._matcher()
        cand = self.candidate_pairs(len(views), views)
        out: List[TwoViewGeometry] = []
        for k, (i, j) in enumerate(cand):
            vi, vj = views[i], views[j]
            m = self._ratio_matches(matcher, vi.descriptors, vj.descriptors)
            if len(m) >= self.min_inliers:
                p1, p2 = vi.keypoints[m[:, 0]], vj.keypoints[m[:, 1]]
                F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1.5, 0.9999, 10000)
                if F is not None and mask is not None and F.shape == (3, 3):
                    keep = mask.ravel().astype(bool)
                    if keep.sum() >= self.min_inliers:
                        m, p1, p2 = m[keep], p1[keep], p2[keep]
                        ci, cj = cameras[vi.camera_id], cameras[vj.camera_id]
                        n1, n2 = undistort_points(ci, p1), undistort_points(cj, p2)
                        thr = 2.0 / ((ci.f + cj.f) / 2)
                        E, emask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.9999, thr)
                        if E is not None and emask is not None and E.shape == (3, 3):
                            ekeep = emask.ravel().astype(bool)
                            ne = int(ekeep.sum())
                            if ne >= self.min_inliers:
                                H, hmask = cv2.findHomography(p1[ekeep], p2[ekeep], cv2.RANSAC, 4.0)
                                hr = float(hmask.sum()) / ne if hmask is not None else 1.0
                                out.append(TwoViewGeometry(i, j, m[ekeep], ne, hr))
            if k % 4 == 0 or k == len(cand) - 1:
                self._p(0.15 + 0.35 * (k + 1) / len(cand), f"Matching {k + 1}/{len(cand)} pairs · {len(out)} verified")
        self._note(f"{len(out)} of {len(cand)} candidate pairs verified")
        return out

    # -- 4 tracks ------------------------------------------------------------
    @staticmethod
    def build_tracks(views: List[View], pairs: List[TwoViewGeometry]) -> List[Track]:
        offsets = np.cumsum([0] + [len(v.keypoints) for v in views])
        parent = np.arange(offsets[-1])

        def find(a):
            root = a
            while parent[root] != root:
                root = parent[root]
            while parent[a] != root:
                parent[a], a = root, parent[a]
            return root

        for p in pairs:
            for a, b in p.matches:
                ra, rb = find(offsets[p.i] + a), find(offsets[p.j] + b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

        groups: Dict[int, List[int]] = {}
        used = set()
        for p in pairs:
            for a, b in p.matches:
                for g in (offsets[p.i] + a, offsets[p.j] + b):
                    if g not in used:
                        used.add(g)
                        groups.setdefault(find(g), []).append(int(g))
        tracks: List[Track] = []
        for members in groups.values():
            obs: Dict[int, int] = {}
            ok = True
            for g in members:
                vid = int(np.searchsorted(offsets, g, side="right") - 1)
                if vid in obs:          # one image seen twice → ambiguous track
                    ok = False
                    break
                obs[vid] = int(g - offsets[vid])
            if ok and len(obs) >= 2:
                tracks.append(Track(obs=obs))
        return tracks

    # -- 5 initial pair --------------------------------------------------------
    def initialize(self, cams, views, pairs, tracks) -> bool:
        ranked = sorted(pairs, key=lambda p: p.inliers * (1.0 - min(0.95, p.h_ratio)), reverse=True)
        for p in ranked[:30]:
            vi, vj = views[p.i], views[p.j]
            ci, cj = cams[vi.camera_id], cams[vj.camera_id]
            n1 = undistort_points(ci, vi.keypoints[p.matches[:, 0]])
            n2 = undistort_points(cj, vj.keypoints[p.matches[:, 1]])
            E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.9999, 1.5 / ci.f)
            if E is None or E.shape != (3, 3):
                continue
            good, R, t, pmask = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask)
            if good < self.min_inliers:
                continue
            vi.rvec, vi.tvec = np.zeros(3), np.zeros(3)
            vj.rvec, vj.tvec = cv2.Rodrigues(R)[0].reshape(3), t.reshape(3)
            n_pts = self.triangulate(cams, views, tracks)
            angle = self._median_angle(views, tracks)
            if n_pts >= max(50, self.min_inliers) and angle > math.radians(2.0):
                self._note(f"initial pair {Path(vi.path).name} ↔ {Path(vj.path).name}: "
                           f"{n_pts} points, median angle {math.degrees(angle):.1f}°")
                self._gauge = (vi.id, vj.id)
                self.bundle_adjust(cams, views, tracks, refine_intrinsics=False)
                return True
            # reject and try the next candidate
            vi.rvec = vi.tvec = vj.rvec = vj.tvec = None
            for t_ in tracks:
                t_.xyz = None
        return False

    @staticmethod
    def _median_angle(views, tracks) -> float:
        angs = []
        for t in tracks:
            if t.xyz is None:
                continue
            reg = [v for v in t.obs if views[v].registered]
            if len(reg) < 2:
                continue
            a, b = camera_center(views[reg[0]]) - t.xyz, camera_center(views[reg[1]]) - t.xyz
            c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            angs.append(math.acos(max(-1.0, min(1.0, c))))
        return float(np.median(angs)) if angs else 0.0

    # -- 6 registration ------------------------------------------------------
    def register_next(self, cams, views, tracks) -> Optional[int]:
        # 2D–3D correspondences per unregistered image
        corr: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for t in tracks:
            if t.xyz is None:
                continue
            for vid, kp in t.obs.items():
                if not views[vid].registered:
                    corr.setdefault(vid, []).append((kp, t.xyz))
        for vid, items in sorted(corr.items(), key=lambda kv: -len(kv[1])):
            if len(items) < max(12, self.min_inliers // 2):
                return None
            v, cam = views[vid], cams[views[vid].camera_id]
            obj = np.array([x for _, x in items], dtype=np.float64)
            img = undistort_points(cam, v.keypoints[[k for k, _ in items]])
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, np.eye(3), None, iterationsCount=2000,
                reprojectionError=self.max_reproj_px / cam.f, confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or inl is None or len(inl) < max(12, self.min_inliers // 2):
                continue
            inl = inl.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], np.eye(3), None, rvec, tvec)
            v.rvec, v.tvec = rvec.reshape(3), tvec.reshape(3)
            return vid
        return None

    # -- 7 triangulation -----------------------------------------------------
    def triangulate(self, cams, views, tracks, only_new: bool = True) -> int:
        P = {}
        for v in views:
            if v.registered:
                R, _ = cv2.Rodrigues(v.rvec)
                P[v.id] = np.hstack([R, v.tvec.reshape(3, 1)])
        made = 0
        for t in tracks:
            if only_new and t.xyz is not None:
                continue
            reg = [vid for vid in t.obs if vid in P]
            if len(reg) < 2:
                continue
            xs = [undistort_points(cams[views[vid].camera_id], views[vid].keypoints[[t.obs[vid]]])[0] for vid in reg]
            X = triangulate_track([P[vid] for vid in reg], xs)
            if not np.all(np.isfinite(X)):
                continue
            # cheirality, reprojection and angle checks
            good = []
            for vid in reg:
                cam, v = cams[views[vid].camera_id], views[vid]
                px, z = project(cam, v.rvec, v.tvec, X[None])
                if z[0] <= 0:
                    continue
                if np.linalg.norm(px[0] - v.keypoints[t.obs[vid]]) <= self.max_reproj_px:
                    good.append(vid)
            if len(good) < 2:
                continue
            centers = [camera_center(views[vid]) for vid in good]
            best = 0.0
            for a in range(len(centers)):
                for b in range(a + 1, len(centers)):
                    r1, r2 = centers[a] - X, centers[b] - X
                    c = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-12)
                    best = max(best, math.acos(max(-1.0, min(1.0, c))))
            if best < self.min_tri_angle:
                continue
            t.xyz = X
            made += 1
        return made

    # -- 8 bundle adjustment ---------------------------------------------------
    def bundle_adjust(self, cams, views, tracks, *, view_ids: Optional[List[int]] = None,
                      fix_first: Optional[List[int]] = None, fix_scale_view: Optional[int] = None,
                      refine_intrinsics: bool = True, max_nfev: int = 60) -> float:
        reg = [v for v in views if v.registered]
        if view_ids is not None:
            free_views = [views[i] for i in view_ids if views[i].registered]
        else:
            free_views = reg
        gauge = getattr(self, "_gauge", None)
        if fix_first is None and gauge is not None:
            # gauge: first camera of the initial pair fixed, distance to the second held
            fix_first, fix_scale_view = [gauge[0]], gauge[1]
        fixed = set(fix_first or [reg[0].id])
        free_views = [v for v in free_views if v.id not in fixed]
        free_set = {v.id for v in free_views}
        pts = [t for t in tracks if t.xyz is not None and (view_ids is None or any(v in free_set for v in t.obs))]
        if not pts:
            return 0.0
        cam_ids = sorted({views[v].camera_id for t in pts for v in t.obs if views[v].registered}) if refine_intrinsics else []

        vidx = {v.id: k for k, v in enumerate(free_views)}
        cidx = {c: k for k, c in enumerate(cam_ids)}
        nv, nc, npt = len(free_views), len(cam_ids), len(pts)
        o_c, o_p = 6 * nv, 6 * nv + 2 * nc

        rows_v, rows_c, rows_p, obs_px, obs_view, obs_cam = [], [], [], [], [], []
        for pi, t in enumerate(pts):
            for vid, kp in t.obs.items():
                v = views[vid]
                if not v.registered:
                    continue
                rows_v.append(vidx.get(vid, -1))
                rows_c.append(cidx.get(v.camera_id, -1))
                rows_p.append(pi)
                obs_px.append(v.keypoints[kp])
                obs_view.append(vid)
                obs_cam.append(v.camera_id)
        rows_v, rows_c, rows_p = map(np.array, (rows_v, rows_c, rows_p))
        obs_px = np.array(obs_px)
        obs_view, obs_cam = np.array(obs_view), np.array(obs_cam)
        m = len(rows_p)

        x0 = np.concatenate([
            np.concatenate([np.r_[v.rvec, v.tvec] for v in free_views]) if nv else np.zeros(0),
            np.concatenate([[cams[c].f, cams[c].k] for c in cam_ids]) if nc else np.zeros(0),
            np.concatenate([t.xyz for t in pts]),
        ])
        fixed_pose = {vid: (views[vid].rvec, views[vid].tvec) for vid in set(obs_view) if vid not in vidx}
        base_cam = {c: (cams[c].f, cams[c].k) for c in set(obs_cam)}
        scale_ref = None
        if fix_scale_view is not None and fix_scale_view in vidx:
            scale_ref = (vidx[fix_scale_view], float(np.linalg.norm(views[fix_scale_view].tvec)))

        poses = x0[:o_c].reshape(nv, 6).copy()
        intr = x0[o_c:o_p].reshape(nc, 2).copy()
        X = x0[o_p:].reshape(npt, 3).copy()
        sel_of = {int(vid): np.nonzero(obs_view == vid)[0] for vid in np.unique(obs_view)}
        n_cam = 6 * nv + 2 * nc
        delta = 2.0                                        # Huber threshold (px)

        # camera-parameter column indices per observation (−1 = not optimised)
        cam_cols = np.full((m, 8), -1, dtype=np.int64)
        okv, okc = rows_v >= 0, rows_c >= 0
        cam_cols[okv, :6] = 6 * rows_v[okv, None] + np.arange(6)
        cam_cols[okc, 6:] = 6 * nv + 2 * rows_c[okc, None] + np.arange(2)

        def evaluate(poses, intr, X, with_jac):
            res = np.empty((m, 2))
            Jc = np.zeros((m, 2, 8)) if with_jac else None
            Jp = np.zeros((m, 2, 3)) if with_jac else None
            for vid, sel in sel_of.items():
                k = vidx.get(vid)
                rvec, tvec = (poses[k, :3], poses[k, 3:]) if k is not None else fixed_pose[vid]
                c = int(obs_cam[sel[0]])
                ci = cidx.get(c)
                f, kk = (intr[ci, 0], intr[ci, 1]) if ci is not None else base_cam[c]
                cam = cams[c]
                Km = np.array([[f, 0, cam.cx], [0, f, cam.cy], [0, 0, 1]])
                rvec = np.asarray(rvec, np.float64); tvec = np.asarray(tvec, np.float64)
                px, J = cv2.projectPoints(X[rows_p[sel]], rvec, tvec, Km, np.array([kk, 0, 0, 0, 0.0]))
                res[sel] = px.reshape(-1, 2) - obs_px[sel]
                if with_jac:
                    J = J.reshape(len(sel), 2, -1)
                    Jc[sel, :, :6] = J[:, :, :6]
                    Jc[sel, :, 6] = J[:, :, 6] + J[:, :, 7]
                    Jc[sel, :, 7] = J[:, :, 10]
                    R, _ = cv2.Rodrigues(rvec)
                    Jp[sel] = J[:, :, 3:6] @ R
            return res, Jc, Jp

        def robust(res):
            n = np.linalg.norm(res, axis=1)
            w = np.where(n <= delta, 1.0, delta / np.maximum(n, 1e-12))
            cost = np.where(n <= delta, n * n, 2 * delta * n - delta * delta).sum()
            return w, cost

        def scale_term(poses):
            if scale_ref is None:
                return 0.0, None, None
            k, s0 = scale_ref
            tv = poses[k, 3:]; nrm = np.linalg.norm(tv) + 1e-12
            r = 100.0 * (nrm - s0)
            jrow = np.zeros(n_cam); jrow[6 * k + 3: 6 * k + 6] = 100.0 * tv / nrm
            return r, jrow, r * r

        res, Jc, Jp = evaluate(poses, intr, X, True)
        w, cost = robust(res)
        sr, sj, sc = scale_term(poses)
        cost += sc or 0.0
        before = float(np.sqrt(np.mean(res ** 2)))
        lam = 1e-3
        iters = max(3, max_nfev // 2)
        for _ in range(iters):
            # ---- normal equations with the Schur complement on points
            wJc = Jc * w[:, None, None]
            wJp = Jp * w[:, None, None]
            Hpp = np.einsum("oai,oaj->oij", wJp, Jp)
            Hpp_full = np.zeros((npt, 3, 3)); np.add.at(Hpp_full, rows_p, Hpp)
            gp = np.zeros((npt, 3)); np.add.at(gp, rows_p, np.einsum("oai,oa->oi", wJp, res))
            Hcc_o = np.einsum("oai,oaj->oij", wJc, Jc)                         # (m,8,8)
            gc_o = np.einsum("oai,oa->oi", wJc, res)                           # (m,8)
            Hcp_o = np.einsum("oai,oaj->oij", wJc, Jp)                         # (m,8,3)
            Hcc = np.zeros((n_cam + 1, n_cam + 1))
            cols = np.where(cam_cols < 0, n_cam, cam_cols)
            np.add.at(Hcc, (cols[:, :, None], cols[:, None, :]), Hcc_o)
            Hcc = Hcc[:n_cam, :n_cam]
            gc = np.zeros(n_cam + 1); np.add.at(gc, cols, gc_o); gc = gc[:n_cam]
            if sj is not None:
                Hcc += np.outer(sj, sj); gc += sj * sr
            # damping
            Hpp_d = Hpp_full + lam * (np.einsum("pii->pi", Hpp_full)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3))
            Hpp_inv = np.linalg.inv(Hpp_d)
            Hcc_d = Hcc + lam * np.diag(np.diag(Hcc) + 1e-9)
            if n_cam:
                # Hcp (n_cam × 3npt) as sparse, Schur S = Hcc − Hcp Hpp⁻¹ Hpc
                ri = np.repeat(cols, 3, axis=1).ravel()
                ci_ = (3 * rows_p[:, None, None] + np.arange(3)[None, None, :]).repeat(8, axis=1).reshape(m, 24).ravel()
                vals = Hcp_o.reshape(m, 24).ravel()
                keep = ri < n_cam
                Hcp = coo_matrix((vals[keep], (ri[keep], ci_[keep])), shape=(n_cam, 3 * npt)).tocsr()
                bi = np.repeat(np.arange(npt), 9) * 3 + np.tile(np.repeat(np.arange(3), 3), npt)
                bj = np.repeat(np.arange(npt), 9) * 3 + np.tile(np.tile(np.arange(3), 3), npt)
                Hinv = coo_matrix((Hpp_inv.ravel(), (bi, bj)), shape=(3 * npt, 3 * npt)).tocsr()
                T = Hcp @ Hinv
                S = Hcc_d - (T @ Hcp.T).toarray()
                rhs = -gc + T @ gp.ravel()
                try:
                    dc = np.linalg.solve(S, rhs)
                except np.linalg.LinAlgError:
                    dc = np.linalg.lstsq(S, rhs, rcond=None)[0]
                dp = -(Hpp_inv @ (gp + (Hcp.T @ dc).reshape(npt, 3))[:, :, None])[:, :, 0]
            else:
                dc = np.zeros(0)
                dp = -(Hpp_inv @ gp[:, :, None])[:, :, 0]
            # ---- try the step
            np_poses = poses + dc[:6 * nv].reshape(nv, 6)
            np_intr = intr + dc[6 * nv:].reshape(nc, 2)
            np_X = X + dp
            res_n, _, _ = evaluate(np_poses, np_intr, np_X, False)
            w_n, cost_n = robust(res_n)
            sr_n, _, sc_n = scale_term(np_poses)
            cost_n += sc_n or 0.0
            if cost_n < cost:
                rel = (cost - cost_n) / max(cost, 1e-12)
                poses, intr, X, cost = np_poses, np_intr, np_X, cost_n
                lam = max(lam / 3, 1e-9)
                res, Jc, Jp = evaluate(poses, intr, X, True)
                w, _ = robust(res)
                sr, sj, _ = scale_term(poses)
                if rel < 1e-6:
                    break
            else:
                lam *= 5
                if lam > 1e8:
                    break

        for k, v in enumerate(free_views):
            v.rvec, v.tvec = poses[k, :3].copy(), poses[k, 3:].copy()
        for k, c in enumerate(cam_ids):
            f, kk = intr[k]
            if 0.3 * cams[c].f < f < 3 * cams[c].f and abs(kk) < 0.5:   # reject runaway intrinsics
                cams[c].f, cams[c].k = float(f), float(kk)
            else:
                logger.info(f"BA: rejected intrinsics f={f:.1f} k={kk:.4f}")
        for pi, t in enumerate(pts):
            t.xyz = X[pi].copy()
        after = float(np.sqrt(np.mean(res ** 2)))
        logger.info(f"BA: {nv} views, {npt} points, {nc} cams, rms {before:.2f} → {after:.2f}px f={[round(cams[c].f) for c in cam_ids]} k={[round(cams[c].k,4) for c in cam_ids]}")
        return after

    def filter_points(self, cams, views, tracks) -> int:
        """Drop observations that disagree with their point, then points left with < 2 views."""
        removed = 0
        for t in tracks:
            if t.xyz is None:
                continue
            errs = []
            for vid in list(t.obs):
                v = views[vid]
                if not v.registered:
                    continue
                px, z = project(cams[v.camera_id], v.rvec, v.tvec, t.xyz[None])
                e = float(np.linalg.norm(px[0] - v.keypoints[t.obs[vid]]))
                if z[0] <= 0 or e > self.max_reproj_px:
                    del t.obs[vid]          # this image no longer claims the point
                else:
                    errs.append(e)
            if len(errs) < 2:
                t.xyz = None
                removed += 1
            else:
                t.error = float(np.mean(errs))
        return removed

    def colorize(self, views, tracks):
        for t in tracks:
            if t.xyz is None:
                continue
            cols = [views[v].colors[k] for v, k in t.obs.items() if views[v].registered]
            t.color = np.median(np.array(cols), axis=0).astype(np.uint8)

    # -- run -------------------------------------------------------------------
    def run(self, paths: List[str]) -> Reconstruction:
        self.log = []
        self._gauge = None
        cams, views = self.extract(paths)
        if len(views) < 2:
            raise ValueError("Need at least two readable images with features.")
        pairs = self.match(cams, views)
        tracks = self.build_tracks(views, pairs)
        self._note(f"{len(tracks)} feature tracks across {len(views)} images")
        if not pairs or not self.initialize(cams, views, pairs, tracks):
            raise ValueError("Could not find a pair of photos with enough overlap and parallax to start. "
                             "Capture with more overlap (60–80 %) and move between shots instead of rotating in place.")
        since_global = 0
        n_total = len(views)
        while True:
            vid = self.register_next(cams, views, tracks)
            if vid is None:
                break
            self.triangulate(cams, views, tracks)
            neigh = self._neighbours(views, tracks, vid, 8)
            self.bundle_adjust(cams, views, tracks, view_ids=[vid] + neigh, refine_intrinsics=False, max_nfev=25)
            since_global += 1
            nreg = len([v for v in views if v.registered])
            if since_global >= max(5, nreg // 5):
                self.bundle_adjust(cams, views, tracks, max_nfev=40)
                self.filter_points(cams, views, tracks)
                self.triangulate(cams, views, tracks)
                since_global = 0
            self._p(0.5 + 0.45 * nreg / n_total, f"Registered {nreg}/{n_total} images")
        # final refinement
        for _ in range(2):
            self.bundle_adjust(cams, views, tracks, max_nfev=80)
            self.filter_points(cams, views, tracks)
            self.triangulate(cams, views, tracks)
        self.filter_points(cams, views, tracks)
        self.colorize(views, tracks)
        rec = Reconstruction(cameras=cams, views=views, tracks=tracks, pairs=pairs, log=self.log)
        st = rec.stats()
        unreg = [Path(v.path).name for v in views if not v.registered]
        if unreg:
            self._note(f"{len(unreg)} image(s) could not be placed: {', '.join(unreg[:10])}{'…' if len(unreg) > 10 else ''}")
        self._p(1.0, f"Done: {st['registered']}/{st['images']} images, {st['points']} points, "
                     f"{st['mean_reprojection_error_px'] or 0:.2f}px mean error")
        return rec

    @staticmethod
    def _neighbours(views, tracks, vid, k) -> List[int]:
        count: Dict[int, int] = {}
        for t in tracks:
            if t.xyz is not None and vid in t.obs:
                for o in t.obs:
                    if o != vid and views[o].registered:
                        count[o] = count.get(o, 0) + 1
        return [o for o, _ in sorted(count.items(), key=lambda kv: -kv[1])[:k]]


# ─────────────────────────────────────────────────────────────── COLMAP export


def _qvec(rvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = math.copysign(x, R[2, 1] - R[1, 2])
    y = math.copysign(y, R[0, 2] - R[2, 0])
    z = math.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def export_colmap(rec: Reconstruction, out_dir: str, *, undistort_threshold: float = 0.005,
                  copy_images: bool = True, extra_points=None) -> dict:
    """
    Write a COLMAP model Brush / gsplat / Nerfstudio can train on:

        out_dir/images/…            original photos (byte-for-byte), or
                                    undistorted copies when lens distortion
                                    is significant
        out_dir/sparse/0/cameras.{bin,txt}
        out_dir/sparse/0/images.{bin,txt}
        out_dir/sparse/0/points3D.{bin,txt}

    Cameras are written as PINHOLE (what splat trainers expect). If the
    estimated radial distortion is non-negligible the images are undistorted
    so that PINHOLE is exact.

    extra_points: optional (points Nx3, colors Nx3 uint8), e.g. a dense
    cloud, appended to points3D without tracks so trainers start from it.
    """
    out = Path(out_dir)
    img_dir = out / "images"
    sp = out / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    sp.mkdir(parents=True, exist_ok=True)

    reg = rec.registered_views
    used_cams = sorted({v.camera_id for v in reg})
    undistort = {c: abs(rec.cameras[c].k) > undistort_threshold for c in used_cams}
    names: Dict[int, str] = {}
    seen = set()
    for v in reg:
        name = Path(v.path).name
        base, ext = Path(name).stem, Path(name).suffix
        k = 1
        while name.lower() in seen:
            name = f"{base}_{k}{ext}"
            k += 1
        seen.add(name.lower())
        names[v.id] = name

    def undistorted_px(cam: Camera, px: np.ndarray) -> np.ndarray:
        n = undistort_points(cam, px)
        return np.stack([cam.f * n[:, 0] + cam.cx, cam.f * n[:, 1] + cam.cy], axis=1)

    if copy_images:
        for v in reg:
            cam = rec.cameras[v.camera_id]
            dst = img_dir / names[v.id]
            if undistort[v.camera_id]:
                img = cv2.imread(v.path, cv2.IMREAD_COLOR)
                K = cam.K()
                img = cv2.undistort(img, K, np.array([cam.k, 0, 0, 0], dtype=np.float64), None, K)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
            else:
                shutil.copy2(v.path, dst)

    # point ids and per-image 2D lists
    pts = rec.points
    pid = {id(t): n + 1 for n, t in enumerate(pts)}
    kp_to_pid: Dict[int, Dict[int, int]] = {}
    for t in pts:
        for vid, kp in t.obs.items():
            if rec.views[vid].registered:
                kp_to_pid.setdefault(vid, {})[kp] = pid[id(t)]

    img_id = {v.id: n + 1 for n, v in enumerate(reg)}
    image_rows = []
    for v in reg:
        cam = rec.cameras[v.camera_id]
        mapping = kp_to_pid.get(v.id, {})
        # write only the keypoints that belong to a 3D point (keeps files small)
        kps = sorted(mapping)
        px = v.keypoints[kps] if kps else np.zeros((0, 2))
        if undistort[v.camera_id] and len(px):
            px = undistorted_px(cam, px)
        pts2d = [(float(px[n, 0]), float(px[n, 1]), mapping[k]) for n, k in enumerate(kps)]
        image_rows.append((img_id[v.id], _qvec(v.rvec), v.tvec, v.camera_id, names[v.id], pts2d, kps))

    # point tracks reference the 2D index inside each image's written list
    idx_in_image = {row[0]: {k: n for n, k in enumerate(row[6])} for row in image_rows}
    by_view = {v.id: img_id[v.id] for v in reg}

    ex_p = np.zeros((0, 3)) if extra_points is None else np.asarray(extra_points[0], dtype=np.float64)
    ex_c = np.zeros((0, 3), np.uint8) if extra_points is None else np.asarray(extra_points[1], dtype=np.uint8)

    # ---------- text
    with open(sp / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(used_cams)}\n")
        for c in used_cams:
            cam = rec.cameras[c]
            f.write(f"{c} PINHOLE {cam.width} {cam.height} {cam.f!r} {cam.f!r} {cam.cx!r} {cam.cy!r}\n")
    with open(sp / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_rows)}\n")
        for iid, q, t, c, name, p2d, _ in image_rows:
            f.write(f"{iid} {q[0]!r} {q[1]!r} {q[2]!r} {q[3]!r} {t[0]!r} {t[1]!r} {t[2]!r} {c} {name}\n")
            f.write(" ".join(f"{x!r} {y!r} {p}" for x, y, p in p2d) + "\n")
    with open(sp / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(pts) + len(ex_p)}\n")
        for t in pts:
            tr = [(by_view[v], idx_in_image[by_view[v]][kp]) for v, kp in t.obs.items() if v in by_view and kp in idx_in_image[by_view[v]]]
            col = t.color if t.color is not None else np.array([128, 128, 128])
            f.write(f"{pid[id(t)]} {t.xyz[0]!r} {t.xyz[1]!r} {t.xyz[2]!r} {int(col[0])} {int(col[1])} {int(col[2])} {t.error!r} "
                    + " ".join(f"{a} {b}" for a, b in tr) + "\n")
        for k, (x, c) in enumerate(zip(ex_p, ex_c)):
            f.write(f"{len(pts) + k + 1} {float(x[0])!r} {float(x[1])!r} {float(x[2])!r} {int(c[0])} {int(c[1])} {int(c[2])} 1.0\n")

    # ---------- binary (COLMAP's native format; most trainers prefer it)
    with open(sp / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(used_cams)))
        for c in used_cams:
            cam = rec.cameras[c]
            f.write(struct.pack("<iiQQ", c, 1, cam.width, cam.height))  # model 1 = PINHOLE
            f.write(struct.pack("<4d", cam.f, cam.f, cam.cx, cam.cy))
    with open(sp / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(image_rows)))
        for iid, q, t, c, name, p2d, _ in image_rows:
            f.write(struct.pack("<i4d3di", iid, *q, *t, c))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(p2d)))
            for x, y, p in p2d:
                f.write(struct.pack("<ddq", x, y, p))
    with open(sp / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(pts) + len(ex_p)))
        for t in pts:
            tr = [(by_view[v], idx_in_image[by_view[v]][kp]) for v, kp in t.obs.items() if v in by_view and kp in idx_in_image[by_view[v]]]
            col = t.color if t.color is not None else np.array([128, 128, 128])
            f.write(struct.pack("<Q3d3Bd", pid[id(t)], *map(float, t.xyz), int(col[0]), int(col[1]), int(col[2]), float(t.error)))
            f.write(struct.pack("<Q", len(tr)))
            for a, b in tr:
                f.write(struct.pack("<ii", a, b))
        for k, (x, c) in enumerate(zip(ex_p, ex_c)):
            f.write(struct.pack("<Q3d3Bd", len(pts) + k + 1, float(x[0]), float(x[1]), float(x[2]), int(c[0]), int(c[1]), int(c[2]), 1.0))
            f.write(struct.pack("<Q", 0))

    # sparse point cloud for quick inspection
    with open(out / "sparse_points.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for t in pts:
            col = t.color if t.color is not None else (128, 128, 128)
            f.write(f"{t.xyz[0]:.6f} {t.xyz[1]:.6f} {t.xyz[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")

    return {"path": str(out), "images": len(image_rows), "points": len(pts), "dense_points": len(ex_p), "cameras": len(used_cams),
            "undistorted": any(undistort.values())}


# ─────────────────────────────────────────────────────────────── reader (for tests / round trips)


def read_colmap_bin(sparse_dir: str) -> dict:
    sp = Path(sparse_dir)
    cams, imgs, pts = {}, {}, {}
    with open(sp / "cameras.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            n = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}[model]
            cams[cid] = (model, w, h, struct.unpack(f"<{n}d", f.read(8 * n)))
    with open(sp / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            vals = struct.unpack("<i4d3di", f.read(64))
            name = b""
            while (ch := f.read(1)) != b"\x00":
                name += ch
            n2 = struct.unpack("<Q", f.read(8))[0]
            p2d = [struct.unpack("<ddq", f.read(24)) for _ in range(n2)]
            imgs[vals[0]] = (np.array(vals[1:5]), np.array(vals[5:8]), vals[8], name.decode(), p2d)
    with open(sp / "points3D.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            v = struct.unpack("<Q3d3Bd", f.read(43))
            nt = struct.unpack("<Q", f.read(8))[0]
            tr = [struct.unpack("<ii", f.read(8)) for _ in range(nt)]
            pts[v[0]] = (np.array(v[1:4]), v[4:7], v[7], tr)
    return {"cameras": cams, "images": imgs, "points": pts}
