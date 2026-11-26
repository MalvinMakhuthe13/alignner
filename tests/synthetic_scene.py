"""Render a textured box on a textured ground plane from an orbit of cameras.

Planar faces are drawn with exact homographies, so ground-truth poses are
known precisely. Used to check the SfM end to end without real photos.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def texture(seed: int, size: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    for scale in (4, 16, 64, 256):
        noise = rng.integers(0, 255, (scale, scale, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 1.0, cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC), 0.35, 0)
    for _ in range(120):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p = tuple(int(x) for x in rng.integers(0, size, 2))
        cv2.circle(img, p, int(rng.integers(6, 40)), c, -1)
    return img


def look_at(C, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])        # rows: camera x (right), y (down), z (forward)
    return R, -R @ C


def faces():
    """(corners 4×3 world, texture seed) — ground plane then box faces (z up)."""
    g = 3.0
    out = [(np.array([[-g, -g, 0], [g, -g, 0], [g, g, 0], [-g, g, 0]], float), 1)]
    b, h = 0.8, 1.2
    P = lambda x, y, z: [x, y, z]
    out += [
        (np.array([P(-b, -b, h), P(b, -b, h), P(b, b, h), P(-b, b, h)], float), 2),      # top
        (np.array([P(-b, -b, 0), P(b, -b, 0), P(b, -b, h), P(-b, -b, h)], float), 3),    # -y
        (np.array([P(b, -b, 0), P(b, b, 0), P(b, b, h), P(b, -b, h)], float), 4),        # +x
        (np.array([P(b, b, 0), P(-b, b, 0), P(-b, b, h), P(b, b, h)], float), 5),        # +y
        (np.array([P(-b, b, 0), P(-b, -b, 0), P(-b, -b, h), P(-b, b, h)], float), 6),    # -x
    ]
    return out


def render(R, t, K, w, h, k1=0.0):
    img = np.full((h, w, 3), 30, np.uint8)
    tex_cache = {}
    for corners, seed in faces():
        Xc = corners @ R.T + t
        if np.any(Xc[:, 2] <= 0.05):
            continue
        normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        C = -R.T @ t
        if seed != 1 and np.dot(normal, C - corners.mean(0)) <= 0:
            continue                    # back face
        px = (Xc @ K.T)
        px = px[:, :2] / px[:, 2:3]
        tex = tex_cache.setdefault(seed, texture(seed))
        s = tex.shape[0]
        src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
        H = cv2.getPerspectiveTransform(src, np.float32(px))
        warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
        img[mask > 0] = warped[mask > 0]
    if k1:
        # apply SIMPLE_RADIAL distortion: sample undistorted image at inverse map
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        u, v = (xs - cx) / f, (ys - cy) / f
        uu, vv = u.copy(), v.copy()
        for _ in range(10):
            d = 1 + k1 * (uu * uu + vv * vv)
            uu, vv = u / d, v / d
        img = cv2.remap(img, (uu * f + cx).astype(np.float32), (vv * f + cy).astype(np.float32), cv2.INTER_LINEAR)
    return img


def make_scene(out_dir: str, n: int = 24, w: int = 1280, h: int = 960, f: float = 1100.0, k1: float = 0.0,
               radius: float = 5.0, height: float = 3.0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    gt = []
    for i in range(n):
        a = 2 * math.pi * i / n
        C = np.array([radius * math.cos(a), radius * math.sin(a), height + 0.4 * math.sin(3 * a)])
        R, t = look_at(C, (0, 0, 0.5))
        cv2.imwrite(str(out / f"img_{i:03d}.jpg"), render(R, t, K, w, h, k1), [cv2.IMWRITE_JPEG_QUALITY, 95])
        gt.append(C)
    return [str(out / f"img_{i:03d}.jpg") for i in range(n)], np.array(gt), K


def make_flyover(out_dir: str, rows: int = 4, cols: int = 6, w: int = 1280, h: int = 960, f: float = 1000.0,
                 altitude: float = 5.0, spacing: float = 0.9):
    """Nadir-ish grid flight (lawn-mower order) over the ground and box."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    paths, gt = [], []
    n = 0
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else reversed(range(cols))
        for c in cs:
            C = np.array([(c - (cols - 1) / 2) * spacing, (r - (rows - 1) / 2) * spacing, altitude])
            target = C + np.array([0.15, 0.1, -1.0])
            R, t = look_at(C, target, up=(0, 1, 0))
            p = out / f"fly_{n:03d}.jpg"
            cv2.imwrite(str(p), render(R, t, K, w, h), [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(p)); gt.append(C); n += 1
    return paths, np.array(gt), K
