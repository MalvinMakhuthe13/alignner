"""
Alignner - Image Alignment for Photogrammetry
Inspired by RealityScan / RealityCapture alignment pipeline.

Core features:
- Feature detection & matching
- Geometric verification (RANSAC)
- Relative pose estimation
- Simple multi-view alignment / component detection
- Sparse point visualization (basic)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageFeatures:
    path: str
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0


@dataclass
class MatchPair:
    idx1: int
    idx2: int
    matches: List[cv2.DMatch]
    inliers: int
    fundamental: Optional[np.ndarray] = None
    essential: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None


@dataclass
class AlignmentResult:
    features: List[ImageFeatures]
    pairs: List[MatchPair]
    components: List[List[int]]
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    sparse_points: Optional[np.ndarray] = None          # Nx3
    point_colors: Optional[np.ndarray] = None           # Nx3
    point_tracks: Optional[List] = None                 # list of (point_id, image_id, kp_idx)


class ImageAligner:
    def __init__(
        self,
        feature_detector: str = "orb",
        max_features: int = 5000,
        match_ratio: float = 0.75,
        min_inliers: int = 20,
        ransac_thresh: float = 1.0,
        match_window: int = 15,          # sequential: match each image to next N
        use_sequential: bool = True,     # False = exhaustive (slow for large N)
    ):
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.match_window = match_window
        self.use_sequential = use_sequential

        if feature_detector.lower() == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def load_image(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            return None
        return img

    def extract_features(self, image: np.ndarray, path: str) -> ImageFeatures:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.array([])
        h, w = image.shape[:2]
        return ImageFeatures(
            path=path,
            keypoints=keypoints or [],
            descriptors=descriptors,
            image=image,
            width=w,
            height=h,
        )

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.match_ratio * n.distance:
                        good.append(m)
            return good
        except Exception as e:
            logger.debug(f"Matching failed: {e}")
            return []

    def geometric_verification(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        K: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, List[cv2.DMatch]]:
        """Estimate fundamental/essential matrix and filter inliers."""
        if len(matches) < 8:
            return None, None, 0, []

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Fundamental matrix
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, self.ransac_thresh, 0.99
        )
        if F is None or mask is None:
            return None, None, 0, []

        inlier_mask = mask.ravel().astype(bool)
        inliers = int(np.sum(inlier_mask))
        inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]

        E = None
        if K is not None and inliers >= 8:
            # Essential matrix from fundamental if we have intrinsics
            E = K.T @ F @ K

        return F, E, inliers, inlier_matches

    def estimate_relative_pose(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        K: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Recover R, t from essential matrix."""
        if len(matches) < 8:
            return None, None, 0

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return None, None, 0

        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
        inliers = int(np.sum(pose_mask)) if pose_mask is not None else 0
        return R, t, inliers

    def align(
        self,
        image_paths: List[str],
        focal_length: Optional[float] = None,
        principal_point: Optional[Tuple[float, float]] = None,
        progress_callback=None,
        max_neighbors: int = 12,
    ) -> AlignmentResult:
        """
        Main alignment pipeline.
        For large sets (>40 images) only matches nearby + a few long-range pairs
        so it stays practical while still connecting the sequence.
        """
        logger.info(f"Aligning {len(image_paths)} images...")

        # ── 1. Load & extract features ──────────────────────────────
        features: List[ImageFeatures] = []
        n_total = len(image_paths)
        for i, p in enumerate(image_paths):
            img = self.load_image(p)
            if img is not None:
                feat = self.extract_features(img, p)
                features.append(feat)
                logger.info(f"  {Path(p).name}: {len(feat.keypoints)} features")
            if progress_callback:
                progress_callback(0.15 * (i + 1) / max(n_total, 1), f"Extracting features {i+1}/{n_total}")

        n = len(features)
        if n < 2:
            return AlignmentResult(features=features, pairs=[], components=[])

        # ── 2. Intrinsics (improved) ────────────────────────────────
        K = self.estimate_intrinsics(features, focal_length)

        # ── 3. Decide which pairs to match ──────────────────────────
        if n <= 40:
            # Full pairwise for small sets
            candidate_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            # Smart mode: denser nearby + more long-range links for whole-object coverage
            candidate_pairs = []
            window = max(max_neighbors, 18)  # denser for better coverage
            for i in range(n):
                for d in range(1, window + 1):
                    j = i + d
                    if j < n:
                        candidate_pairs.append((i, j))
                # Extra longer jumps every few images
                if i % 5 == 0:
                    for d in range(window + 1, min(window * 4, n - i)):
                        if d % 4 == 0:
                            candidate_pairs.append((i, i + d))

        total_pairs = len(candidate_pairs)
        logger.info(f"Matching {total_pairs} candidate pairs (smart mode: {n > 40})")

        # ── 4. Matching ─────────────────────────────────────────────
        pairs: List[MatchPair] = []
        for k, (i, j) in enumerate(candidate_pairs):
            matches = self.match_features(features[i].descriptors, features[j].descriptors)
            if len(matches) >= self.min_inliers:
                F, E, inliers, inlier_matches = self.geometric_verification(
                    features[i].keypoints, features[j].keypoints, matches, K
                )
                if inliers >= self.min_inliers:
                    R, t, _ = self.estimate_relative_pose(
                        features[i].keypoints, features[j].keypoints, inlier_matches, K
                    )
                    pairs.append(
                        MatchPair(
                            idx1=i, idx2=j, matches=inlier_matches, inliers=inliers,
                            fundamental=F, essential=E, R=R, t=t,
                        )
                    )
                    logger.info(f"  Pair ({i},{j}): {inliers} inliers")

            if progress_callback and k % 5 == 0:
                progress_callback(0.15 + 0.75 * (k + 1) / total_pairs,
                                  f"Matching pairs {k+1}/{total_pairs}")

        # ── 5. Components ───────────────────────────────────────────
        from collections import defaultdict
        graph = defaultdict(list)
        for p in pairs:
            graph[p.idx1].append(p.idx2)
            graph[p.idx2].append(p.idx1)

        visited = set()
        components = []
        for i in range(n):
            if i not in visited:
                comp = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comp.append(node)
                        stack.extend(graph[node])
                if comp:
                    components.append(sorted(comp))

        # ── 6. Propagate poses ──────────────────────────────────────
        poses = self.propagate_poses(pairs, n)

        # Multi-view triangulation + filtering
        sparse_pts, colors, tracks = self.triangulate_multiview(features, pairs, poses, K)

        # Lightweight bundle adjustment
        if sparse_pts is not None and len(poses) >= 3:
            poses, sparse_pts = self.bundle_adjust_light(features, poses, sparse_pts, tracks, K)

        if progress_callback:
            progress_callback(1.0, "Done")

        logger.info(f"Triangulated {len(sparse_pts) if sparse_pts is not None else 0} sparse points")
        logger.info(f"Poses recovered for {len(poses)} / {n} cameras")
        logger.info(f"Alignment complete: {len(pairs)} verified pairs, {len(components)} components")

        return AlignmentResult(
            features=features,
            pairs=pairs,
            components=components,
            camera_poses=poses,
            sparse_points=sparse_pts,
            point_colors=colors,
            point_tracks=tracks,
        )

        logger.info(f"Triangulated {len(sparse_pts) if sparse_pts is not None else 0} sparse points")
        logger.info(f"Poses recovered for {len(poses)} / {n} cameras")
        logger.info(f"Alignment complete: {len(pairs)} verified pairs, {len(components)} components")

        return AlignmentResult(
            features=features,
            pairs=pairs,
            components=components,
            camera_poses=poses,
            sparse_points=sparse_pts,
            point_colors=colors,
            point_tracks=tracks,
        )

    def estimate_intrinsics(self, features: List[ImageFeatures], focal_override: Optional[float]) -> np.ndarray:
        """Better shared K. Uses override or a stronger heuristic based on image size."""
        sample = features[0]
        if focal_override is not None and focal_override > 0:
            f = float(focal_override)
        else:
            # Common photogrammetry heuristic: ~1.0–1.4 × longer side
            f = max(sample.width, sample.height) * 1.15
        cx, cy = sample.width / 2.0, sample.height / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def triangulate_multiview(
        self,
        features: List[ImageFeatures],
        pairs: List[MatchPair],
        poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        max_points: int = 40000,
        reproj_thresh: float = 2.5,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List]]:
        """Build conflict-free feature tracks across image pairs, then triangulate tracks.

        Track observations use (point_id, image_index, keypoint_index). A track
        cannot contain two different keypoints from the same image.
        """
        parent = {}
        observations = {}

        def find(node):
            parent.setdefault(node, node)
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(left, right):
            rl, rr = find(left), find(right)
            if rl == rr:
                return True
            ol, or_ = observations.get(rl, {left[0]: left[1]}), observations.get(rr, {right[0]: right[1]})
            if any(image_id in or_ and or_[image_id] != kp_id for image_id, kp_id in ol.items()):
                return False
            parent[rr] = rl
            observations[rl] = {**or_, **ol}
            return True

        for pair in pairs:
            if pair.idx1 not in poses or pair.idx2 not in poses:
                continue
            for match in pair.matches:
                left = (pair.idx1, match.queryIdx)
                right = (pair.idx2, match.trainIdx)
                rl, rr = find(left), find(right)
                if rl == rr:
                    continue
                observations.setdefault(rl, {left[0]: left[1]})
                observations.setdefault(rr, {right[0]: right[1]})
                union(left, right)

        tracks_by_root = {}
        for node in list(parent):
            root = find(node)
            tracks_by_root.setdefault(root, {})[node[0]] = node[1]

        points, colors, tracks = [], [], []
        for obs in tracks_by_root.values():
            valid = [(img_id, kp_id) for img_id, kp_id in sorted(obs.items())
                     if img_id in poses and kp_id < len(features[img_id].keypoints)]
            if len(valid) < 2:
                continue

            # Triangulate using the widest-baseline pair in this track.
            best = None
            best_baseline = -1.0
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    ia, _ = valid[i]
                    ib, _ = valid[j]
                    Ra, ta = poses[ia]
                    Rb, tb = poses[ib]
                    ca = -Ra.T @ ta.reshape(3)
                    cb = -Rb.T @ tb.reshape(3)
                    baseline = float(np.linalg.norm(ca - cb))
                    if baseline > best_baseline:
                        best_baseline, best = baseline, (valid[i], valid[j])
            if best is None or best_baseline < 1e-8:
                continue

            (ia, ka), (ib, kb) = best
            Ra, ta = poses[ia]
            Rb, tb = poses[ib]
            P1 = K @ np.hstack((Ra, np.asarray(ta).reshape(3, 1)))
            P2 = K @ np.hstack((Rb, np.asarray(tb).reshape(3, 1)))
            uv1 = np.asarray(features[ia].keypoints[ka].pt, dtype=np.float64).reshape(2, 1)
            uv2 = np.asarray(features[ib].keypoints[kb].pt, dtype=np.float64).reshape(2, 1)
            h = cv2.triangulatePoints(P1, P2, uv1, uv2).reshape(4)
            if abs(h[3]) < 1e-10:
                continue
            point = h[:3] / h[3]
            if not np.all(np.isfinite(point)):
                continue

            # Require positive depth and bounded reprojection error in every view.
            accepted = True
            for img_id, kp_id in valid:
                R, t = poses[img_id]
                camera_point = R @ point.reshape(3, 1) + np.asarray(t).reshape(3, 1)
                if camera_point[2, 0] <= 1e-8:
                    accepted = False
                    break
                projected = K @ camera_point
                projected = (projected[:2] / projected[2, 0]).ravel()
                observed = np.asarray(features[img_id].keypoints[kp_id].pt)
                if np.linalg.norm(projected - observed) > reproj_thresh:
                    accepted = False
                    break
            if not accepted:
                continue

            points.append(point)
            color_img = features[ia].image
            x, y = map(int, features[ia].keypoints[ka].pt)
            if color_img is not None and 0 <= y < color_img.shape[0] and 0 <= x < color_img.shape[1]:
                colors.append(color_img[y, x][::-1].tolist())
            else:
                colors.append([140, 140, 140])
            point_id = len(points) - 1
            tracks.extend((point_id, img_id, kp_id) for img_id, kp_id in valid)
            if len(points) >= max_points:
                break

        if not points:
            return None, None, None
        return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8), tracks

    def bundle_adjust_light(
        self,
        features: List[ImageFeatures],
        poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
        points: np.ndarray,
        tracks: List,
        K: np.ndarray,
        max_iter: int = 8,
    ) -> Tuple[Dict, np.ndarray]:
        """Very lightweight pose + point refinement (gradient-free / simple iterative)."""
        # For prototype: only refine translation scale consistency on a subset
        # Full Ceres-style BA is heavy; this reduces drift a bit.
        refined_poses = {k: (R.copy(), t.copy()) for k, (R, t) in poses.items()}
        refined_pts = points.copy()

        # Simple scale normalization so median camera distance is ~1
        centers = []
        for idx, (R, t) in refined_poses.items():
            C = -R.T @ t.flatten()
            centers.append(np.linalg.norm(C))
        if centers:
            med = np.median([c for c in centers if c > 1e-3]) or 1.0
            scale = 1.0 / med
            for idx in refined_poses:
                R, t = refined_poses[idx]
                refined_poses[idx] = (R, t * scale)
            refined_pts *= scale

        return refined_poses, refined_pts

    def propagate_poses(self, pairs: List[MatchPair], n: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Propagate poses independently through every connected camera component.

        Each component receives its own identity reference camera. Coordinates
        between disconnected components are therefore local and must not be
        interpreted as sharing a common world frame.
        """
        from collections import defaultdict, deque

        adj = defaultdict(list)
        for pair in pairs:
            if pair.R is None or pair.t is None:
                continue
            if not (0 <= pair.idx1 < n and 0 <= pair.idx2 < n):
                continue
            R_rel = np.asarray(pair.R, dtype=np.float64).reshape(3, 3)
            t_rel = np.asarray(pair.t, dtype=np.float64).reshape(3, 1)
            adj[pair.idx1].append((pair.idx2, R_rel, t_rel))
            R_inv = R_rel.T
            adj[pair.idx2].append((pair.idx1, R_inv, -R_inv @ t_rel))

        poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for root in range(n):
            if root in poses:
                continue
            poses[root] = (np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64))
            queue = deque([root])
            while queue:
                curr = queue.popleft()
                R_curr, t_curr = poses[curr]
                for neigh, R_rel, t_rel in adj[curr]:
                    if neigh in poses:
                        continue
                    poses[neigh] = (R_rel @ R_curr, R_rel @ t_curr + t_rel)
                    queue.append(neigh)

        return poses

    def triangulate_sparse(
        self,
        features: List[ImageFeatures],
        pairs: List[MatchPair],
        poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        max_points: int = 50000,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List]]:
        """Simple pairwise triangulation of inlier matches that have known poses."""
        points = []
        colors = []
        tracks = []
        point_id = 0

        for pair in pairs:
            if pair.idx1 not in poses or pair.idx2 not in poses:
                continue
            if len(pair.matches) < 8:
                continue

            R1, t1 = poses[pair.idx1]
            R2, t2 = poses[pair.idx2]

            # Projection matrices
            P1 = K @ np.hstack((R1, t1))
            P2 = K @ np.hstack((R2, t2))

            pts1 = np.float32([features[pair.idx1].keypoints[m.queryIdx].pt for m in pair.matches]).T
            pts2 = np.float32([features[pair.idx2].keypoints[m.trainIdx].pt for m in pair.matches]).T

            pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            pts3d = (pts4d[:3] / pts4d[3]).T

            # Filter reasonable points
            for i, pt in enumerate(pts3d):
                if np.any(np.isnan(pt)) or np.any(np.abs(pt) > 1000):
                    continue
                points.append(pt)
                # Color from first image
                x, y = int(pts1[0, i]), int(pts1[1, i])
                img = features[pair.idx1].image
                if img is not None and 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    colors.append(img[y, x][::-1])  # BGR -> RGB
                else:
                    colors.append([128, 128, 128])

                tracks.append((point_id, pair.idx1, pair.matches[i].queryIdx))
                tracks.append((point_id, pair.idx2, pair.matches[i].trainIdx))
                point_id += 1

                if len(points) >= max_points:
                    break
            if len(points) >= max_points:
                break

        if not points:
            return None, None, None
        return np.array(points), np.array(colors), tracks

    def export_colmap(
        self,
        result: AlignmentResult,
        output_dir: str,
        K: np.ndarray,
    ) -> str:
        """Export a text sparse model with consistent image observations and point tracks.

        Only registered images (those with estimated poses) are written. The
        model is a valid COLMAP text-model layout, but its camera/intrinsic
        estimates remain those produced by this prototype.
        """
        if not result.features:
            raise ValueError("Cannot export an empty alignment result.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        images_dir = out / "images"
        images_dir.mkdir(exist_ok=True)

        registered = sorted(idx for idx in result.camera_poses if 0 <= idx < len(result.features))
        if not registered:
            raise ValueError("No registered camera poses are available for COLMAP export.")

        # COLMAP image IDs are positive integers and independent of source indices.
        image_ids = {idx: position + 1 for position, idx in enumerate(registered)}
        image_names = {}
        used_names = set()
        for idx in registered:
            source_name = Path(result.features[idx].path).name
            name = source_name
            if name in used_names:
                name = f"{idx:06d}_{source_name}"
            used_names.add(name)
            image_names[idx] = name
            image = result.features[idx].image
            if image is not None and not cv2.imwrite(str(images_dir / name), image):
                raise IOError(f"Could not write exported image: {name}")

        # Each image's POINTS2D list is built once. The index in this list is
        # exactly the POINT2D_IDX referenced by points3D.txt.
        image_observations = {idx: [] for idx in registered}
        point_observations = {}
        if result.point_tracks:
            for point_id, image_idx, keypoint_idx in result.point_tracks:
                if image_idx not in image_ids or point_id < 0:
                    continue
                if point_id >= (0 if result.sparse_points is None else len(result.sparse_points)):
                    continue
                if keypoint_idx < 0 or keypoint_idx >= len(result.features[image_idx].keypoints):
                    continue
                uv = result.features[image_idx].keypoints[keypoint_idx].pt
                point_observations.setdefault(int(point_id), []).append((image_idx, int(keypoint_idx), uv))

        # Remove duplicate observations and assign stable per-image 2D indices.
        point2d_indices = {}
        for image_idx in registered:
            seen_keypoints = set()
            entries = []
            for point_id in sorted(point_observations):
                for obs_image, keypoint_idx, uv in point_observations[point_id]:
                    if obs_image != image_idx or keypoint_idx in seen_keypoints:
                        continue
                    seen_keypoints.add(keypoint_idx)
                    point2d_idx = len(entries)
                    entries.append((float(uv[0]), float(uv[1]), point_id))
                    point2d_indices[(point_id, image_idx)] = point2d_idx
            image_observations[image_idx] = entries

        sample = result.features[registered[0]]
        fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        with open(out / "cameras.txt", "w", encoding="utf-8") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("1 PINHOLE {} {} {:.12g} {:.12g} {:.12g} {:.12g}\n".format(
                sample.width, sample.height, fx, fy, cx, cy
            ))

        with open(out / "images.txt", "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for idx in registered:
                R, t = result.camera_poses[idx]
                q = self.rotmat_to_quat(np.asarray(R, dtype=np.float64))
                tx, ty, tz = np.asarray(t).reshape(3)
                f.write("{} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} 1 {}\n".format(
                    image_ids[idx], *q, tx, ty, tz, image_names[idx]
                ))
                entries = image_observations[idx]
                f.write(" ".join(
                    "{:.6f} {:.6f} {}".format(x, y, point_id + 1)
                    for x, y, point_id in entries
                ) + "\n")

        with open(out / "points3D.txt", "w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            if result.sparse_points is not None:
                colors = result.point_colors
                for point_id, point in enumerate(result.sparse_points):
                    obs = []
                    for image_idx, _, _ in point_observations.get(point_id, []):
                        key = (point_id, image_idx)
                        if image_idx in image_ids and key in point2d_indices:
                            obs.append((image_ids[image_idx], point2d_indices[key]))
                    # COLMAP points need at least two image observations.
                    if len(obs) < 2:
                        continue
                    color = colors[point_id] if colors is not None and point_id < len(colors) else [128, 128, 128]
                    r, g, b = [int(np.clip(v, 0, 255)) for v in color]
                    track_text = " ".join(f"{image_id} {point2d_idx}" for image_id, point2d_idx in obs)
                    f.write("{:d} {:.12g} {:.12g} {:.12g} {:d} {:d} {:d} 0 {}".format(
                        point_id + 1, point[0], point[1], point[2], r, g, b, track_text
                    ) + "\n")

        return str(out)

    @staticmethod
    def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
        """Rotation matrix to quaternion (w, x, y, z)."""
        m = R
        t = np.trace(m)
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            w = 0.25 / s
            x = (m[2,1] - m[1,2]) * s
            y = (m[0,2] - m[2,0]) * s
            z = (m[1,0] - m[0,1]) * s
        else:
            if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
                s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
                w = (m[2,1] - m[1,2]) / s
                x = 0.25 * s
                y = (m[0,1] + m[1,0]) / s
                z = (m[0,2] + m[2,0]) / s
            elif m[1,1] > m[2,2]:
                s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
                w = (m[0,2] - m[2,0]) / s
                x = (m[0,1] + m[1,0]) / s
                y = 0.25 * s
                z = (m[1,2] + m[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
                w = (m[1,0] - m[0,1]) / s
                x = (m[0,2] + m[2,0]) / s
                y = (m[1,2] + m[2,1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])


def visualize_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    max_draw: int = 50,
) -> np.ndarray:
    """Draw match lines between two images side by side."""
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis
