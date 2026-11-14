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

        # ── 2. Intrinsics ───────────────────────────────────────────
        if focal_length is None:
            avg_size = np.mean([max(f.width, f.height) for f in features])
            focal_length = avg_size * 1.2

        sample = features[0]
        cx = principal_point[0] if principal_point else sample.width / 2.0
        cy = principal_point[1] if principal_point else sample.height / 2.0
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float64)

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

        # ── 6. Propagate poses along the connected component ─────────
        poses = self.propagate_poses(pairs, n)

        if progress_callback:
            progress_callback(1.0, "Done")

        # Triangulate sparse points
        sparse_pts, colors, tracks = self.triangulate_sparse(features, pairs, poses, K)
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

    def propagate_poses(self, pairs: List[MatchPair], n: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Chain relative poses from image 0 across the whole connected component."""
        from collections import defaultdict, deque

        # Build adjacency with relative transforms
        adj = defaultdict(list)
        for p in pairs:
            if p.R is None or p.t is None:
                continue
            # Store both directions
            adj[p.idx1].append((p.idx2, p.R, p.t))
            # Inverse
            R_inv = p.R.T
            t_inv = -R_inv @ p.t
            adj[p.idx2].append((p.idx1, R_inv, t_inv))

        poses = {0: (np.eye(3), np.zeros((3, 1)))}
        queue = deque([0])
        visited = {0}

        while queue:
            curr = queue.popleft()
            R_curr, t_curr = poses[curr]
            for neigh, R_rel, t_rel in adj[curr]:
                if neigh in visited:
                    continue
                # Compose: R_neigh = R_rel @ R_curr, t_neigh = R_rel @ t_curr + t_rel
                R_new = R_rel @ R_curr
                t_new = R_rel @ t_curr + t_rel
                poses[neigh] = (R_new, t_new)
                visited.add(neigh)
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
        """Write a minimal COLMAP-compatible folder (cameras.txt, images.txt, points3D.txt + images)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        images_dir = out / "images"
        images_dir.mkdir(exist_ok=True)

        # cameras.txt (PINHOLE)
        with open(out / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            sample = result.features[0]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            f.write(f"1 PINHOLE {sample.width} {sample.height} {fx} {fy} {cx} {cy}\n")

        # images.txt
        with open(out / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(result.features)}\n")

            for img_id, feat in enumerate(result.features):
                name = Path(feat.path).name
                # Copy / save image
                if feat.image is not None:
                    cv2.imwrite(str(images_dir / name), feat.image)

                if img_id in result.camera_poses:
                    R, t = result.camera_poses[img_id]
                    # COLMAP uses world-to-camera: R, t where t = -R @ C
                    q = self.rotmat_to_quat(R)
                    tx, ty, tz = t.flatten()
                else:
                    q = [1, 0, 0, 0]
                    tx = ty = tz = 0

                f.write(f"{img_id+1} {q[0]} {q[1]} {q[2]} {q[3]} {tx} {ty} {tz} 1 {name}\n")
                f.write("\n")  # empty POINTS2D line for now

        # points3D.txt
        with open(out / "points3D.txt", "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            npts = 0 if result.sparse_points is None else len(result.sparse_points)
            f.write(f"# Number of points: {npts}\n")

            if result.sparse_points is not None:
                for pid, (pt, col) in enumerate(zip(result.sparse_points, result.point_colors)):
                    r, g, b = int(col[0]), int(col[1]), int(col[2])
                    f.write(f"{pid+1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0.5\n")

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
