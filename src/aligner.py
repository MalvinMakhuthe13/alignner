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
    components: List[List[int]]  # connected components of aligned images
    camera_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # R, t relative to first


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
            # Smart mode: nearby images + sparse long-range links
            candidate_pairs = []
            for i in range(n):
                # Nearby
                for d in range(1, max_neighbors + 1):
                    j = i + d
                    if j < n:
                        candidate_pairs.append((i, j))
                # Occasional longer jump to help connectivity
                if i % 8 == 0:
                    for d in range(max_neighbors + 1, min(max_neighbors * 3, n - i)):
                        if d % 3 == 0:
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

        # ── 6. Simple poses (relative to image 0 when possible) ─────
        poses = {0: (np.eye(3), np.zeros((3, 1)))}
        for p in pairs:
            if p.R is None or p.t is None:
                continue
            if p.idx1 == 0:
                poses[p.idx2] = (p.R, p.t)
            elif p.idx2 == 0:
                R_inv = p.R.T
                t_inv = -R_inv @ p.t
                poses[p.idx1] = (R_inv, t_inv)

        if progress_callback:
            progress_callback(1.0, "Done")

        logger.info(f"Alignment complete: {len(pairs)} verified pairs, {len(components)} components")
        return AlignmentResult(
            features=features,
            pairs=pairs,
            components=components,
            camera_poses=poses,
        )


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
