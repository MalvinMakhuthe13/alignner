"""
Conservative image-selection preflight for Alignner.

The selector is intentionally conservative:
- rejects unreadable images;
- flags low-quality images for review;
- removes only very strong near-duplicates by default;
- keeps the original file order;
- never deletes or modifies source files.

It is a pre-alignment optimisation aid, not a replacement for COLMAP/SfM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
import csv

import cv2
import numpy as np


@dataclass
class ImageAssessment:
    index: int
    path: str
    filename: str
    width: int
    height: int
    sharpness: float
    exposure: float
    feature_count: int
    quality_score: float
    duplicate_of: Optional[int] = None
    decision: str = "keep"
    reason: str = "Unique candidate"


@dataclass
class SelectionResult:
    assessments: list[ImageAssessment]
    selected_paths: list[str]
    excluded_paths: list[str]

    @property
    def selected_count(self) -> int:
        return len(self.selected_paths)

    @property
    def input_count(self) -> int:
        return len(self.assessments)

    @property
    def reduction_percent(self) -> float:
        if not self.input_count:
            return 0.0
        return 100.0 * (self.input_count - self.selected_count) / self.input_count

    def to_csv(self, destination: str | Path) -> None:
        fields = list(asdict(self.assessments[0]).keys()) if self.assessments else [
            "index", "path", "filename", "width", "height", "sharpness",
            "exposure", "feature_count", "quality_score", "duplicate_of",
            "decision", "reason"
        ]
        with open(destination, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in self.assessments:
                writer.writerow(asdict(item))


class IntelligentImageSelector:
    """
    Conservative selector. A duplicate is removed only when BOTH:
      1) perceptual dHash distance is very small; and
      2) ORB descriptor matching indicates strong visual correspondence.

    Defaults are designed to avoid aggressive culling on photogrammetry sets.
    """

    def __init__(
        self,
        max_features: int = 2500,
        duplicate_hash_distance: int = 3,
        duplicate_match_ratio: float = 0.88,
        min_features_for_keep: int = 35,
        min_sharpness_for_review: float = 18.0,
        max_image_side: int = 1600,
    ):
        self.max_features = max_features
        self.duplicate_hash_distance = duplicate_hash_distance
        self.duplicate_match_ratio = duplicate_match_ratio
        self.min_features_for_keep = min_features_for_keep
        self.min_sharpness_for_review = min_sharpness_for_review
        self.max_image_side = max_image_side
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return image
        scale = max_side / float(longest)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _dhash(gray: np.ndarray) -> int:
        tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = tiny[:, 1:] > tiny[:, :-1]
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)
        return value

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    @staticmethod
    def _quality(sharpness: float, exposure: float, feature_count: int) -> float:
        # Bounded, explainable heuristic; not a learned quality probability.
        sharp = min(1.0, sharpness / 180.0)
        exp = max(0.0, 1.0 - abs(exposure - 127.5) / 127.5)
        feat = min(1.0, feature_count / 1200.0)
        return round(100.0 * (0.45 * sharp + 0.20 * exp + 0.35 * feat), 2)

    def _match_strength(self, desc_a: Optional[np.ndarray], desc_b: Optional[np.ndarray]) -> float:
        if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
            return 0.0
        try:
            pairs = self.matcher.knnMatch(desc_a, desc_b, k=2)
            good = 0
            valid = 0
            for pair in pairs:
                if len(pair) == 2:
                    valid += 1
                    if pair[0].distance < 0.72 * pair[1].distance:
                        good += 1
            denom = max(1, min(len(desc_a), len(desc_b)))
            return min(1.0, good / float(denom))
        except cv2.error:
            return 0.0

    def analyze(self, image_paths: Iterable[str]) -> SelectionResult:
        paths = [str(p) for p in image_paths]
        records: list[ImageAssessment] = []
        descriptors: list[Optional[np.ndarray]] = []
        hashes: list[Optional[int]] = []

        for index, path in enumerate(paths):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                records.append(ImageAssessment(
                    index, path, Path(path).name, 0, 0, 0.0, 0.0, 0, 0.0,
                    decision="exclude", reason="Unreadable image"
                ))
                descriptors.append(None)
                hashes.append(None)
                continue

            h, w = image.shape[:2]
            small = self._resize(image, self.max_image_side)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            exposure = float(np.mean(gray))
            keypoints, desc = self.detector.detectAndCompute(gray, None)
            count = len(keypoints or [])
            score = self._quality(sharpness, exposure, count)

            records.append(ImageAssessment(
                index=index,
                path=path,
                filename=Path(path).name,
                width=w,
                height=h,
                sharpness=round(sharpness, 2),
                exposure=round(exposure, 2),
                feature_count=count,
                quality_score=score,
                decision="keep",
                reason="Unique candidate",
            ))
            descriptors.append(desc)
            hashes.append(self._dhash(gray))

        # Only compare a bounded temporal neighbourhood. For capture sequences,
        # near-duplicates are usually adjacent; this avoids O(n^2) matching.
        duplicate_edges: list[tuple[int, int]] = []
        n = len(records)
        window = 8
        for i in range(n):
            if records[i].decision == "exclude" or hashes[i] is None:
                continue
            for j in range(i + 1, min(n, i + window + 1)):
                if records[j].decision == "exclude" or hashes[j] is None:
                    continue
                if self._hamming(hashes[i], hashes[j]) > self.duplicate_hash_distance:
                    continue
                strength = self._match_strength(descriptors[i], descriptors[j])
                if strength >= self.duplicate_match_ratio:
                    duplicate_edges.append((i, j))

        # Greedy conservative deduplication: retain the better-quality member.
        # If scores tie, retain the earlier image for reproducibility.
        for i, j in duplicate_edges:
            a, b = records[i], records[j]
            if a.decision == "exclude" or b.decision == "exclude":
                continue
            loser, winner = (b, a) if (a.quality_score, -a.index) >= (b.quality_score, -b.index) else (a, b)
            loser.decision = "exclude"
            loser.duplicate_of = winner.index
            loser.reason = f"Near-duplicate of {winner.filename}; lower/equal quality score"

        # Low-quality images are flagged, not automatically discarded.
        for item in records:
            if item.decision == "keep":
                if item.feature_count < self.min_features_for_keep:
                    item.decision = "review"
                    item.reason = "Very few local features; inspect before alignment"
                elif item.sharpness < self.min_sharpness_for_review:
                    item.decision = "review"
                    item.reason = "Potentially blurred; inspect before alignment"

        selected = [r.path for r in records if r.decision != "exclude"]
        excluded = [r.path for r in records if r.decision == "exclude"]
        return SelectionResult(records, selected, excluded)
