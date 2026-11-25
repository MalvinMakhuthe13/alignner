"""
Command-line alignment: a folder of photos → a COLMAP folder for Brush.

    python align.py photos/ out/
    python align.py photos/ out/ --features 12000 --focal 3100

The output folder contains images/ and sparse/0/ (binary + text) and
sparse_points.ply. Open it in Brush to train the splat.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from src.sfm import IncrementalSfM, export_colmap

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="folder of photos")
    ap.add_argument("out", help="output COLMAP folder")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb"])
    ap.add_argument("--features", type=int, default=8000)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--focal", type=float, default=None, help="focal length in pixels (default: EXIF, then refined)")
    ap.add_argument("--exhaustive-limit", type=int, default=60)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s")
    paths = sorted(str(p) for p in Path(a.images).iterdir() if p.suffix.lower() in EXTS)
    if len(paths) < 2:
        print("Need at least two photos.", file=sys.stderr)
        return 2
    sfm = IncrementalSfM(detector=a.detector, max_features=a.features, ratio=a.ratio, min_inliers=a.min_inliers,
                         focal_px=a.focal, exhaustive_limit=a.exhaustive_limit, sequential_window=a.window)
    try:
        rec = sfm.run(paths)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    info = export_colmap(rec, a.out)
    print(json.dumps({**rec.stats(), **info}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
