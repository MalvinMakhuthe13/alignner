"""
Command-line alignment: a folder of photos → a COLMAP folder for Brush.

    python align.py photos/ out/
    python align.py photos/ out/ --features 12000 --focal 3100
    python align.py drone/ out/ --gps --level z --clean
    python align.py photos/ out/ --dense        # dense, object-shaped cloud for Brush

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
    ap.add_argument("--level", choices=["y", "z"], help="level the scene: y-up (Fumoca/Brush) or z-up (RealityScan/Blender), and centre it")
    ap.add_argument("--gps", action="store_true", help="scale to metres and orient east-north-up from photo GPS")
    ap.add_argument("--clean", action="store_true", help="drop stray sparse points (statistical outliers)")
    ap.add_argument("--dense", action="store_true", help="build a dense cloud (looks like the object) and start Brush from it")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s")
    folder = Path(a.images).expanduser()
    if not folder.is_dir():
        print(f"Photo folder not found: {folder}\n"
              "Tip: in File Explorer, open your photo folder, click the address bar, copy the path "
              "and paste it here in quotes.", file=sys.stderr)
        return 2
    paths = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if len(paths) < 2:
        print(f"Found {len(paths)} photo(s) in {folder}; need at least two (.jpg, .png, .tif, .webp).", file=sys.stderr)
        return 2
    print(f"Aligning {len(paths)} photos from {folder} ...")
    sfm = IncrementalSfM(detector=a.detector, max_features=a.features, ratio=a.ratio, min_inliers=a.min_inliers,
                         focal_px=a.focal, exhaustive_limit=a.exhaustive_limit, sequential_window=a.window)
    try:
        rec = sfm.run(paths)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    from src import scene_tools as T
    if a.gps:
        try:
            print("GPS:", T.georeference_from_gps(rec))
        except ValueError as e:
            print(f"GPS skipped: {e}", file=sys.stderr)
    if a.level:
        T.level_scene(rec, up_axis=a.level)
    if a.clean:
        print(f"Removed {T.clean_points(rec):,} stray points")
    extra = None
    if a.dense:
        from src.dense import densify, write_ply
        cloud = densify(rec)
        extra = (cloud.points, cloud.colors)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        write_ply(str(Path(a.out) / "dense_points.ply"), cloud)
        print(f"Dense cloud: {len(cloud):,} points from {cloud.pairs_used} photo pairs")
    info = export_colmap(rec, a.out, extra_points=extra)
    print(json.dumps({**rec.stats(), **info}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
