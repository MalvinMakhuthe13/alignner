# Alignner

**Image & Video Alignment for Photogrammetry**

Prototype inspired by the alignment stage in [RealityScan](https://www.realityscan.com/) (formerly RealityCapture).

Upload photos **or videos**. Videos are automatically turned into frames and then aligned.

## What it does

Real incremental Structure-from-Motion on the CPU (no AI, no GPU):

- SIFT (RootSIFT) features, mutual ratio-test matching, MAGSAC two-view verification
- Feature **tracks** across all views (not just pairs)
- Focal length from EXIF, then refined together with lens distortion
- Robust initial pair, PnP registration of every next photo, multi-view triangulation
- **Bundle adjustment** (Levenberg–Marquardt with Schur complement, Huber loss) — local after every photo, global periodically and at the end, with outlier removal
- Image retrieval (bag of visual words) so unordered photo sets and revisits still connect
- **COLMAP export** (`images/` + `sparse/0/` binary and text, full 2D↔3D tracks), undistorting images when the lens needs it — open the folder straight in **Brush**, gsplat or Nerfstudio
- Video frame extraction and optional near-duplicate removal

On synthetic scenes with known cameras it recovers poses to <0.02 % of the scene size at ~0.2 px mean reprojection error, recovers focal length from a 40 % wrong guess, and recovers lens distortion.

## The app

`python -m streamlit run app.py` — a dark, step-by-step workflow like RealityScan:

1. **Inputs** — photos or video; thumbnails turn green (placed, with point count) or red (not placed) after aligning
2. **Align** — Preview / Normal / High detail presets; cameras placed, points, error, components, lens
3. **Inspect** — 3D view with camera frustums coloured by fit quality; per-image quality analysis (error, points seen, frame coverage)
4. **Scene** — level (Y-up for Fumoca/Brush, Z-up for RealityScan/Blender) and centre; georeference from photo GPS (metres, east-north-up); define a known distance; clean stray points / crop to the main region
5. **Export** — COLMAP folder for Brush (.zip), sparse point cloud (.ply), report (.json)

## Command line

```bash
python align.py path/to/photos out_colmap
python align.py drone_photos out_colmap --gps --level z --clean
# then open out_colmap in Brush
```

## Capture tips

- 60–80 % overlap between neighbouring photos; walk around the object instead of rotating in place
- Keep EXIF (don't strip metadata) — especially for drone/flyover sets, where focal length cannot be recovered from a flat scene alone
- Avoid motion blur; lock exposure if you can

## Tests

```bash
python tests/test_sfm.py          # end-to-end on rendered scenes with known cameras
python tests/test_scene_tools.py  # level, GPS, define distance, cleanup
python -m pytest -q tests/test_image_selector.py
```

## Current limitations

- Runs on the CPU in Python: a few minutes for ~50 photos at 8000 features; large sets (500+) are slow
- One camera model (SIMPLE_RADIAL); fisheye lenses aren't supported
- No GPS / ground-control georeferencing yet

## License

MIT

## Author

Malvin Makhuthe
