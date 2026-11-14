# Alignner

**Image Alignment for Photogrammetry**

A prototype application for multi-view image alignment, inspired by the alignment stage in [RealityScan](https://www.realityscan.com/) (formerly RealityCapture) by Epic Games.

## What it does

- Detects features (ORB or SIFT) across uploaded images
- Performs robust pairwise matching with Lowe’s ratio test
- Verifies geometry with RANSAC (Fundamental / Essential matrix)
- Estimates relative camera poses
- Builds a connectivity graph and reports connected components
- Visualizes approximate camera positions in 3D
- Shows match quality and feature statistics

## Quick Start

```bash
# Clone
git clone https://github.com/MalvinMakhuthe13/alignner.git
cd alignner

# Install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Usage

1. Upload 2 or more overlapping photographs of the same scene/object.
2. Adjust feature detector, max features, and inlier thresholds in the sidebar.
3. Click **Align Images**.
4. Review:
   - Number of verified pairs and components
   - Estimated camera positions
   - Sample match visualization
   - Feature counts per image

## Current Limitations

This is an early prototype:

- Uses a single approximate intrinsic matrix for all images
- Poses are pairwise relative (no full incremental SfM or global bundle adjustment yet)
- No dense reconstruction, meshing, or texturing
- No support for GPS/RTK, control points, or multi-camera calibration

For production-quality alignment and full photogrammetry pipelines, use:

- [RealityScan](https://www.realityscan.com/) (desktop + mobile)
- [COLMAP](https://colmap.github.io/)
- Meshroom / AliceVision

## Roadmap ideas

- Incremental structure-from-motion
- Bundle adjustment (e.g. via g2o / Ceres bindings or scipy)
- Per-image intrinsic estimation
- Export to COLMAP / RealityScan compatible formats
- Masking support (AI or manual)
- Integration with the broader FUMOCA / image_create ecosystem

## License

MIT

## Author

Malvin Makhuthe
