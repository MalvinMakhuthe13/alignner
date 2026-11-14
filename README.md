# Alignner

**Image & Video Alignment for Photogrammetry**

Prototype inspired by the alignment stage in [RealityScan](https://www.realityscan.com/) (formerly RealityCapture).

Upload photos **or videos**. Videos are automatically turned into frames and then aligned.

## Features

- Feature detection (ORB / SIFT)
- Robust matching + RANSAC geometric verification
- Relative pose estimation
- Connected components
- 3D camera position visualization
- Multi-view feature tracks and sparse point triangulation
- COLMAP text-model export (registered cameras, image observations, and point tracks)
- **Video frame extraction** (configurable interval + max frames)
- Download estimated poses as JSON
- Download registered COLMAP sparse models as ZIP

## Quick Start

```bash
git clone https://github.com/MalvinMakhuthe13/alignner.git
cd alignner

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. Upload images **or** videos (or both).
3. Adjust frame extraction rate in the sidebar for videos.
4. Click **Align**.
5. Review pairs, components, camera plot, and download poses.

## Current Limitations

- Approximate shared intrinsics
- Pairwise relative pose initialization followed by robust per-component pose-graph optimization
- Approximate shared intrinsics; mixed camera models and EXIF calibration are not yet handled
- Sparse reconstruction only; no dense reconstruction or meshing yet
- Disconnected components have independent local coordinate systems and cannot be merged without additional constraints
- No GPS / control points

For production work use RealityScan, COLMAP, or Meshroom.

## Roadmap

- Incremental structure-from-motion
- Robust per-component pose-graph optimization and reprojection-error bundle adjustment
- Dense reconstruction and meshing integration
- Masking support
- Better video keyframe selection

## License

MIT

## Author

Malvin Makhuthe
