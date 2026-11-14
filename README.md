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
- **Video frame extraction** (configurable interval + max frames)
- Download estimated poses as JSON

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
- Pairwise relative poses (no full incremental SfM / bundle adjustment yet)
- No dense reconstruction or meshing
- No GPS / control points

For production work use RealityScan, COLMAP, or Meshroom.

## Roadmap

- Incremental structure-from-motion
- Bundle adjustment
- COLMAP / RealityScan export
- Masking support
- Better video keyframe selection

## License

MIT

## Author

Malvin Makhuthe
