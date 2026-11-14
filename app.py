"""
Alignner App - Image & Video Alignment UI
Inspired by RealityScan / RealityCapture alignment.
Supports images + video frame extraction.
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
import plotly.graph_objects as go
from src.aligner import ImageAligner, visualize_matches

st.set_page_config(
    page_title="Alignner",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Alignner")
st.markdown("""
**Image & Video Alignment for Photogrammetry**  
Upload photos **or videos**. Videos are automatically sampled into frames, then aligned like RealityScan.
""")

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Settings")

detector = st.sidebar.selectbox("Feature Detector", ["orb", "sift"], index=0)
max_features = st.sidebar.slider("Max Features / Image", 1000, 12000, 5000, step=500)
match_ratio = st.sidebar.slider("Lowe's Ratio", 0.5, 0.9, 0.75, 0.05)
min_inliers = st.sidebar.slider("Min Inliers", 8, 80, 15)
focal_override = st.sidebar.number_input("Focal Length (px, 0=auto)", 0.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Video Frame Extraction")
fps_extract = st.sidebar.slider("Extract every Nth frame (or FPS)", 1, 30, 5)
max_frames = st.sidebar.slider("Max frames per video", 10, 200, 60)

# ── Input ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Images", "Videos"])

image_files = tab1.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

video_files = tab2.file_uploader(
    "Upload videos",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    accept_multiple_files=True,
)

def extract_frames(video_bytes: bytes, filename: str, every_n: int, max_n: int, tmpdir: Path):
    """Extract frames from video bytes."""
    video_path = tmpdir / filename
    video_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {filename}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    st.info(f"{filename}: {total} frames @ {fps:.1f} fps → extracting ~every {every_n} frames")

    frames = []
    idx = 0
    saved = 0
    while saved < max_n:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out_path = tmpdir / f"{Path(filename).stem}_frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frames.append(str(out_path))
            saved += 1
        idx += 1
    cap.release()
    return frames

# ── Run ──────────────────────────────────────────────────
all_paths = []
tmpdir = Path(tempfile.mkdtemp())

if st.button("🚀 Align", type="primary"):
    with st.spinner("Preparing images..."):
        # Images
        if image_files:
            for f in image_files:
                p = tmpdir / f.name
                p.write_bytes(f.getvalue())
                all_paths.append(str(p))

        # Videos → frames
        if video_files:
            for vf in video_files:
                frames = extract_frames(vf.getvalue(), vf.name, fps_extract, max_frames, tmpdir)
                all_paths.extend(frames)

    if len(all_paths) < 2:
        st.error("Need at least 2 images/frames to align.")
    else:
        with st.spinner(f"Aligning {len(all_paths)} images/frames..."):
            aligner = ImageAligner(
                feature_detector=detector,
                max_features=max_features,
                match_ratio=match_ratio,
                min_inliers=min_inliers,
            )
            focal = float(focal_override) if focal_override > 0 else None
            result = aligner.align(all_paths, focal_length=focal)

        st.success(f"Done — {len(result.features)} images, {len(result.pairs)} verified pairs")

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images / Frames", len(result.features))
        c2.metric("Verified Pairs", len(result.pairs))
        c3.metric("Components", len(result.components))
        c4.metric("Largest Component", max((len(c) for c in result.components), default=0))

        # Components
        st.subheader("Connected Components")
        for i, comp in enumerate(result.components):
            names = [Path(result.features[j].path).name for j in comp]
            st.write(f"**Component {i+1}** ({len(comp)}): {', '.join(names[:8])}{'...' if len(names)>8 else ''}")

        # Pairs table
        if result.pairs:
            st.subheader("Verified Pairs")
            rows = []
            for p in sorted(result.pairs, key=lambda x: -x.inliers)[:50]:
                rows.append({
                    "A": Path(result.features[p.idx1].path).name,
                    "B": Path(result.features[p.idx2].path).name,
                    "Inliers": p.inliers,
                    "Pose": "✓" if p.R is not None else "—",
                })
            st.dataframe(rows, use_container_width=True)

        # 3D cameras
        if result.camera_poses:
            st.subheader("Estimated Camera Positions")
            fig = go.Figure()
            positions, labels = [], []
            for idx, (R, t) in result.camera_poses.items():
                C = -R.T @ t.flatten()
                positions.append(C)
                labels.append(Path(result.features[idx].path).name)

            if positions:
                pos = np.array(positions)
                fig.add_trace(go.Scatter3d(
                    x=pos[:,0], y=pos[:,1], z=pos[:,2],
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=7, color="cyan"),
                    name="Cameras",
                ))
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="diamond"),
                    name="Origin",
                ))
                fig.update_layout(
                    scene=dict(aspectmode="data"),
                    height=520,
                    margin=dict(l=0,r=0,b=0,t=30),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Relative poses (pairwise). Full bundle adjustment coming later.")

        # Sample matches
        if result.pairs:
            st.subheader("Best Matches")
            best = max(result.pairs, key=lambda p: p.inliers)
            f1 = result.features[best.idx1]
            f2 = result.features[best.idx2]
            if f1.image is not None and f2.image is not None:
                vis = visualize_matches(f1.image, f1.keypoints, f2.image, f2.keypoints, best.matches, 50)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{best.inliers} inliers")

        # Export
        if result.camera_poses:
            export = {
                "num_images": len(result.features),
                "components": result.components,
                "poses": {
                    str(i): {
                        "R": R.tolist(),
                        "t": t.flatten().tolist(),
                        "name": Path(result.features[i].path).name,
                    }
                    for i, (R, t) in result.camera_poses.items()
                }
            }
            st.download_button(
                "Download poses (JSON)",
                data=json.dumps(export, indent=2),
                file_name="alignner_poses.json",
                mime="application/json",
            )

st.markdown("---")
st.caption("Alignner prototype • Video frames are extracted locally • Full SfM / COLMAP export planned")
