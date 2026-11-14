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
st.sidebar.subheader("Large sets")
st.sidebar.caption("With >40 images the engine automatically switches to nearby + sparse long-range matching so it stays fast.")
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
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(frac, msg):
                progress_bar.progress(min(1.0, float(frac)))
                status_text.caption(msg)

            aligner = ImageAligner(
                feature_detector=detector,
                max_features=max_features,
                match_ratio=match_ratio,
                min_inliers=min_inliers,
            )
            focal = float(focal_override) if focal_override > 0 else None
            result = aligner.align(
                all_paths,
                focal_length=focal,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0)
            status_text.caption("Finished")

        # Persist result so it survives slider changes
        st.session_state["alignment_result"] = result
        st.session_state["alignment_done"] = True

# ── Display results (persists across reruns) ─────────────────
if st.session_state.get("alignment_done") and "alignment_result" in st.session_state:
    result = st.session_state["alignment_result"]

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

    # Match browser (all pairs)
    if result.pairs:
        st.subheader("Browse Matches")
        sorted_pairs = sorted(result.pairs, key=lambda x: -x.inliers)
        pair_options = []
        for p in sorted_pairs:
            name1 = Path(result.features[p.idx1].path).name
            name2 = Path(result.features[p.idx2].path).name
            pair_options.append(f"{name1}  ↔  {name2}  ({p.inliers} inliers)")

        if "match_idx" not in st.session_state:
            st.session_state.match_idx = 0

        col_prev, col_sel, col_next = st.columns([1, 6, 1])
        with col_prev:
            if st.button("← Prev") and st.session_state.match_idx > 0:
                st.session_state.match_idx -= 1
        with col_next:
            if st.button("Next →") and st.session_state.match_idx < len(pair_options) - 1:
                st.session_state.match_idx += 1
        with col_sel:
            selected_pair = st.selectbox(
                "Select a matched pair",
                pair_options,
                index=st.session_state.match_idx,
                key="pair_select",
            )
            st.session_state.match_idx = pair_options.index(selected_pair)

        chosen = sorted_pairs[st.session_state.match_idx]
        f1 = result.features[chosen.idx1]
        f2 = result.features[chosen.idx2]
        if f1.image is not None and f2.image is not None:
            vis = visualize_matches(
                f1.image, f1.keypoints,
                f2.image, f2.keypoints,
                chosen.matches,
                max_draw=100,
            )
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{chosen.inliers} inliers", use_container_width=True)
            st.caption(f"Pair {st.session_state.match_idx + 1} of {len(sorted_pairs)} • Green lines = verified matches")

    # 3D cameras + sparse points
    if result.camera_poses or (result.sparse_points is not None):
        st.subheader("Cameras + Sparse Points")
        fig = go.Figure()

        # Cameras (as larger markers + path)
        if result.camera_poses:
            positions, labels = [], []
            for idx in sorted(result.camera_poses.keys()):
                R, t = result.camera_poses[idx]
                C = -R.T @ t.flatten()
                positions.append(C)
                labels.append(Path(result.features[idx].path).name)
            if positions:
                pos = np.array(positions)
                fig.add_trace(go.Scatter3d(
                    x=pos[:,0], y=pos[:,1], z=pos[:,2],
                    mode="markers+lines",
                    marker=dict(size=4, color="cyan", line=dict(width=1, color="darkblue")),
                    line=dict(color="deepskyblue", width=4),
                    name="Camera path",
                ))

        # Sparse points
        if result.sparse_points is not None and len(result.sparse_points) > 0:
            pts = result.sparse_points
            if len(pts) > 20000:
                idx = np.random.choice(len(pts), 20000, replace=False)
                pts_disp = pts[idx]
                cols = result.point_colors[idx] if result.point_colors is not None else None
            else:
                pts_disp = pts
                cols = result.point_colors

            fig.add_trace(go.Scatter3d(
                x=pts_disp[:,0], y=pts_disp[:,1], z=pts_disp[:,2],
                mode="markers",
                marker=dict(size=1.8, color=cols if cols is not None else "orange", opacity=0.65),
                name="Sparse points",
            ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers",
            marker=dict(size=8, color="red", symbol="diamond"),
            name="Origin",
        ))
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        st.plotly_chart(fig, use_container_width=True)
        npts = 0 if result.sparse_points is None else len(result.sparse_points)
        nposed = len(result.camera_poses)
        st.caption(f"{nposed} cameras posed • {npts} sparse points. Drag to rotate, scroll to zoom.")

    # Image browser
    st.subheader("Browse Images")
    if result.features:
        names = [Path(f.path).name for f in result.features]
        selected = st.selectbox("Select image", names, index=0)
        sel_idx = names.index(selected)
        feat = result.features[sel_idx]
        if feat.image is not None:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.image(cv2.cvtColor(feat.image, cv2.COLOR_BGR2RGB), caption=selected, use_container_width=True)
            with col_b:
                st.write(f"**Size:** {feat.width} × {feat.height}")
                st.write(f"**Features:** {len(feat.keypoints)}")
                posed = "Yes" if sel_idx in result.camera_poses else "No"
                st.write(f"**Pose recovered:** {posed}")
                if sel_idx in result.camera_poses:
                    R, t = result.camera_poses[sel_idx]
                    C = -R.T @ t.flatten()
                    st.write(f"**Camera center:** ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f})")

    # COLMAP export
    if st.button("Export COLMAP dataset"):
        with st.spinner("Writing COLMAP folder..."):
            sample = result.features[0]
            avg = max(sample.width, sample.height)
            focal = avg * 1.2
            K = np.array([[focal, 0, sample.width/2], [0, focal, sample.height/2], [0, 0, 1]])

            export_tmp = Path(tempfile.mkdtemp())
            out_path = str(export_tmp / "colmap_export")
            aligner = ImageAligner()
            aligner.export_colmap(result, out_path, K)

            import shutil
            zip_base = str(export_tmp / "colmap_export")
            shutil.make_archive(zip_base, "zip", out_path)
            zip_path = zip_base + ".zip"

            with open(zip_path, "rb") as f:
                st.download_button(
                    "Download COLMAP dataset (.zip)",
                    data=f.read(),
                    file_name="colmap_export.zip",
                    mime="application/zip",
                )
            st.success("COLMAP export ready (cameras.txt, images.txt, points3D.txt + images/)")

    # Export JSON
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

    if st.button("Clear results"):
        st.session_state.pop("alignment_result", None)
        st.session_state.pop("alignment_done", None)
        st.rerun()

st.markdown("---")
st.caption("Alignner prototype • Video frames are extracted locally • Full SfM / COLMAP export planned")
