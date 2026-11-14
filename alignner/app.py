"""
Alignner App - Image Alignment UI
Similar in spirit to the Alignment step in RealityScan / RealityCapture.
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tempfile
import plotly.graph_objects as go
from src.aligner import ImageAligner, visualize_matches

st.set_page_config(
    page_title="Alignner - Image Alignment",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Alignner")
st.markdown("""
**Image Alignment for Photogrammetry**  
Inspired by RealityScan's alignment pipeline. Upload overlapping photos of a scene or object to estimate relative camera poses and connectivity.
""")

# Sidebar settings
st.sidebar.header("Alignment Settings")
detector = st.sidebar.selectbox("Feature Detector", ["orb", "sift"], index=0)
max_features = st.sidebar.slider("Max Features per Image", 1000, 10000, 4000, step=500)
match_ratio = st.sidebar.slider("Lowe's Ratio", 0.5, 0.95, 0.75, step=0.05)
min_inliers = st.sidebar.slider("Min Inliers for Pair", 10, 100, 20)
focal_override = st.sidebar.number_input(
    "Focal Length (px, 0 = auto)", value=0.0, min_value=0.0, step=10.0
)

uploaded = st.file_uploader(
    "Upload images (JPG/PNG)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded and len(uploaded) >= 2:
    if st.button("🚀 Align Images", type="primary"):
        with st.spinner("Extracting features and aligning..."):
            # Save uploads to temp
            tmpdir = tempfile.mkdtemp()
            paths = []
            for f in uploaded:
                p = Path(tmpdir) / f.name
                p.write_bytes(f.getvalue())
                paths.append(str(p))

            aligner = ImageAligner(
                feature_detector=detector,
                max_features=max_features,
                match_ratio=match_ratio,
                min_inliers=min_inliers,
            )
            focal = float(focal_override) if focal_override > 0 else None
            result = aligner.align(paths, focal_length=focal)

        st.success("Alignment finished!")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images", len(result.features))
        col2.metric("Verified Pairs", len(result.pairs))
        col3.metric("Components", len(result.components))
        largest = max((len(c) for c in result.components), default=0)
        col4.metric("Largest Component", largest)

        # Components
        st.subheader("Connected Components")
        for i, comp in enumerate(result.components):
            names = [Path(result.features[j].path).name for j in comp]
            st.write(f"**Component {i+1}** ({len(comp)} images): {', '.join(names)}")

        # Pair table
        if result.pairs:
            st.subheader("Verified Image Pairs")
            pair_data = []
            for p in sorted(result.pairs, key=lambda x: -x.inliers):
                name1 = Path(result.features[p.idx1].path).name
                name2 = Path(result.features[p.idx2].path).name
                pair_data.append({
                    "Image A": name1,
                    "Image B": name2,
                    "Inliers": p.inliers,
                    "Has Pose": p.R is not None,
                })
            st.dataframe(pair_data, use_container_width=True)

        # Camera visualization (simple 3D)
        if result.camera_poses:
            st.subheader("Estimated Camera Positions (relative)")
            fig = go.Figure()

            # Camera centers (approximate from R,t — note: t is translation of second relative to first)
            positions = []
            labels = []
            for idx, (R, t) in result.camera_poses.items():
                # Camera center in world (approx): C = -R^T @ t
                C = -R.T @ t.flatten()
                positions.append(C)
                labels.append(Path(result.features[idx].path).name)

            if positions:
                pos = np.array(positions)
                fig.add_trace(go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                    mode="markers+text",
                    text=labels,
                    marker=dict(size=8, color="cyan"),
                    name="Cameras",
                ))

                # Origin
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name="Origin (first image)",
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode="data",
                    ),
                    height=500,
                    margin=dict(l=0, r=0, b=0, t=30),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Note: Positions are relative and approximate (pairwise poses). Full bundle adjustment not yet implemented.")

        # Show sample matches
        if result.pairs:
            st.subheader("Sample Matches")
            best = max(result.pairs, key=lambda p: p.inliers)
            f1 = result.features[best.idx1]
            f2 = result.features[best.idx2]
            if f1.image is not None and f2.image is not None:
                vis = visualize_matches(
                    f1.image, f1.keypoints,
                    f2.image, f2.keypoints,
                    best.matches,
                    max_draw=40,
                )
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Best pair: {best.inliers} inliers")

        # Feature counts
        st.subheader("Feature Detection")
        feat_rows = [
            {"Image": Path(f.path).name, "Features": len(f.keypoints), "Size": f"{f.width}x{f.height}"}
            for f in result.features
        ]
        st.dataframe(feat_rows, use_container_width=True)

else:
    st.info("Upload at least 2 overlapping images to begin alignment.")

st.markdown("---")
st.caption("Alignner is a learning / prototyping tool. For production photogrammetry use RealityScan, COLMAP, or Meshroom.")
