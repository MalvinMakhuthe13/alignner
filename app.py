"""
Alignner — photo & video alignment for photogrammetry and Gaussian splatting.

Upload photos or a video, align them (real incremental SfM with bundle
adjustment), check the cameras and sparse points, then download a COLMAP
folder that Brush, gsplat or Nerfstudio train on directly.
"""
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.image_selector import IntelligentImageSelector
from src.sfm import IncrementalSfM, camera_center, export_colmap

st.set_page_config(page_title="Alignner", page_icon="📷", layout="wide")
st.title("📷 Alignner")
st.markdown("""
**Alignment for photogrammetry and Gaussian splatting.**
Upload photos or videos → **Align** → download a COLMAP folder for Brush.
Original uploads are never modified.
""")

st.sidebar.header("Alignment")
detector = st.sidebar.selectbox("Features", ["sift", "orb"], index=0,
                                help="SIFT is slower but far more accurate. Use ORB only for quick previews.")
max_features = st.sidebar.slider("Features per image", 2000, 16000, 8000, step=1000)
match_ratio = st.sidebar.slider("Match strictness (Lowe ratio)", 0.6, 0.9, 0.8, 0.05,
                                help="Lower = stricter, fewer but cleaner matches.")
min_inliers = st.sidebar.slider("Min verified matches per pair", 15, 100, 30)
focal_override = st.sidebar.number_input("Focal length (px, 0 = from EXIF / auto)", 0.0, step=50.0,
                                         help="Leave at 0. Focal is read from EXIF and then refined.")

st.sidebar.markdown("---")
st.sidebar.subheader("Large sets")
exhaustive_limit = st.sidebar.slider("Match every pair up to N images", 20, 150, 60,
                                     help="Above this, nearby images plus visual retrieval are matched.")
window = st.sidebar.slider("Sequential neighbours", 4, 30, 12)
fps_extract = st.sidebar.slider("Video: keep every Nth frame", 1, 30, 10)
max_frames = st.sidebar.slider("Video: max frames", 10, 400, 150)

st.sidebar.markdown("---")
st.sidebar.subheader("Intelligent selection")
use_selection = st.sidebar.checkbox("Remove strong near-duplicates", value=False,
    help="Only very strong near-duplicates are excluded. Blurry/low-feature images are flagged for review, not discarded.")
review_threshold = st.sidebar.slider("Flag blur below (Laplacian variance)", 5.0, 80.0, 18.0, 1.0)

tab1, tab2 = st.tabs(["Images", "Videos"])
image_files = tab1.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"], accept_multiple_files=True)
video_files = tab2.file_uploader("Upload videos", type=["mp4", "mov", "avi", "mkv", "webm"], accept_multiple_files=True)


def extract_frames(video_bytes: bytes, filename: str, every_n: int, max_n: int, tmpdir: Path):
    safe_name = Path(filename).name
    video_path = tmpdir / safe_name
    video_path.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {filename}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    st.info(f"{filename}: {total} frames @ {fps:.1f} fps → every {every_n} frames (max {max_n})")
    frames, idx, saved = [], 0, 0
    while saved < max_n:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out_path = tmpdir / f"{Path(filename).stem}_frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frames.append(str(out_path))
            saved += 1
        idx += 1
    cap.release()
    return frames


def zip_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder.parent))
    return buf.getvalue()


if st.button("🚀 Align", type="primary"):
    tmpdir = Path(tempfile.mkdtemp(prefix="alignner_"))
    src_dir = tmpdir / "input"
    src_dir.mkdir()
    all_paths = []
    with st.spinner("Preparing images and extracting video frames..."):
        for f in image_files or []:
            p = src_dir / Path(f.name).name
            p.write_bytes(f.getvalue())
            all_paths.append(str(p))
        for vf in video_files or []:
            all_paths.extend(extract_frames(vf.getvalue(), vf.name, fps_extract, max_frames, src_dir))

    if len(all_paths) < 2:
        st.error("Need at least 2 images/frames to align.")
    else:
        original_count = len(all_paths)
        if use_selection:
            with st.spinner(f"Analysing {original_count} images for quality and near-duplicates..."):
                selection = IntelligentImageSelector(min_sharpness_for_review=review_threshold).analyze(all_paths)
                all_paths = selection.selected_paths
            st.session_state["selection_result"] = selection
            st.session_state["selection_csv"] = str(tmpdir / "selection_manifest.csv")
            selection.to_csv(st.session_state["selection_csv"])
            st.info(f"Selection retained {selection.selected_count}/{original_count} images.")

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(frac, msg):
            progress_bar.progress(min(1.0, float(frac)))
            status_text.caption(msg)

        sfm = IncrementalSfM(detector=detector, max_features=max_features, ratio=match_ratio,
                             min_inliers=min_inliers, exhaustive_limit=exhaustive_limit,
                             sequential_window=window,
                             focal_px=float(focal_override) if focal_override > 0 else None,
                             progress=update_progress)
        try:
            with st.spinner(f"Aligning {len(all_paths)} images..."):
                rec = sfm.run(all_paths)
            out_dir = tmpdir / "colmap"
            info = export_colmap(rec, str(out_dir))
            st.session_state["rec"] = rec
            st.session_state["export"] = (str(out_dir), info)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("rec", None)

if "selection_result" in st.session_state:
    selection = st.session_state["selection_result"]
    with st.expander("Intelligent selection report"):
        rows = [{"Image": x.filename, "Decision": x.decision, "Quality": x.quality_score,
                 "Sharpness": x.sharpness, "Features": x.feature_count, "Reason": x.reason}
                for x in selection.assessments]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        csv_path = st.session_state.get("selection_csv")
        if csv_path and Path(csv_path).exists():
            st.download_button("Download selection manifest (CSV)", Path(csv_path).read_bytes(),
                               file_name="alignner_selection_manifest.csv", mime="text/csv")

if "rec" in st.session_state:
    rec = st.session_state["rec"]
    out_dir, info = st.session_state["export"]
    s = rec.stats()
    st.success(f"Aligned {s['registered']} of {s['images']} images · {s['points']:,} sparse points")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cameras placed", f"{s['registered']}/{s['images']}")
    c2.metric("Sparse points", f"{s['points']:,}")
    c3.metric("Mean reprojection error", f"{s['mean_reprojection_error_px'] or 0:.2f} px",
              help="Under 1 px is excellent; under 2 px is good for splatting.")
    c4.metric("Mean track length", f"{s['mean_track_length'] or 0:.1f} views")
    cams = ", ".join(f"{c.width}×{c.height} f={c.f:.0f}px k={c.k:+.3f}" for c in rec.cameras.values())
    st.caption(f"Lens: {cams}" + (" · images were undistorted for training" if info["undistorted"] else ""))

    st.download_button("⬇️ Download COLMAP folder for Brush (.zip)", zip_folder(Path(out_dir)),
                       file_name="alignner_colmap.zip", mime="application/zip", type="primary")
    st.caption("Unzip and open the `colmap` folder in Brush (it contains `images/` and `sparse/0/`).")

    st.subheader("Cameras + sparse points")
    fig = go.Figure()
    reg = rec.registered_views
    if reg:
        pos = np.array([camera_center(v) for v in reg])
        fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers",
                                   marker=dict(size=4, color="#c8ff00"), name="Cameras",
                                   text=[Path(v.path).name for v in reg]))
    pts = rec.points
    if pts:
        ix = np.random.default_rng(0).choice(len(pts), min(40000, len(pts)), replace=False)
        P = np.array([pts[i].xyz for i in ix])
        C = np.array([pts[i].color for i in ix])
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                   marker=dict(size=1.6, color=[f"rgb({r},{g},{b})" for r, g, b in C]),
                                   name="Sparse points"))
    fig.update_layout(scene=dict(aspectmode="data"), height=700, margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)

    unplaced = [Path(v.path).name for v in rec.views if not v.registered]
    if unplaced:
        st.warning(f"{len(unplaced)} image(s) could not be placed: {', '.join(unplaced[:20])}"
                   f"{'…' if len(unplaced) > 20 else ''}. They usually lack overlap with the rest — "
                   "add photos in between or re-shoot with 60–80 % overlap.")
    with st.expander("Alignment log"):
        st.code("\n".join(rec.log) or "—")

    if st.button("Clear results"):
        shutil.rmtree(Path(out_dir).parent, ignore_errors=True)
        for key in ("rec", "export", "selection_result", "selection_csv"):
            st.session_state.pop(key, None)
        st.rerun()

st.markdown("---")
st.caption("Alignner · incremental SfM + bundle adjustment on the CPU · COLMAP export for Brush / gsplat / Nerfstudio")
