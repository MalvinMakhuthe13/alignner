"""
Alignner App - Image & Video Alignment with conservative intelligent selection.
"""
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import tempfile
import json
import io
import zipfile
import plotly.graph_objects as go
from src.aligner import ImageAligner, visualize_matches
from src.image_selector import IntelligentImageSelector
from src.image_contribution import score_image_contributions
from src.viewer import render_ply_viewer

st.set_page_config(page_title="Alignner", page_icon="📷", layout="wide")
st.title("📷 Alignner")
st.markdown("""
**Image & Video Alignment for Photogrammetry**
Upload photos or videos. Optionally run a conservative image-selection preflight
before alignment. Original uploads are never modified.
""")

with st.expander("🧊 Open 3D Viewer", expanded=False):
    render_ply_viewer()

st.sidebar.header("Settings")
detector = st.sidebar.selectbox("Feature Detector", ["orb", "sift"], index=0)
max_features = st.sidebar.slider("Max Features / Image", 1000, 12000, 5000, step=500)
match_ratio = st.sidebar.slider("Lowe's Ratio", 0.5, 0.9, 0.75, 0.05)
min_inliers = st.sidebar.slider("Min Inliers", 8, 80, 15)
min_support_ratio = st.sidebar.slider(
    "Min track support ratio (experimental)",
    0.40, 0.70, 0.70, 0.05,
    help="Default 0.70 preserves production behaviour. Lower values are experimental and only affect the next Align run."
)
focal_override = st.sidebar.number_input("Focal Length (px, 0=auto)", 0.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Large sets")
st.sidebar.caption("The aligner uses nearby + sparse long-range matching for larger sets.")
fps_extract = st.sidebar.slider("Extract every Nth frame (or FPS)", 1, 30, 5)
max_frames = st.sidebar.slider("Max frames per video", 10, 200, 60)

st.sidebar.markdown("---")
st.sidebar.subheader("Intelligent selection")
use_selection = st.sidebar.checkbox("Enable conservative duplicate removal", value=False,
    help="Only very strong near-duplicates are excluded. Blurry/low-feature images are flagged for review, not discarded.")
review_threshold = st.sidebar.slider("Flag blur below (Laplacian variance)", 5.0, 80.0, 18.0, 1.0)

tab1, tab2 = st.tabs(["Images", "Videos"])
image_files = tab1.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
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
    st.info(f"{filename}: {total} frames @ {fps:.1f} fps → sampling every {every_n} frames (max {max_n})")
    frames, idx, saved = [], 0, 0
    while saved < max_n:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out_path = tmpdir / f"{Path(filename).stem}_frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            frames.append(str(out_path))
            saved += 1
        idx += 1
    cap.release()
    return frames

if st.button("🚀 Align", type="primary"):
    tmpdir = Path(tempfile.mkdtemp(prefix="alignner_"))
    all_paths = []
    with st.spinner("Preparing images and extracting video frames..."):
        for f in image_files or []:
            p = tmpdir / Path(f.name).name
            p.write_bytes(f.getvalue())
            all_paths.append(str(p))
        for vf in video_files or []:
            all_paths.extend(extract_frames(vf.getvalue(), vf.name, fps_extract, max_frames, tmpdir))

    if len(all_paths) < 2:
        st.error("Need at least 2 images/frames to align.")
    else:
        original_count = len(all_paths)
        selection = None
        if use_selection:
            with st.spinner(f"Analysing {original_count} images for quality and near-duplicates..."):
                selector = IntelligentImageSelector(min_sharpness_for_review=review_threshold)
                selection = selector.analyze(all_paths)
                all_paths = selection.selected_paths
            st.session_state["selection_result"] = selection
            st.session_state["selection_csv"] = str(tmpdir / "selection_manifest.csv")
            selection.to_csv(st.session_state["selection_csv"])
            st.info(f"Selection retained {selection.selected_count}/{original_count} images; "
                    f"{len(selection.excluded_paths)} strong near-duplicates excluded. "
                    "Review-flagged images remain included.")

        if len(all_paths) < 2:
            st.error("Fewer than 2 images remain. Disable selection or add more images.")
        else:
            with st.spinner(f"Aligning {len(all_paths)} images/frames..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                stage_status = st.empty()
                stage_metrics = st.empty()
                triangulation_bar = st.progress(0)
                export_notice = st.empty()
                live_preview_title = st.empty()
                live_preview_metrics = st.empty()
                live_preview_chart = st.empty()

                # Keep each run separate so Brush can continue reading
                # an earlier export without files being overwritten.
                run_root = (
                    Path.cwd()
                    / "Alignner_Output"
                    / f"run_{datetime.now():%Y%m%d_%H%M%S}"
                )
                images_dir = run_root / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                def update_progress(frac, msg):
                    progress_bar.progress(min(1.0, float(frac)))
                    status_text.caption(msg)

                def render_live_preview(event):
                    poses = event.get("camera_poses")
                    points = event.get("sparse_points")
                    point_colors = event.get("point_colors")
                    tracks = event.get("point_tracks") or []
                    components = event.get("components") or []

                    if poses is None:
                        poses = live_preview_state.get("camera_poses", {})
                    else:
                        live_preview_state["camera_poses"] = poses
                    if components:
                        live_preview_state["components"] = components
                    else:
                        components = live_preview_state.get("components", [])
                    if points is None:
                        points = live_preview_state.get("sparse_points")
                    else:
                        live_preview_state["sparse_points"] = points
                        live_preview_state["point_colors"] = point_colors
                        live_preview_state["point_tracks"] = tracks
                    if points is None:
                        points = np.empty((0, 3), dtype=np.float64)
                    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

                    live_poses = live_preview_state.get("camera_poses", {})
                    live_preview_title.subheader("Live reconstruction preview")
                    live_preview_metrics.caption(
                        f"{len(live_poses)} registered cameras • "
                        f"{len(points):,} triangulated points"
                    )
                    if not live_poses and not len(points):
                        return

                    fig = go.Figure()
                    palette_live = [
                        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                        "#bcbd22", "#17becf",
                    ]
                    component_of_live = {
                        image_idx: ci
                        for ci, component in enumerate(components)
                        for image_idx in component
                    }
                    for idx, (R, t) in sorted(live_poses.items()):
                        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
                        t = np.asarray(t, dtype=np.float64).reshape(3)
                        centre = -R.T @ t
                        ci = component_of_live.get(idx, 0)
                        color = palette_live[ci % len(palette_live)]
                        fig.add_trace(go.Scatter3d(
                            x=[centre[0]], y=[centre[1]], z=[centre[2]],
                            mode="markers",
                            marker=dict(size=5, color=color),
                            name=f"Camera {idx + 1}",
                            showlegend=False,
                            hovertemplate=f"Camera {idx + 1}<extra></extra>",
                        ))
                        forward = R.T[:, 2]
                        right = R.T[:, 0]
                        up = -R.T[:, 1]
                        scale = 0.15
                        base = centre + forward * scale
                        hw, hh = scale * 0.6, scale * 0.45
                        corners = [
                            base + right * hw + up * hh,
                            base - right * hw + up * hh,
                            base - right * hw - up * hh,
                            base + right * hw - up * hh,
                        ]
                        for corner in corners:
                            fig.add_trace(go.Scatter3d(
                                x=[centre[0], corner[0]],
                                y=[centre[1], corner[1]],
                                z=[centre[2], corner[2]],
                                mode="lines",
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                        closed = corners + [corners[0]]
                        fig.add_trace(go.Scatter3d(
                            x=[p[0] for p in closed],
                            y=[p[1] for p in closed],
                            z=[p[2] for p in closed],
                            mode="lines",
                            line=dict(color=color, width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        ))

                    if len(points):
                        display_points = points
                        display_colors = live_preview_state.get("point_colors")
                        sample_indices = np.arange(len(display_points))
                        if len(display_points) > 12000:
                            # Deterministic preview sampling avoids visual flicker.
                            sample_indices = np.linspace(
                                0, len(display_points) - 1, 12000, dtype=int
                            )
                            display_points = display_points[sample_indices]
                        if display_colors is not None:
                            display_colors = np.asarray(display_colors)
                            if (
                                display_colors.ndim == 2
                                and len(display_colors) == len(points)
                                and display_colors.shape[1] >= 3
                            ):
                                sampled_rgb = display_colors[sample_indices, :3]
                                marker_colors = [
                                    f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
                                    for rgb in sampled_rgb
                                ]
                            else:
                                marker_colors = "#d9e6f2"
                        else:
                            marker_colors = "#d9e6f2"
                        fig.add_trace(go.Scatter3d(
                            x=display_points[:, 0],
                            y=display_points[:, 1],
                            z=display_points[:, 2],
                            mode="markers",
                            marker=dict(size=1.8, color=marker_colors, opacity=0.75),
                            name="Live sparse points",
                        ))
                    fig.update_layout(
                        scene=dict(
                            aspectmode="data",
                            xaxis_title="X",
                            yaxis_title="Y",
                            zaxis_title="Z",
                        ),
                        height=560,
                        margin=dict(l=0, r=0, b=0, t=20),
                        showlegend=False,
                    )
                    live_preview_chart.plotly_chart(
                        fig, use_container_width=True
                    )

                triangulation_diagnostics = {}

                live_preview_state = {
                    "camera_poses": {},
                    "components": [],
                    "sparse_points": None,
                    "point_colors": None,
                    "point_tracks": [],
                }

                def update_stage(event):
                    stage = event.get("stage", "pipeline")
                    state = event.get("status", "running")
                    message = event.get("message", "")
                    if stage == "triangulation_diagnostics":
                        triangulation_diagnostics.update(
                            event.get("diagnostics", {})
                        )
                    stage_status.caption(
                        f"**{stage.replace('_', ' ').title()}** — {state}: {message}"
                    )

                    if stage == "live_preview":
                        render_live_preview(event)

                    if stage == "triangulation":
                        fraction = float(event.get("progress", 0.0))
                        triangulation_bar.progress(
                            min(1.0, max(0.0, fraction))
                        )
                        stage_metrics.caption(
                            "Tracks: "
                            f"{event.get('processed_tracks', 0):,} / "
                            f"{event.get('total_tracks', 0):,}  ?  "
                            f"Accepted points: "
                            f"{event.get('accepted_points', 0):,}  ?  "
                            f"Rejected tracks: "
                            f"{event.get('rejected_tracks', 0):,}"
                        )

                    if stage == "initial_export" and state == "completed":
                        export_notice.success(
                            "Initial COLMAP model is ready for Brush: "
                            f"{event.get('export_path', '')}"
                        )

                aligner = ImageAligner(
                    feature_detector=detector,
                    max_features=max_features,
                    match_ratio=match_ratio,
                    min_inliers=min_inliers,
                )

                def export_initial_model(snapshot):
                    if snapshot.camera_matrix is None:
                        raise RuntimeError(
                            "Initial export failed: camera matrix is missing."
                        )

                    sparse_dir = (
                        run_root
                        / "exports"
                        / "01_initial"
                        / "sparse"
                        / "0"
                    )

                    exported_path = aligner.export_colmap(
                        snapshot,
                        str(sparse_dir),
                        snapshot.camera_matrix,
                        images_dir=str(images_dir),
                    )

                    required_files = (
                        sparse_dir / "cameras.txt",
                        sparse_dir / "images.txt",
                        sparse_dir / "points3D.txt",
                    )
                    missing = [
                        str(file) for file in required_files
                        if not file.is_file() or file.stat().st_size == 0
                    ]
                    if missing:
                        raise RuntimeError(
                            "Initial COLMAP export is incomplete: "
                            + ", ".join(missing)
                        )

                    st.session_state["initial_colmap_path"] = exported_path
                    st.session_state["alignner_output_dir"] = str(run_root)
                    return exported_path

                focal = float(focal_override) if focal_override > 0 else None

                result = aligner.align(
                    all_paths,
                    focal_length=focal,
                    progress_callback=update_progress,
                    stage_callback=update_stage,
                    early_export_callback=export_initial_model,
                    min_support_ratio=float(min_support_ratio),
                )

                # Save and display triangulation diagnostics.
                if triangulation_diagnostics:
                    diagnostics_dir = run_root / "diagnostics"
                    diagnostics_dir.mkdir(parents=True, exist_ok=True)
                    diagnostics_path = (
                        diagnostics_dir / "triangulation_diagnostics.json"
                    )
                    with diagnostics_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            triangulation_diagnostics,
                            f,
                            indent=2,
                            default=lambda value: (
                                value.item() if hasattr(value, "item")
                                else str(value)
                            ),
                        )

                    st.session_state["triangulation_diagnostics"] = (
                        triangulation_diagnostics
                    )
                    st.subheader("Triangulation Diagnostics")
                    d = triangulation_diagnostics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Feature tracks", f"{d.get('total_tracks', 0):,}")
                    m2.metric("Accepted points", f"{d.get('accepted_points', 0):,}")
                    m3.metric("Rejected tracks", f"{d.get('rejected_tracks', 0):,}")
                    m4.metric("Rejected observations", f"{d.get('rejected_observations', 0):,}")
                    with st.expander("Full triangulation diagnostic report", expanded=True):
                        st.json(triangulation_diagnostics)
                    st.caption(f"Diagnostic JSON saved to: {diagnostics_path}")
                else:
                    st.warning(
                        "No triangulation diagnostics event was received. "
                        "The reconstruction result is otherwise unchanged."
                    )

                # Save the refined result separately after bundle adjustment.
                refined_path = None
                if (
                    result.camera_matrix is not None
                    and result.sparse_points is not None
                    and result.point_tracks
                    and result.camera_poses
                ):
                    refined_dir = (
                        run_root
                        / "exports"
                        / "02_refined"
                        / "sparse"
                        / "0"
                    )
                    refined_path = aligner.export_colmap(
                        result,
                        str(refined_dir),
                        result.camera_matrix,
                        images_dir=str(images_dir),
                    )
                    st.session_state["refined_colmap_path"] = refined_path

                st.session_state["alignner_output_dir"] = str(run_root)
                progress_bar.progress(1.0)
                status_text.caption("Alignment and export finished")

                if refined_path:
                    st.success(
                        f"Refined COLMAP model exported: {refined_path}"
                    )

            st.session_state.pop("contribution_report", None)
            st.session_state["alignment_result"] = result
            st.session_state["alignment_done"] = True

if "selection_result" in st.session_state:
    selection = st.session_state["selection_result"]
    st.subheader("Intelligent selection report")
    a, b, c = st.columns(3)
    a.metric("Input images", selection.input_count)
    b.metric("Retained", selection.selected_count)
    c.metric("Excluded duplicates", len(selection.excluded_paths))
    st.caption("Review items are retained. Only strong near-duplicates are automatically excluded.")
    rows = [{
        "Image": x.filename, "Decision": x.decision, "Quality": x.quality_score,
        "Sharpness": x.sharpness, "Features": x.feature_count, "Reason": x.reason
    } for x in selection.assessments]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    csv_path = st.session_state.get("selection_csv")
    if csv_path and Path(csv_path).exists():
        st.download_button("Download selection manifest (CSV)", Path(csv_path).read_bytes(),
                           file_name="alignner_selection_manifest.csv", mime="text/csv")

if st.session_state.get("alignment_done") and "alignment_result" in st.session_state:
    result = st.session_state["alignment_result"]
    st.success(f"Done — {len(result.features)} images, {len(result.pairs)} verified pairs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images / Frames", len(result.features))
    c2.metric("Verified Pairs", len(result.pairs))
    c3.metric("Components", len(result.components))
    c4.metric("Largest Component", max((len(c) for c in result.components), default=0))

    # Post-alignment contribution analysis. This is diagnostic only:
    # it does not remove images or change the reconstruction.
    st.subheader("Image Contribution Report")
    st.caption(
        "Ranks images using verified geometric support, multi-view tracks, "
        "per-image feature coverage, reprojection quality and parallax. "
        "Low scores are review flags, not deletion instructions."
    )

    if "contribution_report" not in st.session_state:
        with st.spinner("Evaluating image contributions..."):
            st.session_state["contribution_report"] = score_image_contributions(result)

    contribution_report = st.session_state["contribution_report"]

    if contribution_report.warnings:
        for warning in contribution_report.warnings:
            st.warning(warning)

    ranked_images = sorted(
        contribution_report.images,
        key=lambda item: item.contribution_score,
        reverse=True,
    )

    high_count = sum(x.assessment == "high contribution" for x in ranked_images)
    review_count = sum(
        x.assessment in ("needs review", "possible redundancy ? review before excluding")
        for x in ranked_images
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Images evaluated", len(ranked_images))
    m2.metric("High contribution", high_count)
    m3.metric("Needs review", review_count)

    contribution_rows = []
    for rank, item in enumerate(ranked_images, start=1):
        feature = result.features[item.image_index]
        reprojection = (
            round(item.mean_reprojection_error_px, 3)
            if item.mean_reprojection_error_px is not None else None
        )
        parallax = (
            round(item.mean_parallax_deg, 2)
            if item.mean_parallax_deg is not None else None
        )

        contribution_rows.append({
            "Rank": rank,
            "Image": Path(feature.path).name,
            "Score": round(item.contribution_score, 1),
            "Assessment": item.assessment,
            "Verified pairs": item.pair_count,
            "Pose inliers": item.pose_inliers,
            "3D point tracks": item.point_track_count,
            "Image coverage": round(item.grid_coverage * 100, 1),
            "Mean reprojection (px)": reprojection,
            "Mean parallax (?)": parallax,
            "Geometric inlier ratio": round(item.geometric_inlier_ratio, 3),
            "Pose retention ratio": round(item.pose_retention_ratio, 3),
            "Explanation": "; ".join(item.notes),
        })

    if contribution_rows:
        st.dataframe(
            contribution_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No image contribution metrics were available.")

    st.info(
        "Important: contribution scores are relative rankings within connected "
        "components. They are not probabilities, and images must not be removed "
        "solely because of a low score."
    )


    if result.camera_poses:
        pose_export = {
            "num_images": len(result.features),
            "components": [
                [int(idx) for idx in component]
                for component in result.components
            ],
            "poses": {
                str(int(idx)): {
                    "R": R.tolist(),
                    "t": t.flatten().tolist(),
                    "name": Path(result.features[idx].path).name,
                }
                for idx, (R, t) in result.camera_poses.items()
                if 0 <= idx < len(result.features)
            },
        }

        st.download_button(
            "Download poses (JSON)",
            data=json.dumps(pose_export, indent=2),
            file_name="alignner_poses.json",
            mime="application/json",
            key="download_poses_json",
        )

    if result.camera_poses:
        st.subheader("Export")
        st.caption("Download the registered sparse reconstruction as a COLMAP text model. Camera intrinsics are currently approximate.")
        if st.button("Prepare COLMAP sparse model", type="secondary"):
            with st.spinner("Building COLMAP model..."):
                export_dir = Path(tempfile.mkdtemp(prefix="alignner_colmap_")) / "sparse" / "0"
                sample = result.features[0]
                focal = float(focal_override) if focal_override > 0 else max(sample.width, sample.height) * 1.15
                K_export = np.array([[focal, 0, sample.width / 2.0],
                                     [0, focal, sample.height / 2.0],
                                     [0, 0, 1]], dtype=np.float64)
                aligner_export = ImageAligner(feature_detector=detector, max_features=max_features,
                                              match_ratio=match_ratio, min_inliers=min_inliers)
                aligner_export.export_colmap(result, str(export_dir), K_export)
                archive_buffer = io.BytesIO()
                with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file_path in export_dir.rglob("*"):
                        if file_path.is_file():
                            archive.write(file_path, file_path.relative_to(export_dir.parent))
                st.session_state["colmap_export_zip"] = archive_buffer.getvalue()
        if st.session_state.get("colmap_export_zip"):
            st.download_button("Download COLMAP sparse model (.zip)",
                               data=st.session_state["colmap_export_zip"],
                               file_name="alignner_colmap_sparse_model.zip",
                               mime="application/zip")

    st.subheader("Connected Components")
    for i, comp in enumerate(result.components):
        names = [Path(result.features[j].path).name for j in comp]
        st.write(f"**Component {i+1}** ({len(comp)}): {', '.join(names[:8])}{'...' if len(names)>8 else ''}")

    if result.pairs:
        st.subheader("Browse Matches")
        sorted_pairs = sorted(result.pairs, key=lambda x: -x.inliers)
        pair_options = [f"{Path(result.features[p.idx1].path).name} ↔ {Path(result.features[p.idx2].path).name} ({p.inliers} inliers)" for p in sorted_pairs]
        if "match_idx" not in st.session_state:
            st.session_state.match_idx = 0
        col_prev, col_sel, col_next = st.columns([1, 6, 1])
        with col_prev:
            if st.button("← Prev", disabled=st.session_state.match_idx <= 0):
                st.session_state.match_idx = max(0, st.session_state.match_idx - 1)
                st.rerun()
        with col_next:
            if st.button("Next →", disabled=st.session_state.match_idx >= len(pair_options)-1):
                st.session_state.match_idx = min(len(pair_options)-1, st.session_state.match_idx + 1)
                st.rerun()
        with col_sel:
            selected_pair = st.selectbox("Select a matched pair", pair_options,
                                         index=st.session_state.match_idx, key="pair_select")
            new_idx = pair_options.index(selected_pair)
            if new_idx != st.session_state.match_idx:
                st.session_state.match_idx = new_idx
                st.rerun()
        chosen = sorted_pairs[st.session_state.match_idx]
        f1, f2 = result.features[chosen.idx1], result.features[chosen.idx2]
        if f1.image is not None and f2.image is not None:
            vis = visualize_matches(f1.image, f1.keypoints, f2.image, f2.keypoints,
                                    chosen.matches, max_draw=100)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{chosen.inliers} inliers",
                     use_container_width=True)
            st.caption(f"Pair {st.session_state.match_idx+1} of {len(sorted_pairs)} • Green lines = verified matches")

    # ── Component-aware diagnostic visualisation (geometry unchanged) ──
    if len(result.components) > 1:
        st.warning(
            f"{len(result.components)} independent components detected. "
            "Each component has its own local coordinate frame. "
            "They are shown together only for inspection and are **not** in a common world frame."
        )

    # Build component index map
    component_of = {}
    for ci, comp in enumerate(result.components):
        for idx in comp:
            component_of[idx] = ci

    # Distinct colours
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Component show/hide
    st.subheader("Components")
    show_component = []
    cols = st.columns(min(4, max(1, len(result.components))))
    for ci, comp in enumerate(result.components):
        with cols[ci % len(cols)]:
            show_component.append(
                st.checkbox(
                    f"Component {ci+1} ({len(comp)} images)",
                    value=True,
                    key=f"show_comp_{ci}"
                )
            )

    # Point colouring mode
    colour_mode = st.selectbox(
        "Sparse point colouring",
        ["Component membership", "Observation count (track length)", "Reprojection error"],
        index=0,
        key="point_colour_mode"
    )

    # Camera selector for inspection
    camera_options = []
    for idx in sorted(result.camera_poses):
        name = Path(result.features[idx].path).name
        ci = component_of.get(idx, -1)
        camera_options.append((idx, f"{name} (comp {ci+1})"))
    selected_cam_label = st.selectbox(
        "Inspect camera",
        [label for _, label in camera_options] if camera_options else ["(none)"],
        key="inspect_camera"
    )
    selected_cam_idx = None
    if camera_options:
        for idx, label in camera_options:
            if label == selected_cam_label:
                selected_cam_idx = idx
                break

    if selected_cam_idx is not None:
        feat = result.features[selected_cam_idx]
        ci = component_of.get(selected_cam_idx, -1)
        pair_count = sum(1 for p in result.pairs if selected_cam_idx in (p.idx1, p.idx2))
        st.write(f"**Component:** {ci+1} • **Verified pairs:** {pair_count} • "
                 f"**Pose recovered:** Yes")
        if feat.image is not None:
            st.image(cv2.cvtColor(feat.image, cv2.COLOR_BGR2RGB),
                     caption=Path(feat.path).name, use_container_width=True)

    # Build Plotly figure
    if result.camera_poses or (result.sparse_points is not None and len(result.sparse_points)):
        st.subheader("Cameras + Sparse Points (component-aware)")
        fig = go.Figure()

        # Helper: camera centre (OpenCV convention X_cam = R X_world + t)
        def camera_center(R, t):
            return -R.T @ np.asarray(t).reshape(3)

        # Frustum wireframe scale (scene-relative later; fixed for now)
        frustum_scale = 0.15

        for ci, comp in enumerate(result.components):
            if not show_component[ci]:
                continue
            color = palette[ci % len(palette)]
            centers = []
            labels = []
            for idx in sorted(comp):
                if idx not in result.camera_poses:
                    continue
                R, t = result.camera_poses[idx]
                C = camera_center(R, t)
                centers.append(C)
                labels.append(Path(result.features[idx].path).name)

                # Viewing direction (camera +Z in world)
                forward = R.T[:, 2]
                # Simple pyramid frustum
                right = R.T[:, 0]
                up = -R.T[:, 1]  # OpenCV Y-down → flip for visual
                apex = C
                base_center = C + forward * frustum_scale
                hw, hh = frustum_scale * 0.6, frustum_scale * 0.45
                corners = [
                    base_center + right * hw + up * hh,
                    base_center - right * hw + up * hh,
                    base_center - right * hw - up * hh,
                    base_center + right * hw - up * hh,
                ]
                # Draw pyramid edges
                for cnr in corners:
                    fig.add_trace(go.Scatter3d(
                        x=[apex[0], cnr[0]], y=[apex[1], cnr[1]], z=[apex[2], cnr[2]],
                        mode="lines", line=dict(color=color, width=2),
                        showlegend=False, hoverinfo="skip"
                    ))
                # Base
                base_closed = corners + [corners[0]]
                fig.add_trace(go.Scatter3d(
                    x=[p[0] for p in base_closed],
                    y=[p[1] for p in base_closed],
                    z=[p[2] for p in base_closed],
                    mode="lines", line=dict(color=color, width=2),
                    showlegend=False, hoverinfo="skip"
                ))

            if centers:
                arr = np.array(centers)
                fig.add_trace(go.Scatter3d(
                    x=arr[:,0], y=arr[:,1], z=arr[:,2],
                    mode="markers+text",
                    marker=dict(size=6, color=color),
                    text=labels,
                    textposition="top center",
                    name=f"Component {ci+1} ({len(comp)})"
                ))

        # Sparse points coloured by selected mode
        if result.sparse_points is not None and len(result.sparse_points):
            pts = result.sparse_points
            # Map each point to a component via its tracks
            point_component = {}
            obs_count = {}
            if result.point_tracks:
                for pid, img_idx, _ in result.point_tracks:
                    point_component[pid] = component_of.get(img_idx, -1)
                    obs_count[pid] = obs_count.get(pid, 0) + 1

            colors = []
            for i in range(len(pts)):
                ci = point_component.get(i, -1)
                if colour_mode == "Component membership":
                    colors.append(palette[ci % len(palette)] if ci >= 0 else "#aaaaaa")
                elif colour_mode == "Observation count (track length)":
                    n = obs_count.get(i, 0)
                    # Simple heat: more observations = darker blue
                    intensity = min(1.0, n / 8.0)
                    colors.append(f"rgba(0, {int(80+intensity*175)}, 255, 0.8)")
                else:
                    # Reprojection error not stored; mark as unavailable
                    colors.append("#cccccc")

            # Subsample if huge
            if len(pts) > 20000:
                ix = np.random.choice(len(pts), 20000, replace=False)
                pts = pts[ix]
                colors = [colors[i] for i in ix]

            fig.add_trace(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode="markers",
                marker=dict(size=1.6, color=colors, opacity=0.7),
                name="Sparse points"
            ))

        fig.update_layout(
            scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            height=720, margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        if colour_mode == "Reprojection error":
            st.caption("Reprojection error colouring is unavailable: per-point reprojection errors are not stored in the current result.")
        st.caption(
            f"{len(result.camera_poses)} cameras • "
            f"{0 if result.sparse_points is None else len(result.sparse_points)} sparse points. "
            "Components use independent local frames."
        )

    # Per-component COLMAP export
    if result.camera_poses and result.components:
        st.subheader("Per-component COLMAP export")
        st.caption("Each component is exported in its own local frame. No transforms are invented.")
        for ci, comp in enumerate(result.components):
            if st.button(f"Export Component {ci+1} ({len(comp)} images)", key=f"export_comp_{ci}"):
                with st.spinner(f"Exporting component {ci+1}..."):
                    export_dir = Path(tempfile.mkdtemp(prefix=f"alignner_comp{ci+1}_")) / "sparse" / "0"
                    sample = result.features[comp[0]]
                    focal = float(focal_override) if focal_override > 0 else max(sample.width, sample.height) * 1.15
                    K_export = np.array([[focal, 0, sample.width / 2.0],
                                         [0, focal, sample.height / 2.0],
                                         [0, 0, 1]], dtype=np.float64)
                    aligner_export = ImageAligner()
                    aligner_export.export_colmap_component(
                        result, ci, str(export_dir), K_export
                    )
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for p in export_dir.rglob("*"):
                            if p.is_file():
                                zf.write(p, p.relative_to(export_dir.parent))
                    st.session_state[f"comp_export_{ci}"] = buf.getvalue()
            if st.session_state.get(f"comp_export_{ci}"):
                st.download_button(
                    f"Download Component {ci+1} COLMAP model",
                    data=st.session_state[f"comp_export_{ci}"],
                    file_name=f"alignner_component_{ci+1}_colmap.zip",
                    mime="application/zip",
                    key=f"dl_comp_{ci}"
                )

    st.subheader("Browse Images")
    if result.features:
        names = [Path(f.path).name for f in result.features]
        selected = st.selectbox("Select image", names, index=0)
        feat = result.features[names.index(selected)]
        if feat.image is not None:
            ca, cb = st.columns([2, 1])
            with ca:
                st.image(cv2.cvtColor(feat.image, cv2.COLOR_BGR2RGB), caption=selected, use_container_width=True)
            with cb:
                st.write(f"**Size:** {feat.width} × {feat.height}")
                st.write(f"**Features:** {len(feat.keypoints)}")
                st.write(f"**Pose recovered:** {'Yes' if names.index(selected) in result.camera_poses else 'No'}")

    if st.button("Clear results"):
        for key in ("alignment_result", "alignment_done", "selection_result", "selection_csv", "match_idx", "contribution_report"):
            st.session_state.pop(key, None)
        st.rerun()

st.markdown("---")
st.caption("Alignner prototype • Selection is conservative and opt-in • Full SfM / production COLMAP export remains separate")
