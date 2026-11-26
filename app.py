"""
Alignner — photo & video alignment for photogrammetry and Gaussian splatting.

Workflow (like RealityScan):  Inputs → Align → Inspect → Scene → Export
Real incremental SfM with bundle adjustment on the CPU, no AI. Exports a
COLMAP folder that Brush, gsplat and Nerfstudio train on directly.
"""
import copy
import io
import json
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
from src import scene_tools as T
from src.dense import densify, write_ply, brush_cloud, DenseCloud

st.set_page_config(page_title="Alignner", page_icon="📷", layout="wide", initial_sidebar_state="expanded")

ACCENT = "#c8ff00"
st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  div[data-testid="stMetric"] {{ background: #15181d; border: 1px solid #23272e; border-radius: 10px; padding: 10px 14px; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  .an-title {{ display:flex; align-items:baseline; gap:12px; margin-bottom:.2rem }}
  .an-title b {{ font-size:1.6rem; }}
  .an-title span {{ color:#8a919b; font-size:.95rem }}
  .an-step {{ color:#8a919b; font-size:.85rem; margin:-.4rem 0 .8rem }}
  .thumb {{ position:relative; border-radius:8px; overflow:hidden; border:2px solid #23272e; margin-bottom:6px }}
  .thumb.ok {{ border-color:#3a5a00 }}
  .thumb.bad {{ border-color:#6b1d1d }}
  .thumb img {{ width:100%; display:block }}
  .thumb .badge {{ position:absolute; right:4px; bottom:4px; font:600 11px/1 system-ui; padding:3px 6px; border-radius:6px; background:rgba(0,0,0,.7); color:{ACCENT} }}
  .thumb.bad .badge {{ color:#ff8a8a }}
  .thumb .name {{ position:absolute; left:4px; top:4px; font:500 10px/1 system-ui; padding:2px 5px; border-radius:5px; background:rgba(0,0,0,.55); color:#ddd; max-width:90%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap }}
</style>
<div class="an-title"><b>📷 Alignner</b><span>alignment for photogrammetry &amp; Gaussian splatting · CPU only · no AI</span></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────── sidebar: quality presets
PRESETS = {
    "Preview (fast)": dict(detector="sift", max_features=4000, ratio=0.8, min_inliers=25),
    "Normal": dict(detector="sift", max_features=8000, ratio=0.8, min_inliers=30),
    "High detail": dict(detector="sift", max_features=14000, ratio=0.78, min_inliers=40),
}
with st.sidebar:
    st.header("Alignment")
    preset = st.radio("Quality", list(PRESETS), index=1,
                      help="Preview is quick for checking a capture; High detail finds more points for big or detailed scenes.")
    p = dict(PRESETS[preset])
    with st.expander("Advanced"):
        p["max_features"] = st.slider("Features per image", 2000, 20000, p["max_features"], 1000)
        p["ratio"] = st.slider("Match strictness (lower = stricter)", 0.6, 0.9, p["ratio"], 0.02)
        p["min_inliers"] = st.slider("Min verified matches per pair", 15, 100, p["min_inliers"])
        focal = st.number_input("Focal length (px) — 0 = from photo EXIF, then refined", 0.0, step=50.0)
        exhaustive_limit = st.slider("Match every pair up to N images", 20, 150, 60,
                                     help="Above this, neighbours + visual retrieval are matched (much faster).")
        window = st.slider("Sequential neighbours", 4, 30, 12)
        p["detector"] = st.selectbox("Features", ["sift", "orb"], index=0)
    st.header("Video")
    fps_extract = st.slider("Keep every Nth frame", 1, 30, 10)
    max_frames = st.slider("Max frames per video", 10, 400, 150)
    st.header("Preflight")
    use_selection = st.checkbox("Remove strong near-duplicates", value=False)
    review_threshold = st.slider("Flag blur below", 5.0, 80.0, 18.0, 1.0)

ss = st.session_state


# ─────────────────────────────────────────── helpers
def extract_frames(video_bytes, filename, every_n, max_n, tmpdir: Path):
    vp = tmpdir / Path(filename).name
    vp.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(vp))
    frames, idx, saved = [], 0, 0
    while cap.isOpened() and saved < max_n:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            out = tmpdir / f"{Path(filename).stem}_frame_{saved:04d}.jpg"
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frames.append(str(out)); saved += 1
        idx += 1
    cap.release()
    vp.unlink(missing_ok=True)
    return frames


def thumb_b64(path, w=240):
    import base64
    img = cv2.imread(path)
    if img is None:
        return ""
    h = int(img.shape[0] * w / img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def zip_folder(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent))
    return buf.getvalue()


def err_color(e):
    if e is None:
        return "#ff5a5a"
    t = min(1.0, max(0.0, (e - 0.3) / 1.7))          # 0.3px green → 2px red
    r, g = int(80 + 175 * t), int(255 - 150 * t)
    return f"rgb({r},{g},60)"


def scene_figure(rec, show_cams=True, show_pts=True, max_pts=60000, pt_size=1.6, dense=None):
    fig = go.Figure()
    reg = rec.registered_views
    if show_pts and dense is not None and len(dense):
        ix = np.random.default_rng(0).choice(len(dense), min(max_pts, len(dense)), replace=False)
        P = dense.points[ix].astype(np.float64); C = dense.colors[ix]
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="Dense cloud",
                                   marker=dict(size=pt_size, color=[f"rgb({r},{g},{b})" for r, g, b in C]), hoverinfo="skip"))
    elif show_pts and rec.points:
        pts = rec.points
        ix = np.random.default_rng(0).choice(len(pts), min(max_pts, len(pts)), replace=False)
        P = np.array([pts[i].xyz for i in ix]); C = np.array([pts[i].color for i in ix])
        # Keep far strays from zooming the view out (display only; ④ Scene removes them for real)
        lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
        pad = (hi - lo) * 0.25
        keep = np.all((P >= lo - pad) & (P <= hi + pad), axis=1)
        P, C = P[keep], C[keep]
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="Sparse points",
                                   marker=dict(size=pt_size, color=[f"rgb({r},{g},{b})" for r, g, b in C]),
                                   hoverinfo="skip"))
    if show_cams and reg:
        q = {x.view_id: x for x in T.image_quality(rec)}
        centers = np.array([camera_center(v) for v in reg])
        spread = np.median(np.linalg.norm(centers - centers.mean(0), axis=1)) if len(reg) > 1 else 1.0
        d = max(1e-3, 0.12 * spread)
        xs, ys, zs = [], [], []
        for v in reg:
            cam = rec.cameras[v.camera_id]
            Rc, _ = cv2.Rodrigues(v.rvec)
            C = camera_center(v)
            hw, hh = cam.width / (2 * cam.f) * d, cam.height / (2 * cam.f) * d
            corners = [Rc.T @ (np.array(c) - v.tvec) for c in ([-hw, -hh, d], [hw, -hh, d], [hw, hh, d], [-hw, hh, d])]
            for a, b in [(C, corners[0]), (C, corners[1]), (C, corners[2]), (C, corners[3]),
                         (corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]:
                xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Cameras", line=dict(color=ACCENT, width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers", name="Camera centres",
            marker=dict(size=4, color=[err_color(q[v.id].mean_error_px) for v in reg]),
            text=[f"{q[v.id].name}<br>{q[v.id].observations} pts · {q[v.id].mean_error_px or 0:.2f}px" for v in reg],
            hoverinfo="text"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0d10", plot_bgcolor="#0b0d10", height=720,
                      margin=dict(l=0, r=0, b=0, t=10), legend=dict(orientation="h", y=1.02, x=0),
                      scene=dict(aspectmode="data", bgcolor="#0b0d10",
                                 xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig


def set_rec(rec, out_dir):
    ss["rec"] = rec
    ss["rec_aligned"] = copy.deepcopy(rec)          # for "Reset scene edits"
    ss["out_dir"] = out_dir
    ss["scene_log"] = []
    ss.pop("export_zip", None)


# ─────────────────────────────────────────── tabs
t_in, t_align, t_inspect, t_scene, t_export = st.tabs(["① Inputs", "② Align", "③ Inspect", "④ Scene", "⑤ Export"])

with t_in:
    st.markdown('<div class="an-step">Add the photos (or a video) of one subject or site. 60–80 % overlap between neighbouring shots works best.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        image_files = st.file_uploader("Photos", type=["jpg", "jpeg", "png", "webp", "tif", "tiff"], accept_multiple_files=True)
    with c2:
        video_files = st.file_uploader("Videos", type=["mp4", "mov", "avi", "mkv", "webm"], accept_multiple_files=True)
    if st.button("Load inputs", type="primary", disabled=not (image_files or video_files)):
        tmp = Path(tempfile.mkdtemp(prefix="alignner_"))
        src = tmp / "input"; src.mkdir()
        paths = []
        with st.spinner("Preparing…"):
            for f in image_files or []:
                p_ = src / Path(f.name).name
                p_.write_bytes(f.getvalue()); paths.append(str(p_))
            for vf in video_files or []:
                paths += extract_frames(vf.getvalue(), vf.name, fps_extract, max_frames, src)
            if use_selection and len(paths) > 2:
                sel = IntelligentImageSelector(min_sharpness_for_review=review_threshold).analyze(paths)
                ss["selection"] = sel
                paths = sel.selected_paths
            ss["inputs"] = paths
            ss["thumbs"] = {p_: thumb_b64(p_) for p_ in paths}
            ss["workdir"] = str(tmp)
            for k in ("rec", "rec_aligned", "out_dir"):
                ss.pop(k, None)
        st.success(f"{len(paths)} images ready — go to ② Align.")

    inputs = ss.get("inputs", [])
    if inputs:
        rec = ss.get("rec")
        q = {Path(x.name).name: x for x in T.image_quality(rec)} if rec else {}
        reg = sum(1 for x in q.values() if x.registered)
        st.caption(f"{len(inputs)} images" + (f" · {reg} placed" if rec else ""))
        cols = st.columns(6)
        for i, path in enumerate(inputs):
            name = Path(path).name
            info = q.get(name)
            cls = "" if not info else ("ok" if info.registered else "bad")
            badge = "" if not info else (f"✓ {info.observations}" if info.registered else "not placed")
            cols[i % 6].markdown(f'<div class="thumb {cls}"><img src="data:image/jpeg;base64,{ss["thumbs"].get(path, "")}">'
                                 f'<div class="name">{name}</div>{f"<div class=badge>{badge}</div>" if badge else ""}</div>',
                                 unsafe_allow_html=True)

with t_align:
    inputs = ss.get("inputs", [])
    st.markdown('<div class="an-step">Finds matching features, places every camera and builds the sparse point cloud, then refines everything together (bundle adjustment).</div>', unsafe_allow_html=True)
    if not inputs:
        st.info("Add photos in ① Inputs first.")
    else:
        exclude = st.multiselect("Leave out images (optional)", [Path(x).name for x in inputs])
        go_btn = st.button(f"🚀 Align {len(inputs) - len(exclude)} images", type="primary")
        if go_btn:
            paths = [x for x in inputs if Path(x).name not in set(exclude)]
            bar = st.progress(0.0); msg = st.empty()

            def prog(f, m):
                bar.progress(min(1.0, float(f))); msg.caption(m)
            sfm = IncrementalSfM(detector=p["detector"], max_features=p["max_features"], ratio=p["ratio"],
                                 min_inliers=p["min_inliers"], exhaustive_limit=exhaustive_limit,
                                 sequential_window=window, focal_px=focal or None, progress=prog)
            try:
                rec = sfm.run(paths)
                out_dir = str(Path(ss["workdir"]) / "colmap")
                set_rec(rec, out_dir)
                msg.caption("Done.")
            except ValueError as e:
                st.error(str(e))
        rec = ss.get("rec")
        if rec:
            s = rec.stats()
            comps = T.components(rec)
            m = st.columns(5)
            m[0].metric("Cameras placed", f"{s['registered']}/{s['images']}")
            m[1].metric("Sparse points", f"{s['points']:,}")
            m[2].metric("Mean error", f"{s['mean_reprojection_error_px'] or 0:.2f} px", help="Under 1 px is excellent.")
            m[3].metric("Track length", f"{s['mean_track_length'] or 0:.1f} views")
            m[4].metric("Components", len(comps), help="Groups of photos that connect to each other. 1 is ideal.")
            lens = " · ".join(f"{c.width}×{c.height}  f {c.f:.0f}px  k {c.k:+.3f}" for c in rec.cameras.values())
            st.caption(f"Lens: {lens}")
            unplaced = [Path(v.path).name for v in rec.views if not v.registered]
            if unplaced:
                st.warning(f"{len(unplaced)} image(s) not placed: {', '.join(unplaced[:12])}{'…' if len(unplaced) > 12 else ''}. "
                           "Add photos in between, or leave them out and align again.")
            with st.expander("Alignment log"):
                st.code("\n".join(rec.log) or "—")

with t_inspect:
    rec = ss.get("rec")
    if not rec:
        st.info("Align first.")
    else:
        c = st.columns([1, 1, 1, 1])
        show_cams = c[0].toggle("Cameras", True)
        show_pts = c[1].toggle("Points", True)
        pt_size = c[2].slider("Point size", 0.5, 4.0, 1.6, 0.1)
        max_pts = c[3].select_slider("Max points shown", [10000, 30000, 60000, 120000], 60000)
        dense = ss.get("dense")
        use_dense = dense is not None and st.toggle("Show dense cloud", True)
        st.plotly_chart(scene_figure(rec, show_cams, show_pts, max_pts, pt_size, dense if use_dense else None), use_container_width=True)
        st.caption("Camera dots are coloured by their error: green = sharp fit, red = poor fit or worth re-shooting.")
        st.subheader("Quality analysis")
        rows = [{"Image": x.name, "Placed": "✓" if x.registered else "✗", "Features": x.features,
                 "3D points seen": x.observations, "Error (px)": None if x.mean_error_px is None else round(x.mean_error_px, 3),
                 "Coverage": f"{x.coverage * 100:.0f}%"} for x in T.image_quality(rec)]
        rows.sort(key=lambda r: (r["Placed"] == "✓", -(r["Error (px)"] or 99)))
        st.dataframe(rows, use_container_width=True, hide_index=True, height=320)
        st.caption("Coverage = how much of the frame has matched points. Low coverage or high error images are the first to re-shoot or leave out.")

with t_scene:
    rec = ss.get("rec")
    if not rec:
        st.info("Align first.")
    else:
        st.markdown('<div class="an-step">Orient, scale and tidy the model. Cameras and points change together, so the export stays exact.</div>', unsafe_allow_html=True)
        log = ss.setdefault("scene_log", [])
        a, b = st.columns(2)
        with a:
            st.markdown("**Level & centre**")
            up = st.radio("Up axis", ["Y-up (Fumoca, three.js, Brush)", "Z-up (RealityScan, Blender)"], horizontal=False)
            centre = st.checkbox("Move the subject to the origin", True)
            if st.button("Level scene"):
                info = T.level_scene(rec, up_axis="y" if up.startswith("Y") else "z", center=centre)
                log.append(f"Levelled ({info['up_axis']}-up){' and centred' if centre else ''}")
            st.markdown("**Georeference from photo GPS**")
            st.caption("Drone and phone photos carry GPS. This scales the model to metres and orients it east-north-up.")
            if st.button("Use GPS"):
                try:
                    info = T.georeference_from_gps(rec)
                    log.append(f"Georeferenced from {info['images_with_gps']} GPS photos · fit {info['rms_m']:.2f} m RMS")
                except ValueError as e:
                    st.error(str(e))
        with b:
            st.markdown("**Define distance**")
            reg = rec.registered_views
            names = [Path(v.path).name for v in reg]
            if len(reg) >= 2:
                ia = st.selectbox("Camera A", range(len(reg)), format_func=lambda i: names[i])
                ib = st.selectbox("Camera B", range(len(reg)), index=min(1, len(reg) - 1), format_func=lambda i: names[i])
                metres = st.number_input("Real distance between them (m)", 0.0, 10000.0, 1.0, 0.01)
                if st.button("Apply scale"):
                    try:
                        sc = T.define_distance(rec, reg[ia].id, reg[ib].id, metres)
                        log.append(f"Scaled ×{sc:.4f} so {names[ia]} ↔ {names[ib]} = {metres} m")
                    except ValueError as e:
                        st.error(str(e))
            st.markdown("**Clean sparse cloud**")
            min_track = st.slider("Keep points seen in at least N photos", 2, 6, 2)
            strength = st.select_slider("Outlier removal", ["Off", "Gentle", "Normal", "Strong"], "Normal")
            crop = st.checkbox("Crop to the main region (drops far-away strays)", True)
            if st.button("Clean points"):
                std = {"Off": None, "Gentle": 3.5, "Normal": 2.5, "Strong": 1.8}[strength]
                n = T.clean_points(rec, min_track=min_track, outlier_k=8 if std else 0, outlier_std=std or 0)
                if crop and rec.points:
                    lo, hi = T.robust_bounds(rec, 1, 99)
                    pad = (hi - lo) * 0.15
                    n += T.crop_to_region(rec, lo - pad, hi + pad)
                log.append(f"Removed {n:,} weak/stray points ({len(rec.points):,} left)")
        st.markdown("**Dense cloud — see the real object**")
        st.caption("Stereo-matches neighbouring photos and fuses the depth into a dense, clean, coloured cloud "
                   "that already looks like the object (like RealityScan / OpenMVS). Brush can start training from it.")
        dq = st.radio("Detail", ["Preview", "Normal", "High"], index=1, horizontal=True)
        if st.button("Build dense cloud", type="primary"):
            side, step = {"Preview": (700, 3), "Normal": (1000, 2), "High": (1600, 1)}[dq]
            bar = st.progress(0.0)
            try:
                ss["dense"] = densify(rec, max_side=side, step=step, progress=lambda f, m: bar.progress(min(1.0, f), m))
                ss["dense_log_len"] = len(log)
                log.append(f"Dense cloud: {len(ss['dense']):,} points from {ss['dense'].pairs_used} photo pairs")
            except ValueError as e:
                st.error(str(e))
        if ss.get("dense") is not None and ss.get("dense_log_len", -1) < len(log) - 1:
            ss.pop("dense", None)                    # model changed after the dense build
            st.info("The scene changed — build the dense cloud again.")
        if log:
            st.success(" · ".join(log[-3:]))
            if ss.get("export_log_len") != len(log):
                ss.pop("export_zip", None)          # scene changed → export must be prepared again
        if st.button("↺ Reset scene edits"):
            ss["rec"] = copy.deepcopy(ss["rec_aligned"]); ss["scene_log"] = []; ss.pop("export_zip", None); ss.pop("dense", None)
            st.rerun()
        st.plotly_chart(scene_figure(rec, True, True, 40000, 1.6, ss.get("dense")), use_container_width=True)

with t_export:
    rec = ss.get("rec")
    if not rec:
        st.info("Align first.")
    else:
        out_dir = Path(ss["out_dir"])
        st.markdown('<div class="an-step">The COLMAP folder opens directly in Brush (and gsplat, Nerfstudio, COLMAP). Scene edits from ④ are included.</div>', unsafe_allow_html=True)
        dense = ss.get("dense")
        start = st.radio("Starting points for Brush", [
            "Brush-ready: even, clean, small gaps filled (recommended)",
            "Everything: sparse + full dense cloud",
            "Sparse points only"], index=0,
            help="Brush trains faster and cleaner from evenly spread points that already cover the object.")
        target = st.select_slider("Brush-ready size", [100_000, 200_000, 300_000, 500_000, 800_000], 300_000,
                                  format_func=lambda n: f"{n // 1000}k points") if start.startswith("Brush") else None
        if dense is None and not start.startswith("Sparse"):
            st.caption("Tip: build the dense cloud in ④ Scene first — the Brush-ready cloud uses it to fill the surface.")
        add_dense = dense is not None and start.startswith("Everything")
        if st.button("Prepare export", type="primary"):
            if out_dir.exists():
                shutil.rmtree(out_dir)
            with st.spinner("Writing COLMAP model and images…"):
                extra, binfo = None, None
                if start.startswith("Brush"):
                    try:
                        bp, bc, binfo = brush_cloud(rec, dense, target=target)
                        extra = (bp, bc)
                    except ValueError as e:
                        st.error(str(e))
                elif add_dense:
                    extra = (dense.points, dense.colors)
                info = export_colmap(rec, str(out_dir), extra_points=extra)
                if binfo:
                    write_ply(str(out_dir / "brush_points.ply"), DenseCloud(points=extra[0], colors=extra[1], support=np.zeros(len(extra[0]), np.int32), pairs_used=0))
                    ss["export_brush"] = (out_dir / "brush_points.ply").read_bytes()
                    info["brush"] = binfo
                else:
                    ss.pop("export_brush", None)
                if dense is not None:
                    write_ply(str(out_dir / "dense_points.ply"), dense)
                    ss["export_dense"] = (out_dir / "dense_points.ply").read_bytes()
                else:
                    ss.pop("export_dense", None)
                report = {**rec.stats(), **info, "scene_edits": ss.get("scene_log", []),
                          "cameras": [{"id": c.id, "width": c.width, "height": c.height, "f": c.f, "k1": c.k} for c in rec.cameras.values()]}
                (out_dir / "alignner_report.json").write_text(json.dumps(report, indent=2))
                ss["export_zip"] = zip_folder(out_dir)
                ss["export_ply"] = (out_dir / "sparse_points.ply").read_bytes()
                ss["export_info"] = info
                ss["export_log_len"] = len(ss.get("scene_log", []))
        if "export_zip" in ss:
            info = ss["export_info"]
            b = info.get("brush")
            st.success(f"{info['images']} cameras · {info['points']:,} sparse points"
                       + (f" + {b['points']:,} Brush-ready points (even spacing, {b['gaps_filled']:,} gaps filled)" if b else (f" + {info['dense_points']:,} dense" if info.get('dense_points') else ""))
                       + (" · images undistorted for training" if info["undistorted"] else ""))
            c = st.columns(4)
            if ss.get("export_brush"):
                c[3].download_button("⬇️ Brush-ready cloud (.ply)", ss["export_brush"], "brush_points.ply")
            elif ss.get("export_dense"):
                c[3].download_button("⬇️ Dense cloud (.ply)", ss["export_dense"], "dense_points.ply")
            c[0].download_button("⬇️ COLMAP folder for Brush (.zip)", ss["export_zip"], "alignner_colmap.zip", "application/zip", type="primary")
            c[1].download_button("⬇️ Sparse point cloud (.ply)", ss["export_ply"], "sparse_points.ply")
            c[2].download_button("⬇️ Report (.json)", (out_dir / "alignner_report.json").read_bytes(), "alignner_report.json")
            st.caption("Unzip, then open the `colmap` folder in Brush. It contains `images/` and `sparse/0/`.")

st.markdown("---")
st.caption("Alignner · incremental SfM + bundle adjustment · RootSIFT · MAGSAC · Schur-complement LM · COLMAP export")
