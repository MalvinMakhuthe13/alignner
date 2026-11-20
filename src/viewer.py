from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def render_ply_viewer() -> None:
    """Render an isolated, read-only Three.js PLY inspection viewport."""
    st.header("3D Viewer")
    st.caption(
        "Inspect an existing PLY reconstruction in an embedded 3D viewport. "
        "The uploaded source is read-only; this viewer does not alter reconstruction files."
    )

    uploaded = st.file_uploader(
        "Load a PLY point cloud or mesh",
        type=["ply"],
        key="alignner_ply_viewer_upload",
        help="Start with scene_dense_cloudcompare_test.ply for the first test.",
    )
    if uploaded is None:
        st.info("Choose a PLY file to open the 3D workspace.")
        return

    payload = uploaded.getvalue()
    if not payload:
        st.error("The selected PLY file is empty.")
        return

    left, right, third = st.columns([2, 1, 1])
    with left:
        st.write(f"**File:** {uploaded.name}")
        st.caption(f"{len(payload):,} bytes")
    with right:
        point_size = st.slider(
            "Point size", min_value=1, max_value=10, value=3,
            key="alignner_viewer_point_size",
        )
    with third:
        show_axes = st.checkbox("Show axes", value=True, key="alignner_viewer_axes")

    encoded = base64.b64encode(payload).decode("ascii")
    encoded_json = json.dumps(encoded)
    point_size_json = json.dumps(point_size / 1000.0)
    axes_json = json.dumps(show_axes)

    html = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#101722;color:#e5edf7;font-family:Inter,Segoe UI,Arial,sans-serif}
#view{position:absolute;inset:0}
#hud{position:absolute;z-index:2;top:12px;left:14px;padding:9px 12px;border:1px solid #344256;border-radius:8px;background:rgba(12,20,31,.82);font-size:12px;line-height:1.55;pointer-events:none}
#help{position:absolute;z-index:2;right:14px;bottom:12px;padding:8px 10px;border-radius:8px;background:rgba(12,20,31,.78);color:#cbd5e1;font-size:11px;pointer-events:none}
#error{position:absolute;left:16px;right:16px;top:70px;color:#fecaca;background:#451a1a;padding:12px;border-radius:8px;display:none;z-index:3;white-space:pre-wrap}
</style>
</head>
<body>
<div id="view"></div>
<div id="hud">ALIGNNER 3D VIEWER<br><span id="stats">Loading PLY…</span></div>
<div id="help">Left drag: orbit · Right drag: pan · Wheel: zoom · Double-click: focus</div>
<div id="error"></div>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const encoded = __PLY_BASE64__;
const pointSize = __POINT_SIZE__;
const showAxes = __SHOW_AXES__;
const root = document.getElementById('view');
const errorBox = document.getElementById('error');
function fail(err){errorBox.style.display='block';errorBox.textContent='Could not load this PLY file.\n'+(err?.stack||err?.message||String(err));document.getElementById('stats').textContent='Load failed';}
try {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
  const renderer = new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.setSize(root.clientWidth,root.clientHeight);
  renderer.setClearColor(0x101722,1);
  root.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101722);
  const camera = new THREE.PerspectiveCamera(50,root.clientWidth/root.clientHeight,0.001,1e9);
  const controls = new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;
  controls.dampingFactor=0.075;
  controls.screenSpacePanning=true;
  controls.zoomSpeed=1.15;
  controls.rotateSpeed=0.8;

  const loader = new PLYLoader();
  const geometry = loader.parse(bytes.buffer);
  if(!geometry.getAttribute('position') || geometry.getAttribute('position').count===0) throw new Error('No vertex positions were found in the PLY file.');
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const hasColor=!!geometry.getAttribute('color');
  const isMesh=geometry.index!==null || geometry.getAttribute('normal')!==undefined;
  let object;
  if(isMesh && geometry.index){
    const material=new THREE.MeshStandardMaterial({color:hasColor?0xffffff:0xcbd5e1,vertexColors:hasColor,roughness:0.82,metalness:0.02,side:THREE.DoubleSide});
    object=new THREE.Mesh(geometry,material);
    scene.add(new THREE.HemisphereLight(0xffffff,0x334155,2.0));
    const key=new THREE.DirectionalLight(0xffffff,2.0);key.position.set(1,2,3);scene.add(key);
  } else {
    const material=new THREE.PointsMaterial({size:pointSize, sizeAttenuation:true, vertexColors:hasColor, color:hasColor?0xffffff:0x8ec5ff});
    object=new THREE.Points(geometry,material);
  }
  scene.add(object);

  if(showAxes){const axes=new THREE.AxesHelper(Math.max(geometry.boundingSphere.radius*0.25,0.01));scene.add(axes);}
  const center=geometry.boundingSphere.center;
  const radius=Math.max(geometry.boundingSphere.radius,0.01);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(radius*1.8,radius*1.25,radius*2.2));
  camera.near=Math.max(radius/10000,0.0001);
  camera.far=Math.max(radius*100,100);
  camera.updateProjectionMatrix();
  controls.update();

  document.getElementById('stats').textContent =
    geometry.getAttribute('position').count.toLocaleString()+' vertices · '+(hasColor?'RGB colour':'default colour')+' · '+(isMesh?'mesh':'point cloud');

  renderer.domElement.addEventListener('dblclick',()=>{
    controls.target.copy(center);controls.update();
  });
  const resize=()=>{
    const w=root.clientWidth,h=root.clientHeight;
    if(!w||!h)return;
    camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);
  };
  new ResizeObserver(resize).observe(root);
  function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}
  animate();
} catch(err){fail(err);}
</script>
</body>
</html>
"""
    html = (
        html.replace("__PLY_BASE64__", encoded_json)
        .replace("__POINT_SIZE__", point_size_json)
        .replace("__SHOW_AXES__", axes_json)
    )
    components.html(html, height=760, scrolling=False)
