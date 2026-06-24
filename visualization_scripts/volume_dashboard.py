#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import SimpleITK as sitk
from PIL import Image
from skimage.measure import marching_cubes


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "nnUNet" / "nnUNet_raw"
SEARCH_ROOTS = (
    RAW_ROOT,
    PROJECT_ROOT / "predictions",
    PROJECT_ROOT / "lh_pretraining_smoke_test" / "artifacts",
    PROJECT_ROOT / "safe_zone_experiments",
)
IMAGE_SUFFIXES = (".nii.gz", ".nii", ".mha")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse CT volumes, labels, predictions, and 3D overlays."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--case", default="dataset8_007")
    parser.add_argument("--open_browser", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate discovery and rendering, then exit without starting a server.",
    )
    return parser.parse_args()


def strip_image_suffix(name: str) -> str:
    for suffix in IMAGE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def case_id(path: Path) -> str:
    return strip_image_suffix(path.name).removesuffix("_0000")


def display_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT)
    parts = list(relative.parts)
    if len(parts) >= 3 and parts[:2] == ["nnUNet", "nnUNet_raw"]:
        parts = parts[2:]
    return " / ".join(parts[:-1])


def classify(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    stem = strip_image_suffix(path.name)
    if stem.endswith("_0000") or "imagestr" in parts or "imagests" in parts:
        return "image"
    return "mask"


def find_volumes() -> dict:
    datasets: dict[str, dict[str, dict[str, list[dict]]]] = {}
    all_cases: dict[str, dict[str, list[dict]]] = {}
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for suffix in IMAGE_SUFFIXES:
            pattern = f"*{suffix}"
            for path in root.rglob(pattern):
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                cid = case_id(path)
                dataset = next(
                    (part for part in path.parts if part.startswith("Dataset")),
                    "Experiment outputs",
                )
                entry = {
                    "id": str(resolved),
                    "name": display_name(path),
                    "path": str(resolved),
                }
                group = datasets.setdefault(dataset, {}).setdefault(
                    cid, {"images": [], "masks": []}
                )
                kind = "images" if classify(path) == "image" else "masks"
                group[kind].append(entry)
                all_cases.setdefault(cid, {"images": [], "masks": []})[kind].append(entry)

    for cases in datasets.values():
        for cid, group in cases.items():
            global_group = all_cases[cid]
            group["images"] = list(
                {item["path"]: item for item in global_group["images"]}.values()
            )
            group["masks"] = list(
                {item["path"]: item for item in global_group["masks"]}.values()
            )
            group["images"].sort(key=lambda item: item["name"])
            group["masks"].sort(key=lambda item: item["name"])
    return datasets


CATALOG = find_volumes()


@lru_cache(maxsize=12)
def read_volume(path_string: str):
    image = sitk.ReadImage(path_string)
    array = sitk.GetArrayFromImage(image)
    return array, tuple(image.GetSpacing()[::-1]), tuple(image.GetSize()[::-1])


def get_path(query: dict[str, list[str]], key: str) -> Path:
    value = query.get(key, [None])[0]
    if not value:
        raise ValueError(f"Missing query parameter: {key}")
    path = Path(value).resolve()
    if not path.is_file() or not any(path.is_relative_to(root) for root in SEARCH_ROOTS):
        raise FileNotFoundError(path)
    return path


def filter_mask(array: np.ndarray, label_value: int) -> np.ndarray:
    if label_value == 0:
        return array > 0
    return array == label_value


def aligned_arrays(image_path: Path, mask_paths: list[Path], label_value: int = 0):
    image, spacing, _ = read_volume(str(image_path))
    masks = []
    for path in mask_paths:
        mask, mask_spacing, _ = read_volume(str(path))
        if mask.shape != image.shape or not np.allclose(mask_spacing, spacing):
            raise ValueError(
                f"Geometry mismatch: {path.name} has {mask.shape}, image has {image.shape}"
            )
        masks.append(filter_mask(mask, label_value))
    return image, masks, spacing


def window_ct(array: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2
    scaled = np.clip((array.astype(np.float32) - low) / width, 0, 1)
    return (scaled * 255).astype(np.uint8)


def extract_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    index = int(np.clip(index, 0, array.shape[axis] - 1))
    if axis == 0:
        output = array[index, :, :]
    elif axis == 1:
        output = array[:, index, :]
    else:
        output = array[:, :, index]
    return np.flipud(output)


def slice_png(
    image: np.ndarray,
    reference: np.ndarray | None,
    comparison: np.ndarray | None,
    axis: int,
    index: int,
    center: float,
    width: float,
) -> bytes:
    gray = extract_slice(window_ct(image, center, width), axis, index)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    if reference is not None:
        mask = extract_slice(reference, axis, index)
        rgb[mask] = rgb[mask] * 0.35 + np.array([40, 220, 90]) * 0.65
    if comparison is not None:
        mask = extract_slice(comparison, axis, index)
        rgb[mask] = rgb[mask] * 0.35 + np.array([240, 75, 75]) * 0.65
    if reference is not None and comparison is not None:
        overlap = extract_slice(reference & comparison, axis, index)
        rgb[overlap] = rgb[overlap] * 0.2 + np.array([255, 210, 50]) * 0.8
    output = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    return buffer.getvalue()


def binary_metrics(reference: np.ndarray, comparison: np.ndarray) -> dict:
    tp = int(np.logical_and(reference, comparison).sum())
    fp = int(np.logical_and(~reference, comparison).sum())
    fn = int(np.logical_and(reference, ~comparison).sum())
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "dice": 1.0 if denominator == 0 else 2 * tp / denominator,
        "iou": 1.0 if union == 0 else tp / union,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reference_voxels": int(reference.sum()),
        "comparison_voxels": int(comparison.sum()),
    }


def mesh(mask: np.ndarray, spacing, step_size: int = 3):
    if not mask.any():
        return None
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
        step_size=step_size,
        allow_degenerate=False,
    )
    vertices -= np.asarray(spacing)
    return {
        "x": vertices[:, 2].round(3).tolist(),
        "y": vertices[:, 1].round(3).tolist(),
        "z": vertices[:, 0].round(3).tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


def json_bytes(value) -> bytes:
    return json.dumps(value).encode("utf-8")


HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pelvic Volume Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-3.3.0.min.js"></script>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #10151d; color: #ecf2f8; }
    header { padding: 18px 24px; background: #171f2a; border-bottom: 1px solid #2a3544; }
    h1 { margin: 0 0 4px; font-size: 22px; } .muted { color: #94a3b8; }
    main { padding: 18px; display: grid; gap: 16px; }
    .card { background: #171f2a; border: 1px solid #2a3544; border-radius: 10px; padding: 14px; }
    .controls { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; }
    label { display: grid; gap: 5px; font-size: 12px; color: #aebacd; }
    select, input, button { background: #0f141c; color: #eef4fb; border: 1px solid #38475a;
      border-radius: 6px; padding: 8px; min-width: 0; }
    button { cursor: pointer; background: #2563eb; font-weight: 650; }
    .slices { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .slice img { width: 100%; height: 360px; object-fit: contain; background: black; border-radius: 6px; }
    .metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
    .metric { background: #0f141c; padding: 10px; border-radius: 6px; }
    .metric strong { display: block; font-size: 20px; color: #f8fafc; }
    .paths { font: 12px ui-monospace, monospace; overflow-wrap: anywhere; line-height: 1.6; }
    #plot3d { height: 720px; }
    .legend span { margin-right: 18px; } .green { color: #28dc5a; }
    .red { color: #f04b4b; } .yellow { color: #ffd232; }
    @media (max-width: 1000px) {
      .controls, .slices { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
<header><h1>Pelvic Volume Dashboard</h1>
  <div class="muted">Inspect CT slices, masks, predictions, metrics, and 3D geometry.</div></header>
<main>
  <section class="card controls">
    <label>Dataset<select id="dataset"></select></label>
    <label>Case<select id="case"></select></label>
    <label>CT / image source<select id="image"></select></label>
    <label>Reference mask (green)<select id="reference"></select></label>
    <label>Comparison mask (red)<select id="comparison"></select></label>
    <label>Label class
      <select id="labelclass">
        <option value="0">All foreground labels</option>
        <option value="1">1 - sacrum</option>
        <option value="2">2 - right hip</option>
        <option value="3">3 - left hip</option>
        <option value="4">4 - lumbar vertebra</option>
      </select>
    </label>
    <label>CT window
      <select id="window"><option value="400,1800">Bone (C 400 / W 1800)</option>
        <option value="40,400">Soft tissue (C 40 / W 400)</option>
        <option value="-600,1500">Lung/air (C -600 / W 1500)</option></select>
    </label>
    <button id="load">Load selected case</button>
    <button id="build3d">Build / refresh 3D</button>
    <div class="legend"><span class="green">Reference</span><span class="red">Comparison</span>
      <span class="yellow">Overlap</span></div>
  </section>
  <section class="card"><div id="metrics" class="metric-grid"></div></section>
  <section class="slices">
    <div class="card slice"><label>Axial <input id="z" type="range"></label><img id="zimg"></div>
    <div class="card slice"><label>Coronal <input id="y" type="range"></label><img id="yimg"></div>
    <div class="card slice"><label>Sagittal <input id="x" type="range"></label><img id="ximg"></div>
  </section>
  <section class="card"><div id="plot3d"></div></section>
  <section class="card paths" id="paths"></section>
</main>
<script>
const initialCase = __INITIAL_CASE__;
let catalog = {}, current = null;
const el = id => document.getElementById(id);
const option = (value, text) => { const o=document.createElement('option'); o.value=value; o.textContent=text; return o; };
function fill(select, values, preferred) {
  select.innerHTML=''; values.forEach(v => select.appendChild(option(v.value,v.text)));
  if (preferred && values.some(v=>v.value===preferred)) select.value=preferred;
}
function refreshCases() {
  const cases=Object.keys(catalog[el('dataset').value]||{}).sort();
  fill(el('case'), cases.map(x=>({value:x,text:x})), cases.includes(initialCase)?initialCase:null); refreshSources();
}
function refreshSources() {
  const group=(catalog[el('dataset').value]||{})[el('case').value]||{images:[],masks:[]};
  fill(el('image'), group.images.map(x=>({value:x.path,text:x.name})));
  fill(el('reference'), [{value:'',text:'None'},...group.masks.map(x=>({value:x.path,text:x.name}))]);
  fill(el('comparison'), [{value:'',text:'None'},...group.masks.map(x=>({value:x.path,text:x.name}))]);
  const gt=group.masks.find(x=>x.name.includes('labelsTs')||x.name.includes('labelsTr'));
  const erased=group.masks.find(x=>x.name.includes('erased_input_single'));
  const normal=group.masks.find(x=>x.name.includes('normal_input_single'));
  if (gt) el('reference').value=gt.path;
  if (normal) el('comparison').value=normal.path; else if (erased) el('comparison').value=erased.path;
}
function params(extra={}) {
  const [center,width]=el('window').value.split(',');
  return new URLSearchParams({image:el('image').value,reference:el('reference').value,
    comparison:el('comparison').value,label:el('labelclass').value,center,width,...extra});
}
function refreshSlice(axis) {
  const slider=el(axis), img=el(axis+'img');
  img.src='/api/slice?'+params({axis,index:slider.value})+'&t='+Date.now();
}
async function loadCase() {
  if (!el('image').value) return;
  const data=await fetch('/api/info?'+params()).then(r=>r.json());
  if (data.error) return alert(data.error);
  current=data;
  ['z','y','x'].forEach((axis,i)=>{ const s=el(axis); s.max=data.shape[i]-1; s.value=Math.floor(data.shape[i]/2); refreshSlice(axis); });
  const m=data.metrics;
  el('metrics').innerHTML=m ? [
    ['Dice',m.dice.toFixed(4)],['IoU',m.iou.toFixed(4)],['False positive',m.fp.toLocaleString()],
    ['False negative',m.fn.toLocaleString()],['Pred / GT voxels',m.comparison_voxels.toLocaleString()+' / '+m.reference_voxels.toLocaleString()]
  ].map(x=>`<div class="metric"><span class="muted">${x[0]}</span><strong>${x[1]}</strong></div>`).join('') :
    '<div class="muted">Choose both a reference and comparison mask to calculate metrics.</div>';
  el('paths').innerHTML=`<b>Image:</b> ${data.image}<br><b>Reference:</b> ${data.reference||'None'}<br><b>Comparison:</b> ${data.comparison||'None'}`;
  el('paths').innerHTML+=`<br><b>Label class:</b> ${data.label_class}`;
  Plotly.purge('plot3d');
}
async function build3d() {
  el('plot3d').innerHTML='<div class="muted">Building surfaces...</div>';
  const data=await fetch('/api/mesh?'+params()).then(r=>r.json());
  if (data.error) return alert(data.error);
  const traces=[];
  if(data.reference) traces.push({type:'mesh3d',...data.reference,name:'Reference',color:'#28dc5a',opacity:.52,flatshading:true});
  if(data.comparison) traces.push({type:'mesh3d',...data.comparison,name:'Comparison',color:'#f04b4b',opacity:.52,flatshading:true});
  Plotly.newPlot('plot3d',traces,{paper_bgcolor:'#171f2a',plot_bgcolor:'#171f2a',font:{color:'#ecf2f8'},
    margin:{l:0,r:0,t:35,b:0},scene:{aspectmode:'data',xaxis:{title:'X'},yaxis:{title:'Y'},zaxis:{title:'Z'}},
    title:'3D mask comparison'},{responsive:true});
}
el('dataset').onchange=refreshCases; el('case').onchange=refreshSources;
['z','y','x'].forEach(a=>el(a).oninput=()=>refreshSlice(a));
el('window').onchange=()=>['z','y','x'].forEach(refreshSlice);
el('labelclass').onchange=loadCase;
el('load').onclick=loadCase; el('build3d').onclick=build3d;
fetch('/api/catalog').then(r=>r.json()).then(data=>{ catalog=data;
  const datasets=Object.keys(data).sort(); const preferred=datasets.find(d=>data[d][initialCase]);
  fill(el('dataset'),datasets.map(x=>({value:x,text:x})),preferred); refreshCases(); loadCase(); });
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def send_body(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_body(status, json_bytes(value), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                body = HTML.replace(
                    "__INITIAL_CASE__", json.dumps(self.server.initial_case)
                ).encode("utf-8")
                self.send_body(200, body, "text/html; charset=utf-8")
            elif parsed.path == "/api/catalog":
                self.send_json(CATALOG)
            elif parsed.path == "/api/info":
                self.handle_info(query)
            elif parsed.path == "/api/slice":
                self.handle_slice(query)
            elif parsed.path == "/api/mesh":
                self.handle_mesh(query)
            else:
                self.send_json({"error": "Not found"}, 404)
        except Exception as error:
            self.send_json({"error": str(error)}, 400)

    def selected(self, query):
        image_path = get_path(query, "image")
        reference_path = get_path(query, "reference") if query.get("reference", [""])[0] else None
        comparison_path = get_path(query, "comparison") if query.get("comparison", [""])[0] else None
        label_value = int(query.get("label", [0])[0])
        paths = [path for path in (reference_path, comparison_path) if path]
        image, masks, spacing = aligned_arrays(image_path, paths, label_value)
        iterator = iter(masks)
        reference = next(iterator) if reference_path else None
        comparison = next(iterator) if comparison_path else None
        return image_path, reference_path, comparison_path, image, reference, comparison, spacing, label_value

    def handle_info(self, query):
        image_path, reference_path, comparison_path, image, reference, comparison, spacing, label_value = self.selected(query)
        metrics = (
            binary_metrics(reference, comparison)
            if reference is not None and comparison is not None
            else None
        )
        self.send_json(
            {
                "image": str(image_path),
                "reference": str(reference_path) if reference_path else None,
                "comparison": str(comparison_path) if comparison_path else None,
                "shape": image.shape,
                "spacing_zyx": spacing,
                "label_class": label_value,
                "metrics": metrics,
            }
        )

    def handle_slice(self, query):
        _, _, _, image, reference, comparison, _, _ = self.selected(query)
        axis_name = query.get("axis", ["z"])[0]
        axis = {"z": 0, "y": 1, "x": 2}[axis_name]
        index = int(query.get("index", [image.shape[axis] // 2])[0])
        center = float(query.get("center", [400])[0])
        width = max(float(query.get("width", [1800])[0]), 1)
        body = slice_png(image, reference, comparison, axis, index, center, width)
        self.send_body(200, body, "image/png")

    def handle_mesh(self, query):
        _, _, _, _, reference, comparison, spacing, _ = self.selected(query)
        self.send_json(
            {
                "reference": mesh(reference, spacing) if reference is not None else None,
                "comparison": mesh(comparison, spacing) if comparison is not None else None,
            }
        )

    def log_message(self, message, *args):
        print(f"[dashboard] {message % args}")


def run_check(initial_case: str):
    matches = [
        (dataset, group)
        for dataset, cases in CATALOG.items()
        if (group := cases.get(initial_case))
    ]
    if not matches:
        raise FileNotFoundError(f"No discovered case named {initial_case}")
    dataset, group = matches[0]
    if not group["images"]:
        raise FileNotFoundError(f"No image discovered for {initial_case}")
    image_path = Path(group["images"][0]["path"])
    mask_paths = [Path(item["path"]) for item in group["masks"][:2]]
    image, masks, _ = aligned_arrays(image_path, mask_paths)
    slice_png(
        image,
        masks[0] if masks else None,
        masks[1] if len(masks) > 1 else None,
        0,
        image.shape[0] // 2,
        400,
        1800,
    )
    print(
        f"Dashboard check passed: {len(CATALOG)} datasets; "
        f"{dataset}/{initial_case}; {len(group['images'])} image(s), "
        f"{len(group['masks'])} mask(s)."
    )


def main():
    args = parse_args()
    if args.check:
        run_check(args.case)
        return
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.initial_case = args.case
    url = f"http://{args.host}:{args.port}/"
    print(f"Volume dashboard: {url}")
    print("In VS Code, forward this port and open the forwarded address.")
    print("Press CTRL+C to stop the dashboard.")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
