"""Streamlit app for trying the trained office-item detector.

You can upload a still image or use a connected camera to see detections.
"""
from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from threading import Lock
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "train" / "final_model" / "best.pt"
UPLOAD_EXTENSIONS = [
    "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif",
    "heic", "heif", "avif",
]
register_heif_opener()
st.set_page_config(page_title="Office Vision", page_icon="◉", layout="wide")


@st.cache_resource(show_spinner="Loading detector…")
def load_model(model_path: str):
    # Cache the model so Streamlit doesn't reload it on every page update.
    return YOLO(model_path)


def draw_predictions(image: np.ndarray, model, confidence: float, imgsz: int):
    # The model output is BGR; Streamlit needs RGB to show the colors correctly.
    result = model.predict(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
    plotted = result.plot()
    boxes = result.boxes
    detections = []
    if boxes is not None:
        names = result.names
        for box in boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class": names[class_id],
                "confidence": float(box.conf.item()),
                "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            })
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), detections


def load_uploaded_image(uploaded_file) -> Image.Image:
    """Open an upload, fix its orientation, and convert it to RGB."""
    try:
        with Image.open(uploaded_file) as source:
            # Use the first frame for animations and apply the photo's saved orientation.
            source.seek(0)
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError("The uploaded file is not a readable supported image.") from exc


class DetectorProcessor(VideoProcessorBase):
    def __init__(self, model_path: str, confidence: float, imgsz: int):
        self.model = load_model(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.lock = Lock()
        self.fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.detections = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        # The camera thread and page can both access this data, so use the lock here.
        with self.lock:
            result = self.model.predict(
                image, conf=self.confidence, imgsz=self.imgsz, verbose=False
            )[0]
            output = result.plot()
            now = time.perf_counter()
            elapsed = now - self.last_frame_time
            if elapsed > 0:
                # Smooth the FPS value so it doesn't jump around as much.
                self.fps = 0.85 * self.fps + 0.15 / elapsed if self.fps else 1 / elapsed
            self.last_frame_time = now
            self.detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    self.detections.append({
                        "class": result.names[class_id],
                        "confidence": float(box.conf.item()),
                        "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    })
        return av.VideoFrame.from_ndarray(output, format="bgr24")


st.markdown("""
<style>
:root { --ink:#eaf0ef; --muted:#91a4a0; --line:rgba(190,220,211,.14); --mint:#b7f36b; }
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% 0%, #16302e 0%, #0b1518 45%, #091114 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1440px; padding-top:2.2rem; }
.eyebrow { color:var(--mint); font:500 11px 'DM Mono',monospace; letter-spacing:.18em; text-transform:uppercase; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid var(--line); padding:12px 0 26px; margin-bottom:25px; }
.hero h1 { font-size:clamp(36px,5vw,62px); letter-spacing:-.065em; line-height:.98; margin:12px 0 0; color:#f0f5f1; }
.hero p { color:var(--muted); max-width:410px; line-height:1.7; margin:0 0 4px; font-size:14px; }
.panel { border:1px solid var(--line); background:rgba(18,31,34,.68); border-radius:16px; padding:18px 20px; }
.panel-title { font:500 11px 'DM Mono',monospace; color:#97aca7; letter-spacing:.12em; text-transform:uppercase; margin-bottom:16px; }
.metric { background:rgba(255,255,255,.035); border:1px solid var(--line); border-radius:12px; padding:13px 15px; }
.metric small { color:var(--muted); font:11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
.metric strong { display:block; color:var(--mint); font-size:22px; margin-top:4px; }
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap:8px; }
div[data-testid="stRadio"] label { border:1px solid var(--line); border-radius:10px; padding:9px 14px; background:#142124; }
.stButton button, [data-testid="stFileUploader"] section { border-radius:10px; }
div[data-testid="stCaptionContainer"] { color:#91a4a0; }
</style>
<div class="hero">
 <div><div class="eyebrow">Edge vision / 01</div><h1>Office object<br>recognition.</h1></div>
 <p>Live detection and single-image inspection, powered by your local YOLO model. Inference stays on this device.</p>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every="1s")
def render_camera_status(ctx):
    st.markdown("<div class='panel-title'>Session status</div>", unsafe_allow_html=True)
    if ctx.video_processor:
        processor = ctx.video_processor
        # Copy the current values while the camera thread is updating them.
        with processor.lock:
            current_detections = list(processor.detections)
            fps = processor.fps
        a, b = st.columns(2)
        a.metric("Processing rate", f"{fps:.1f} FPS")
        b.metric("Objects", len(current_detections))
        if current_detections:
            st.markdown("<div class='panel-title' style='margin-top:24px'>Current detections</div>", unsafe_allow_html=True)
            for item in current_detections:
                st.markdown(f"**{item['class'].replace('_', ' ').title()}** · {item['confidence']:.0%}")
        else:
            st.caption("No objects above the confidence threshold.")
    else:
        st.markdown("<div class='metric'><small>Camera</small><strong>Ready when you are</strong></div>", unsafe_allow_html=True)
        st.caption("Press **START** to grant camera access and begin local inference.")

with st.sidebar:
    # These settings are shared by camera and uploaded-image detection.
    st.markdown("<div class='eyebrow'>System controls</div>", unsafe_allow_html=True)
    st.markdown("### Detector setup")
    model_path = st.text_input("Checkpoint path", value=os.environ.get("OFFICE_DETECTOR_MODEL", str(DEFAULT_MODEL)))
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    imgsz = st.select_slider("Inference size", options=[224, 256, 288, 320, 416, 480], value=320)
    st.caption("Smaller input sizes can improve Raspberry Pi throughput.")
    st.markdown("---")
    st.caption("Camera access requires HTTPS or localhost in most browsers. Select the Pi camera when launching this app on the Raspberry Pi.")

if not Path(model_path).is_file():
    # There's no point showing the controls if the model file isn't there.
    st.error(f"Detector checkpoint not found: `{model_path}`")
    st.info("Copy the trained detector to `train/final_model/best.pt`, or set `OFFICE_DETECTOR_MODEL` to its path. The checkpoint is excluded from Git.")
    st.stop()

try:
    # Load once here so both input modes use the same model.
    detector = load_model(model_path)
except Exception as exc:
    st.error(f"Could not load the detector: {exc}")
    st.stop()

mode = st.radio("Operating mode", ["Live camera", "Image file"], horizontal=True, label_visibility="collapsed")
if mode == "Live camera":
    # Pass the selected settings into the processor when the camera starts.
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.markdown("<div class='panel-title'>Live camera feed</div>", unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="office-detector",
            video_processor_factory=lambda: DetectorProcessor(model_path, confidence, imgsz),
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": []},
            async_processing=True,
        )
    with right:
        render_camera_status(ctx)
    st.caption("Video frames are processed locally by the loaded model. Actual FPS depends on the camera, hardware, and model checkpoint.")
else:
    uploaded = st.file_uploader(
        "Choose an image",
        type=UPLOAD_EXTENSIONS,
        help="Supports JPEG, PNG, WebP, BMP, TIFF, GIF, HEIC/HEIF, and AVIF.",
    )
    if uploaded:
        try:
            # Convert the upload for the model; this doesn't change the original file.
            image = load_uploaded_image(uploaded)
        except ValueError as exc:
            st.error(str(exc))
        else:
            col_image, col_results = st.columns([1.55, 1], gap="large")
            with st.spinner("Detecting objects…"):
                # Convert to BGR, the format Ultralytics expects for image arrays.
                image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                annotated, detections = draw_predictions(image_bgr, detector, confidence, imgsz)
            with col_image:
                st.markdown("<div class='panel-title'>Detection result</div>", unsafe_allow_html=True)
                st.image(annotated, width="stretch")
            with col_results:
                st.markdown("<div class='panel-title'>Objects found</div>", unsafe_allow_html=True)
                st.metric("Detection count", len(detections))
                if detections:
                    st.dataframe([
                        {"Class": d["class"].replace("_", " ").title(), "Confidence": f"{d['confidence']:.1%}", "Box (x1, y1, x2, y2)": ", ".join(map(str, d["xyxy"]))}
                        for d in detections
                    ], hide_index=True, width="stretch")
                else:
                    st.info("No objects found. Try lowering the confidence threshold.")
                buffer = BytesIO()
                Image.fromarray(annotated).save(buffer, format="PNG")
                # Make the download in memory instead of saving a temporary file.
                st.download_button("Download annotated image", data=buffer.getvalue(), file_name="office-detections.png", mime="image/png")
    else:
        st.markdown("Upload one image to inspect its objects, confidence scores, and bounding boxes.")
