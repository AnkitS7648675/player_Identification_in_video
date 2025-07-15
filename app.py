import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(page_title="Player & Ball Detection", layout="centered")
st.title("🎯 YOLOv11 Player & Ball Detection on Video")

# Upload video
uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

# Load your trained model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your pretrained player/ball detection model

if uploaded_video is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.success("📼 Video uploaded. Running detection...")

    # Load model
    model = load_model()

    # Read input video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output path
    output_path = os.path.join(tempfile.gettempdir(), "detected_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Frame processing
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()

    SKIP = 10
    frame_idx = 0
    annotated_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SKIP == 0:
            results = model.predict(frame, conf=0.3, imgsz=320)
            annotated_frame = results[0].plot()

        if annotated_frame is not None:
            out.write(annotated_frame)

        # Update progress bar
        frame_idx += 1
        progress = int((frame_idx / total_frames) * 100)
        progress_bar.progress(min(progress, 100))
        status_text.text(f"🔄 Processing frame {frame_idx} of {total_frames}")

    cap.release()
    out.release()

    st.success("✅ Detection complete!")
        
    with open(output_path, "rb") as file:
            st.download_button(
            label="📥 Download Detected Video",
            data=file,
            file_name="player_detection_output.mp4",
            mime="video/mp4"
        )
