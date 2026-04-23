from pathlib import Path
import streamlit as st
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🤖",
    layout="wide"
)

# ---------------- TITLE ---------------- #
st.markdown("""
<h1 style='text-align: center; color: #00ADB5;'>
🔍 AI Surveillance System (YOLOv8)
</h1>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Model Config")

model_type = st.sidebar.selectbox(
    "Select Model",
    config.DETECTION_MODEL_LIST
)

confidence = st.sidebar.slider("Confidence", 30, 100, 50) / 100

model_path = Path(config.DETECTION_MODEL_DIR, model_type)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_cached_model(path):
    try:
        return load_model(path)
    except Exception as e:
        return None

model = load_cached_model(model_path)

if model is None:
    st.error(f"❌ Error loading model at: {model_path}")
    st.stop()

# ---------------- SOURCE ---------------- #
st.sidebar.header("📥 Input Source")

source = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

# ---------------- DASHBOARD ---------------- #
st.sidebar.header("🚨 Dashboard")

col1, col2, col3 = st.sidebar.columns(3)

col1.metric("Alerts", 0)
col2.metric("People", 0)
col3.metric("Intrusions", 0)

# ---------------- RUN ---------------- #
if source == "Image":
    infer_uploaded_image(confidence, model)

elif source == "Video":
    infer_uploaded_video(confidence, model)

elif source == "Webcam":
    infer_uploaded_webcam(confidence, model)

else:
    st.warning("Select a valid source")