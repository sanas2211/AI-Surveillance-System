from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import numpy as np

from modules.face_recognition import recognize_faces
from modules.intrusion import draw_zone, check_intrusion
from modules.logger import save_log

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(str(model_path))
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


# ---------------- PROCESS DETECTIONS ---------------- #
def process_detections(results, model):
    counts = {}

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            counts[name] = counts.get(name, 0) + 1

    return counts


# ---------------- CORE FRAME PROCESSING ---------------- #
def process_frame(frame, model, conf):
    results = model.predict(frame, conf=conf)
    frame = results[0].plot()

    # Draw intrusion zone
    frame = draw_zone(frame)

    # Face recognition
    faces = recognize_faces(frame)
    for name, top, right, bottom, left in faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Intrusion detection
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            if check_intrusion((x1, y1, x2, y2)):
                save_log("Intrusion Detected")
                cv2.putText(frame, "INTRUSION!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    counts = process_detections(results, model)

    return frame, counts


# ---------------- IMAGE ---------------- #
def infer_uploaded_image(conf, model):
    source_img = st.sidebar.file_uploader(
        "Choose an image",
        type=("jpg", "jpeg", "png")
    )

    col1, col2 = st.columns(2)

    if source_img:
        image = Image.open(source_img)
        frame = np.array(image)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Processing..."):
                processed_frame, counts = process_frame(frame, model, conf)

                with col2:
                    st.image(processed_frame, channels="BGR",
                             caption="Detected Image", use_column_width=True)

                    st.subheader("Detected Objects")
                    if counts:
                        for k, v in counts.items():
                            st.write(f"{k}: {v}")
                    else:
                        st.write("No objects detected")


# ---------------- VIDEO ---------------- #
def infer_uploaded_video(conf, model):
    source_video = st.sidebar.file_uploader("Choose a video")

    if source_video:
        st.video(source_video)

        if st.button("Run Detection"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(source_video.read())

            cap = cv2.VideoCapture(tfile.name)

            st_frame = st.empty()
            st_text = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, counts = process_frame(frame, model, conf)

                st_frame.image(processed_frame, channels="BGR")

                text = " | ".join([f"{k}:{v}" for k, v in counts.items()])
                st_text.write(text if text else "No objects")

            cap.release()


# ---------------- WEBCAM ---------------- #
def infer_uploaded_webcam(conf, model):
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        st_frame = st.empty()
        st_text = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, counts = process_frame(frame, model, conf)

            st_frame.image(processed_frame, channels="BGR")

            text = " | ".join([f"{k}:{v}" for k, v in counts.items()])
            st_text.write(text if text else "No objects")

        cap.release()