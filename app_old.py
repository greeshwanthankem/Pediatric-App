from pathlib import Path
import traceback

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import load_model, run_inference, prepare_display_image
from utils import (
    draw_detections,
    generate_summary,
    save_uploaded_file,
    get_display_class_name,
)

st.set_page_config(page_title="Pediatric Fracture Detection", layout="wide")

st.title("Pediatric Bone Fracture Detection Web App")
st.write("Upload a pediatric wrist X-ray image to detect possible fracture-related regions.")

with st.sidebar:
    st.header("Model Settings")
    weights_path = st.text_input(
        "Model weights path",
        value="OurModel.pt",
        help="Provide the local path to your trained YOLO weights file (.pt).",
    )
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Upload an image to start detection.")
    st.stop()


def show_streamlit_exception(title: str, exc: Exception) -> None:
    st.error(f"{title}: {exc}")
    st.exception(exc)
    traceback.print_exc()


left_col, right_col = st.columns(2)

try:
    # Do NOT force convert("RGB") here
    original_image = Image.open(uploaded_file)
except (UnidentifiedImageError, OSError, ValueError) as exc:
    show_streamlit_exception("Invalid image file", exc)
    st.stop()

try:
    # Use normalized/prepared version for display and drawing
    display_image = prepare_display_image(original_image)
except Exception as exc:
    show_streamlit_exception("Failed to prepare image for display", exc)
    st.stop()

with left_col:
    st.subheader("Original Image")
    st.image(display_image, width="stretch")

try:
    saved_image_path = save_uploaded_file(uploaded_file, "outputs")
    st.caption(f"Uploaded image saved to: {saved_image_path}")
except Exception as exc:
    st.warning(f"Image could not be saved to disk: {exc}")

weights_path = weights_path.strip()
if not weights_path:
    st.error("Model weights path is empty. Please provide a valid .pt file path in the sidebar.")
    st.stop()

weights_file = Path(weights_path)
if not weights_file.exists() or not weights_file.is_file():
    st.error(f"Weights file not found: {weights_file.resolve()}")
    st.stop()

with st.spinner("Loading model and running inference..."):
    try:
        model = load_model(str(weights_file))
    except Exception as exc:
        show_streamlit_exception("Model loading failed", exc)
        st.stop()

    try:
        detections = run_inference(model, original_image, conf_threshold=conf_threshold)
    except Exception as exc:
        show_streamlit_exception("Inference failed", exc)
        st.stop()

try:
    annotated_image = draw_detections(display_image, detections)
except Exception as exc:
    show_streamlit_exception("Failed to draw detections", exc)
    st.stop()

summary = generate_summary(detections)

with right_col:
    st.subheader("Annotated Result")
    st.image(annotated_image, width="stretch")

st.subheader("Detection Summary")
st.write(
    {
        "total_detections": summary["total_detections"],
        "detected_class_names": summary["detected_class_names"],
        "highest_confidence": summary["highest_confidence"],
        "status_message": summary["status_message"],
    }
)

st.subheader("Detections Table")
if detections:
    table_rows = [
        {
            "class_name": get_display_class_name(det["class_name"]),
            "confidence": round(float(det["confidence"]), 4),
            "bbox": [round(float(v), 2) for v in det["bbox"]],
        }
        for det in detections
    ]
    st.dataframe(pd.DataFrame(table_rows), width="stretch")
else:
    st.info("No detections found for the selected confidence threshold.")