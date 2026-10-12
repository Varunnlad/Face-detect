import json
from pathlib import Path
import cv2, numpy as np
import streamlit as st
import tensorflow as tf
from liveness_utils import prepare, spoof_probability
import time


MODEL_PATH = "selfie_prototype/liveness_mnv3.keras"
CONFIG_PATH = "selfie_prototype/liveness_config.json"   

SAVE_DIR = Path("selfie_prototype/submissions")
(SAVE_DIR / "real").mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "spoof").mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Live selfie check")
st.title("Live selfie check")
st.caption("Take a selfie or upload a photo. A face check runs first; then the model says REAL or SPOOF.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model at '{MODEL_PATH}'. Run the notebook first.\n\n{e}")
    st.stop()

cfg = json.load(open(CONFIG_PATH)) if Path(CONFIG_PATH).exists() else {}
threshold = st.sidebar.slider(
    "Spoof threshold", 0.05, 0.95, float(cfg.get("threshold", 0.5)), 0.05,
    help="Flag as SPOOF when P(spoof) is above this. Lower = catches more cheats but flags more honest photos.")
if cfg:
    st.sidebar.caption(f"On the test photos that passed the face gate, the saved threshold caught "
                       f"{cfg['test_spoof_caught']:.0%} of spoofs and passed {cfg['test_real_passed']:.0%} of real photos "
                       f"({cfg['n_spoof_test']} spoof test photos).")
else:
    st.sidebar.warning("No config file found, so the threshold defaults to 0.5.")


def analyse(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.warning("Could not read that image.")
        return
    ok, reason, crop = prepare(img)
    if not ok:
        st.error(f"Photo rejected: {reason}. Please retake it.")
        return
    p = spoof_probability(model, crop)
    verdict = "spoof" if p > threshold else "real"

    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{stamp}_{p:.2f}.jpg"
    cv2.imwrite(str(SAVE_DIR / verdict / fname), img)     

    if verdict == "spoof":
        st.error("SPOOF - flag for review")
    else:
        st.success("REAL - passed")
    st.metric("P(spoof)", f"{p:.2f}")
    st.progress(min(max(p, 0.0), 1.0))
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Face crop the model saw", width=160)


tab_cam, tab_up = st.tabs(["Camera", "Upload"])
with tab_cam:
    photo = st.camera_input("Take a selfie")
    if photo is None:
        st.info("Press the take-photo button above. Your browser will ask for camera permission - click Allow.")
    else:
        analyse(photo.getvalue())
with tab_up:
    up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if up is not None:
        analyse(up.getvalue())