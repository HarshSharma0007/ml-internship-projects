import streamlit as st
import os
from PIL import Image
from utils.inference import predict_class, CLASS_NAMES
from utils.captum_utils import explain_image
import pandas as pd

# 🐾 Emoji map
ANIMAL_EMOJIS = {
    "Bear": "🐻", "Bird": "🐦", "Cat": "🐱", "Cow": "🐮", "Deer": "🦌",
    "Dog": "🐶", "Dolphin": "🐬", "Elephant": "🐘", "Giraffe": "🦒", "Horse": "🐴",
    "Kangaroo": "🦘", "Lion": "🦁", "Panda": "🐼", "Tiger": "🐯", "Zebra": "🦓"
}

# 🔢 Visual confidence bar
def confidence_bar(p):
    return "█" * (p // 10) + "░" * (10 - p // 10)

# 🌐 Layout
st.set_page_config(page_title="Animal Classifier 🐾", layout="wide")

# 🧠 Session state
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None

# ⚙️ Attribution selector
st.sidebar.markdown("### ⚙️ Attribution Method")
attr_method = st.sidebar.radio("Choose method", ["Integrated Gradients", "Saliency"])

# 📦 Class list
st.sidebar.markdown("### 📦 Classes")
cols = st.sidebar.columns(3)
for i, cls in enumerate(CLASS_NAMES):
    cols[i % 3].markdown(f"{ANIMAL_EMOJIS.get(cls, '')} {cls}")

# 🖼️ Sample gallery (2-column layout)
st.sidebar.markdown("### 🖼️ Sample Gallery")
SAMPLE_DIR = os.path.join("assets", "sample_images")
sample_paths = sorted([
    os.path.join(SAMPLE_DIR, f)
    for f in os.listdir(SAMPLE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

sample_cols = st.sidebar.columns(2)
for i, path in enumerate(sample_paths):
    img = Image.open(path)
    with sample_cols[i % 2]:
        st.image(img, width=100)
        if st.button("Select", key=f"sample_{i}"):
            st.session_state.uploaded = img
            st.rerun()

# 📤 Upload widget
uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.session_state.uploaded = Image.open(uploaded_file).convert("RGB")

# 🔁 Reset button
if st.button("🔄 Reset Interface"):
    st.session_state.uploaded = None
    st.rerun()

# 🔮 Prediction
image = st.session_state.uploaded
if image is not None:
    with st.spinner("Generating prediction and attribution..."):
        pred_class, top_probs = predict_class(image)
        target_idx = CLASS_NAMES.index(pred_class)
        heatmap = explain_image(image, target_idx, method=attr_method)

    col1, col2, col3 = st.columns([1.2, 1.5, 1.5])

    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🔮 Prediction")
        st.markdown(
            f"<h2 style='text-align:center; color:#FF4B4B'>{ANIMAL_EMOJIS.get(pred_class, '')} {pred_class}</h2>",
            unsafe_allow_html=True
        )
        top3 = list(top_probs.items())[:3]
        df = pd.DataFrame([
            {"Class": f"{ANIMAL_EMOJIS.get(cls, '')} {cls}", "Confidence": int(p * 100)}
            for cls, p in top3
        ])
        df["Confidence Bar"] = df["Confidence"].apply(confidence_bar)
        st.dataframe(
            df[["Confidence Bar", "Confidence"]].set_index(df["Class"]).style.format({"Confidence": "{}%"})
        )

    with col3:
        st.subheader("🧠 Attribution")
        st.image(heatmap, caption=f"{ANIMAL_EMOJIS.get(pred_class, '')} {pred_class} via {attr_method}", use_column_width=True)

else:
    st.info("👈 Upload an image or select a sample to get started.")

st.markdown("---")
