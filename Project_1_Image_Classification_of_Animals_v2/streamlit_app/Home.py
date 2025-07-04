import streamlit as st
import torch
from PIL import Image
from utils.inference import predict_class
from utils.captum_utils import explain_image

st.set_page_config(page_title="Animal Classifier 🐾", layout="wide")
st.title("🐾 Animal Classifier with Attribution")
st.write("Upload an image to see the predicted class and what influenced the model's decision.")

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        pred_class, top_probs = predict_class(image)
        heatmap = explain_image(image, pred_class)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Class")
            st.write(f"**{pred_class}**")
            st.bar_chart(top_probs)

        with col2:
            st.subheader("Attribution Heatmap")
            st.image(heatmap, use_column_width=True)
