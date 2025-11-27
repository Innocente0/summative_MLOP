import os
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Base URL of your API
API_URL = "http://localhost:8000"  # change to your deployed URL in production


st.set_page_config(page_title="Animal Classifier", layout="wide")
st.title("🐾 Animal Image Classification Pipeline")

# Sidebar: model status
with st.sidebar:
    st.header("Model Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success("API is running")
            st.caption(f"Model path: {data.get('model_path', 'N/A')}")
        else:
            st.error("API returned an error")
    except Exception as e:
        st.error("API not reachable")
        st.write(str(e))


tab_predict, tab_retrain, tab_insights = st.tabs(
    ["🔍 Predict", "📤 Upload & Retrain", "📊 Data Insights"]
)

# ---------------------- PREDICT TAB ----------------------
with tab_predict:
    st.subheader("Predict Animal Class from Image")

    file = st.file_uploader(
        "Upload an animal image", type=["jpg", "jpeg", "png"]
    )

    if file is not None:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            try:
                files = {"file": (file.name, file.getvalue(), file.type)}
                resp = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=30,
                )
                if resp.status_code == 200:
                    res = resp.json()
                    st.success(
                        f"Prediction: {res['predicted_class']} "
                        f"(confidence: {res['confidence']:.2f})"
                    )
                else:
                    st.error(f"Prediction failed: {resp.text}")
            except Exception as e:
                st.error("Prediction request failed")
                st.write(str(e))


# ---------------------- RETRAIN TAB ----------------------
with tab_retrain:
    st.subheader("Upload New Data and Trigger Retraining")

    label = st.text_input("Label / Class (e.g., 'Bear')")

    new_files = st.file_uploader(
        "Upload one or more images for this label",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if new_files and label:
        if st.button("Upload for Retraining"):
            for f in new_files:
                files = {"file": (f.name, f.getvalue(), f.type)}
                data = {"label": label}
                resp = requests.post(
                    f"{API_URL}/upload-data",
                    files=files,
                    data=data,
                    timeout=60,
                )
            st.success("All files uploaded for retraining.")

    st.markdown("---")
    st.write("When you have added enough new data, click below:")

    epochs = st.number_input(
        "Retraining epochs", min_value=1, max_value=10, value=3, step=1
    )

    if st.button("Trigger Retraining"):
        try:
            params = {"epochs": int(epochs)}
            resp = requests.post(f"{API_URL}/retrain", params=params, timeout=600)
            if resp.status_code == 200:
                st.success("Retraining complete!")
                st.json(resp.json())
            else:
                st.error(f"Retraining failed: {resp.text}")
        except Exception as e:
            st.error("Retraining request failed")
            st.write(str(e))


# ---------------------- DATA INSIGHTS TAB ----------------------
with tab_insights:
    st.subheader("Dataset Insights (Training Set)")

    train_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train")

    if os.path.exists(train_dir):
        class_counts = {}
        for cls in os.listdir(train_dir):
            cls_path = os.path.join(train_dir, cls)
            if os.path.isdir(cls_path):
                class_counts[cls] = len(
                    [
                        f
                        for f in os.listdir(cls_path)
                        if os.path.isfile(os.path.join(cls_path, f))
                    ]
                )

        if class_counts:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(class_counts.keys(), class_counts.values())
            plt.xticks(rotation=45, ha="right")
            ax.set_ylabel("Number of Images")
            ax.set_title("Class Distribution in Training Data")
            st.pyplot(fig)

            st.markdown(
                """
                **Interpretation example:**  
                Classes with significantly fewer images may be harder for the model 
                to learn and can have lower precision/recall. Adding more samples 
                for underrepresented classes can improve performance.
                """
            )
        else:
            st.info("No images found in training directory.")
    else:
        st.info("Training directory not found. Make sure 'data/train' exists.")
