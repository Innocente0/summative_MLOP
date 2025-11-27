# Animals Image Classification ML Pipeline

This project implements an end-to-end Machine Learning pipeline for **multi-class animal image classification**.  
It includes:

- Offline model training in a Jupyter/Colab notebook  
- A custom CNN model saved as a pre-trained model  
- FastAPI backend for prediction and retraining  
- Streamlit web UI for interaction and visualizations  
- Locust-based load testing to simulate flood requests to the model API  
- Docker-ready structure for deployment on a cloud platform

---

## 🔗 URLs

> Replace the placeholders below with your actual deployed URLs.

- **API base URL (FastAPI)**: `https://your-api-url.com`
- **Web UI (Streamlit)**: `https://your-ui-url.com`

---

## 📌 Project Description

The goal of this project is to demonstrate the full Machine Learning lifecycle on **non-tabular data (images)**:

1. **Data Acquisition & Preprocessing**
   - Animal images organized into class folders (e.g., Bear, Bird, Cat, etc.).
   - Dataset split into `train` and `test` sets.
   - Preprocessing includes resizing, normalization (0–1), and augmentation (rotation, zoom, shifts, flips).

2. **Model Creation & Optimization**
   - Custom Convolutional Neural Network (CNN) built with TensorFlow/Keras.
   - Optimization techniques used:
     - L2 regularization
     - Dropout
     - Adam optimizer
     - Early stopping
     - Best-model checkpointing
   - The trained model is saved as `models/base_animal_cnn.h5` and acts as a **custom pre-trained model** for retraining.

3. **Evaluation**
   - The notebook evaluates the model using multiple metrics:
     - Loss
     - Accuracy
     - F1-score
     - Precision
     - Recall
   - Additional insights with training/validation curves and confusion matrix.

4. **Serving & Retraining**
   - FastAPI backend with endpoints to:
     - Predict from a single image (`/predict`)
     - Upload new labeled images for retraining (`/upload-data`)
     - Trigger model retraining (`/retrain`)
     - Check model health/uptime (`/health`)
   - Retraining continues from the saved pre-trained model and updates the deployed model.

5. **User Interface**
   - Streamlit web app that allows users to:
     - Upload an image and get a prediction
     - Upload bulk images for retraining
     - Trigger retraining via a button
     - View dataset insights (e.g., class distribution)

6. **Flood Request Simulation**
   - Locust is used to simulate concurrent users sending requests to the `/predict` endpoint.
   - Results show how latency and throughput change with different numbers of Docker containers.

---

## 📁 Project Structure

```text
project_name/
│
├── README.md
│
├── notebook/
│   └── animal_pipeline.ipynb          # Training, evaluation, and saving model
│
├── src/
│   ├── preprocessing.py               # Data generators & preprocessing
│   ├── model.py                       # Model architecture & compilation
│   └── prediction.py                  # Single-image prediction utilities
│
├── api/
│   ├── app.py                         # FastAPI backend (predict, upload, retrain)
│   ├── requirements.txt
│   └── Dockerfile
│
├── ui/
│   └── streamlit_app.py               # Web UI for prediction & retraining
│
├── locust/
│   └── locustfile.py                  # Flood request simulation config
│
├── data/
│   ├── train/                         # Training data (image folders per class)
│   └── test/                          # Test data (image folders per class)
│
└── models/
    ├── base_animal_cnn.h5             # Custom pre-trained model
    └── fine_tuned_animal_cnn.h5       # (Optional) model after retraining
