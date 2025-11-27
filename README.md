# Animals Image Classification ML Pipeline

This project implements an end-to-end Machine Learning pipeline for **multi-class animal image classification** using a custom Convolutional Neural Network (CNN).  
It includes:

- Offline model training in a Jupyter notebook  
- A custom CNN model saved as a pre-trained model  
- FastAPI backend for prediction and retraining  
- Streamlit web UI for interaction and visualizations  
- Locust-based load testing to simulate flood requests to the model API  
- Docker-ready structure for deployment on a cloud platform

## URLs

- **Video Demo**: https://youtu.be/feqH8974cfs
- **Github Link**: https://github.com/Innocente0/summative_MLOP.git 
- **API base URL (FastAPI)**: `http://127.0.0.1:8000/docs#/`
- **Web UI (Streamlit)**: `http://localhost:8501/`

## Project Description

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

## Project Structure

```text
project_name/
│
├── README.md
│
├── notebook/
│   └── animal_pipeline.ipynb          
│
├── src/
│   ├── preprocessing.py              
│   ├── model.py                      
│   └── prediction.py                  
│
├── api/
│   ├── app.py                        
│   ├── requirements.txt
│   └── Dockerfile
│
├── ui/
│   └── streamlit_app.py            
├── locust/
│   └── locustfile.py              
│
├── data/
│   ├── train/                      
│   └── test/                       
│
└── models/
    ├── base_animal_cnn.h5             
    └── fine_tuned_animal_cnn.h5      

 How To Set It Up

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Innocente0/summative_MLOP.git

create and activate a virtual enivornment:
 - Python -m venv venv
 - pip install -r requirements.txt
 - python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
 - streamlit run ui/streamlit_app.py
 - locust -f locust/locustfile.py --host=http://127.0.0.1:8000

