import os
import csv
import shutil
import sys
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# Make 'src' importable when running from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.prediction import predict_image, load_label_map  # noqa: E402
from src.preprocessing import get_generators  # noqa: E402

app = FastAPI(
    title="Animal Image Classification API",
    description="Predict, upload & retrain an animal classifier.",
)

# Allow frontend / local tools to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

BASE_MODEL_PATH = os.path.join(MODEL_DIR, "base_animal_cnn.h5")
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "fine_tuned_animal_cnn.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
METADATA_CSV = os.path.join(MODEL_DIR, "uploaded_metadata.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# Load model (fine-tuned if it exists, otherwise base)
if os.path.exists(FINE_TUNED_MODEL_PATH):
    CURRENT_MODEL_PATH = FINE_TUNED_MODEL_PATH
else:
    CURRENT_MODEL_PATH = BASE_MODEL_PATH

if not os.path.exists(CURRENT_MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {CURRENT_MODEL_PATH}. "
        "Make sure you copied base_animal_cnn.h5 from Colab into /models."
    )

model = load_model(CURRENT_MODEL_PATH)
label_map = load_label_map(CLASS_INDICES_PATH, TRAIN_DIR)


@app.get("/health")
def health():
    """
    Simple health check endpoint showing model path.
    """
    return {
        "status": "ok",
        "model_path": CURRENT_MODEL_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of a single uploaded image.
    """
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    class_idx, confidence, probs = predict_image(model, temp_path)
    predicted_label = label_map.get(class_idx, str(class_idx))

    return {
        "predicted_class": predicted_label,
        "class_index": class_idx,
        "confidence": confidence,
        "probabilities": probs,
    }


@app.post("/upload-data")
async def upload_data(
    file: UploadFile = File(...),
    label: str = Form(...),
):
    """
    Upload a labeled image to be used for future retraining.
    The image is saved under data/train/<label>/.
    We also log the upload into a simple CSV "metadata" file.
    """
    label = label.strip()
    if not label:
        return {"error": "Label must not be empty."}

    # Save into training directory
    label_dir = os.path.join(TRAIN_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    save_path = os.path.join(label_dir, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Log metadata
    header_needed = not os.path.exists(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["filepath", "label"])
        writer.writerow([save_path, label])

    return {
        "message": "File uploaded and saved for retraining.",
        "saved_to": save_path,
    }


@app.post("/retrain")
def retrain(epochs: Optional[int] = 3):
    """
    Retrain (fine-tune) the model using the updated data/train directory.
    Starts from the current saved model (fine-tuned or base).
    """
    global model, CURRENT_MODEL_PATH, label_map

    # Choose starting checkpoint (fine-tuned if exists, else base)
    start_path = FINE_TUNED_MODEL_PATH if os.path.exists(FINE_TUNED_MODEL_PATH) else BASE_MODEL_PATH
    if not os.path.exists(start_path):
        return {"error": f"Cannot retrain. Model file not found at {start_path}."}

    # Reload model from disk
    retrain_model = load_model(start_path)

    # Rebuild generators (this will also update class_indices.json)
    train_gen, val_gen, _ = get_generators(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        save_class_indices_path=CLASS_INDICES_PATH,
    )

    retrain_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
    )

    # Save fine-tuned model
    retrain_model.save(FINE_TUNED_MODEL_PATH)
    CURRENT_MODEL_PATH = FINE_TUNED_MODEL_PATH
    model = retrain_model

    # Reload label map
    label_map = load_label_map(CLASS_INDICES_PATH, TRAIN_DIR)

    return {
        "message": "Retraining complete.",
        "new_model_path": CURRENT_MODEL_PATH,
        "epochs": epochs,
    }
