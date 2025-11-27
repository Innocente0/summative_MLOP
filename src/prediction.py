import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = (128, 128)


def load_label_map(
    json_path="models/class_indices.json",
    train_dir="data/train",
):
    """
    Load label mapping from JSON if available.
    If not, build it from the directory structure in data/train.
    Returns: dict index -> class_name
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            class_indices = json.load(f)
        idx2class = {v: k for k, v in class_indices.items()}
        return idx2class

    # Fallback: build from data/train folder names
    classes = [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    classes = sorted(classes)
    idx2class = {idx: name for idx, name in enumerate(classes)}
    return idx2class


def predict_image(model, img_file, img_size=IMG_SIZE):
    """
    Load an image file and return (class_idx, confidence, probs_list).
    """
    img = image.load_img(img_file, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    return class_idx, confidence, preds[0].tolist()
