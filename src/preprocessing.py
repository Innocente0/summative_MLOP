import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def get_generators(
    train_dir="data/train",
    test_dir="data/test",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    save_class_indices_path="models/class_indices.json",
):
    """
    Create training, validation and test generators.
    This mirrors the preprocessing you used in Colab.
    """

    # Train + validation generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    # Test generator (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # Save class_indices for prediction label mapping
    os.makedirs(os.path.dirname(save_class_indices_path), exist_ok=True)
    with open(save_class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f)

    return train_gen, val_gen, test_gen
