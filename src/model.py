from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


def build_animal_cnn(
    input_shape=(128, 128, 3),
    num_classes=15,
    l2_reg=1e-4,
    lr=1e-3,
):
    """
    Custom CNN model with L2 regularization and dropout.
    """

    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
                input_shape=input_shape,
            ),
            MaxPooling2D(),
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            MaxPooling2D(),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            MaxPooling2D(),
            Flatten(),
            Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
