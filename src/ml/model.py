from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def create_model(
    num_classes: int = 2,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    base: str = "mobilenet_v2",
    weights: str | None = None,  
    train_base: bool = False,
    learning_rate: float = 1e-3,
    compile_model: bool = True,
    metrics: list | None = None,
    **kwargs,
) -> tf.keras.Model:
    """
    Returns a tf.keras.Model for image classification.
    Accepts extra **kwargs to stay compatible with train.py call signatures.
    """
    if base.lower() == "mobilenet_v2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape, weights=weights
        )
    else:
        # Simple fallback CNN if base is unknown
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        activation = "sigmoid" if num_classes == 1 else "softmax"
        outputs = layers.Dense(1 if num_classes == 1 else num_classes, activation=activation)(x)
        model = models.Model(inputs, outputs, name="simple_cnn")
        if compile_model:
            loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
            if metrics is None:
                metrics = ["accuracy"]
            model.compile(optimizer=optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
        return model

    base_model.trainable = bool(train_base)

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Dense(1 if num_classes == 1 else num_classes, activation=activation)(x)
    model = models.Model(inputs, outputs, name=f"{base}_head")

    if compile_model:
        loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
        if metrics is None:
            metrics = ["accuracy"]
        model.compile(optimizer=optimizers.Adam(learning_rate), loss=loss, metrics=metrics)

    return model