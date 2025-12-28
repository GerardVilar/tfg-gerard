# shared/model_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

# Charge CIFAR
(_x_train, _y_train), (x_test_raw, y_test_raw) = cifar10.load_data()

# Preprocessing
x_test = x_test_raw.astype("float32") / 255.0
y_test = tf.keras.utils.to_categorical(y_test_raw, NUM_CLASSES)


def build_model():
    model = models.Sequential(
        [
            layers.Input(shape=IMG_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# Weights convert

def model_to_vector(model) -> list[float]:
    weights = model.get_weights()
    flat = np.concatenate([w.flatten() for w in weights])
    return flat.astype("float32").tolist()


def vector_to_model(model, vector: list[float]):
    vec = np.array(vector, dtype="float32")
    shapes = [w.shape for w in model.get_weights()]
    sizes = [int(np.prod(s)) for s in shapes]
    splits = np.split(vec, np.cumsum(sizes)[:-1])
    new_weights = [arr.reshape(shape) for arr, shape in zip(splits, shapes)]
    model.set_weights(new_weights)


def init_model(dim_ignored: int = 0):
    model = build_model()
    return model_to_vector(model)


def average_models(list_of_vectors: list[list[float]]) -> list[float]:
    arr = np.stack([np.array(v, dtype="float32") for v in list_of_vectors])
    return np.mean(arr, axis=0).tolist()

_EVAL_MODEL = None

def _get_eval_model():
    global _EVAL_MODEL
    if _EVAL_MODEL is None:
        _EVAL_MODEL = build_model()
    return _EVAL_MODEL

# Global evaluation of the server
def evaluate_vector_on_test(vector: list[float], max_samples: int = 1000):
    model = _get_eval_model()
    vector_to_model(model, vector)

    if max_samples is not None:
        x = x_test[:max_samples]
        y = y_test[:max_samples]
    else:
        x = x_test
        y = y_test

    loss, acc = model.evaluate(x, y, verbose=0)
    return loss, acc