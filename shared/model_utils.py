import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Image shape for CIFAR-10 samples
IMG_SHAPE = (32, 32, 3)

# Number of classes in CIFAR-10
NUM_CLASSES = 10

# Load CIFAR-10 dataset
(_x_train, _y_train), (x_test_raw, y_test_raw) = cifar10.load_data()

# Preprocess test images (normalization)
x_test = x_test_raw.astype("float32") / 255.0

# Convert test labels to categorical format
y_test = tf.keras.utils.to_categorical(y_test_raw, NUM_CLASSES)


# Build and compile the CNN model used by all agents and the server
def build_model():
    model = models.Sequential(
        [
            # Input layer for CIFAR-10 images
            layers.Input(shape=IMG_SHAPE),

            # First convolutional block
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Flatten feature maps into a vector
            layers.Flatten(),

            # Dropout for regularization
            layers.Dropout(0.5),

            # Output layer with softmax activation
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # Compile the model with standard classification settings
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# =========================
# Weight conversion helpers
# =========================

# Convert model weights into a single flat vector
def model_to_vector(model) -> list[float]:
    # Retrieve all model weights
    weights = model.get_weights()

    # Flatten and concatenate all weight tensors
    flat = np.concatenate([w.flatten() for w in weights])

    # Return weights as a Python list
    return flat.astype("float32").tolist()


# Load a flat vector of weights back into a model
def vector_to_model(model, vector: list[float]):
    # Convert the flat list into a NumPy array
    vec = np.array(vector, dtype="float32")

    # Retrieve original weight shapes from the model
    shapes = [w.shape for w in model.get_weights()]

    # Compute number of elements for each weight tensor
    sizes = [int(np.prod(s)) for s in shapes]

    # Split the flat vector according to weight sizes
    splits = np.split(vec, np.cumsum(sizes)[:-1])

    # Reshape each split to the original tensor shape
    new_weights = [
        arr.reshape(shape) for arr, shape in zip(splits, shapes)
    ]

    # Assign reconstructed weights to the model
    model.set_weights(new_weights)


# Initialize a fresh model and return its weights as a vector
def init_model(dim_ignored: int = 0):
    model = build_model()
    return model_to_vector(model)


# Compute the element-wise average of multiple model vectors
def average_models(list_of_vectors: list[list[float]]) -> list[float]:
    # Stack all vectors into a 2D array
    arr = np.stack([
        np.array(v, dtype="float32") for v in list_of_vectors
    ])

    # Compute mean across models
    return np.mean(arr, axis=0).tolist()


# Cached model instance used only for evaluation
_EVAL_MODEL = None


# Lazily initialize the evaluation model
def _get_eval_model():
    global _EVAL_MODEL
    if _EVAL_MODEL is None:
        _EVAL_MODEL = build_model()
    return _EVAL_MODEL


# Evaluate a model vector on the CIFAR-10 test set
def evaluate_vector_on_test(vector: list[float], max_samples: int = 1000):
    # Retrieve or initialize the evaluation model
    model = _get_eval_model()

    # Load the provided weights into the model
    vector_to_model(model, vector)

    # Optionally limit the number of test samples
    if max_samples is not None:
        x = x_test[:max_samples]
        y = y_test[:max_samples]
    else:
        x = x_test
        y = y_test

    # Evaluate the model on the selected test data
    loss, acc = model.evaluate(x, y, verbose=0)

    return loss, acc