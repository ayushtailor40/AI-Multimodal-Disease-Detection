
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "lung_cancer_processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BEST_PATH = MODEL_DIR / "best_lung_cancer_model_v2.keras"
FINAL_PATH = MODEL_DIR / "lung_cancer_model_v2.keras"
CLASS_PATH = MODEL_DIR / "lung_cancer_classes_v2.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Check dataset paths
for split in ["train", "val"]:
    if not (DATASET / split).is_dir():
        raise FileNotFoundError(f"Missing folder: {DATASET / split}")

# Load training and validation data only.
# Test data will not be used for training or tuning.
train_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

class_names = train_raw.class_names
expected = ["Benign", "Malignant", "Normal"]

if class_names != expected:
    raise ValueError(
        f"Unexpected class order: {class_names}. Expected: {expected}"
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
    class_names=class_names,
)

# Calculate class weights from training labels
train_labels = np.concatenate(
    [labels.numpy() for _, labels in train_raw]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels,
)

class_weights = {
    int(i): float(w) for i, w in enumerate(weights)
}

print("\nClasses:", class_names)
print("Training class counts:",
      {name: int(np.sum(train_labels == i))
       for i, name in enumerate(class_names)})
print("Class weights:", class_weights)

# Improve input pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_raw.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Mild augmentation; no horizontal/vertical flips
augmentation = models.Sequential([
    layers.RandomRotation(0.025),
    layers.RandomZoom(0.05),
], name="ct_augmentation_v2")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(
    len(class_names), activation="softmax"
)(x)

model = models.Model(inputs, outputs)

checkpoint = ModelCheckpoint(
    str(BEST_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

def make_callbacks(patience=4):
    return [
        checkpoint,
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

# Stage 1: train classifier head
print("\n===== STAGE 1: CLASSIFIER TRAINING =====")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

# Stage 2: fine-tune only upper feature layers
print("\n===== STAGE 2: FINE-TUNING =====")

# Load the best Stage 1 checkpoint before fine-tuning
model = tf.keras.models.load_model(BEST_PATH)
base_model = model.get_layer("efficientnetb0")
base_model.trainable = True

# Freeze most layers and keep BatchNormalization frozen
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

# Save the best checkpoint as the v2 model
best_model = tf.keras.models.load_model(BEST_PATH)
best_model.save(FINAL_PATH)

with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print("\n===== TRAINING COMPLETE =====")
print("Best v2 model:", BEST_PATH)
print("Final v2 model:", FINAL_PATH)
print("Class labels:", CLASS_PATH)
print("Old models were not overwritten.")
print(
    "Note: patient-level split is not verified; "
    "this model is for research only."
)
