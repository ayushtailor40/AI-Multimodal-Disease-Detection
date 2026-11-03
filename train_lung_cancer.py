
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "lung_cancer_processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "best_lung_cancer_model.keras"
FINAL_MODEL = MODEL_DIR / "lung_cancer_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

# Load images from separate folders
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names
print("\nClass order:", class_names)

if set(class_names) != {"Benign", "Malignant", "Normal"}:
    raise ValueError(f"Unexpected classes: {class_names}")

# Calculate class weights from training labels
train_labels = np.concatenate([y.numpy() for _, y in train_ds])
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(np.unique(train_labels), weights)
}
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Mild augmentation for CT images
augmentation = models.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.08),
], name="ct_augmentation")

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
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ModelCheckpoint(
        str(BEST_MODEL),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\nStarting Lung Cancer model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

model.save(FINAL_MODEL)

print("\nTraining complete.")
print("Best checkpoint:", BEST_MODEL)
print("Final model:", FINAL_MODEL)
print("Classes:", class_names)
