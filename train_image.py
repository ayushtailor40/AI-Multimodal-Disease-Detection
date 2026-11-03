import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# 1. SETTINGS
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "dataset/chest_xray/train"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 2. LOAD DATASET
# =========================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("\nClasses:")
print(train_ds.class_names)


# =========================================================
# 3. CALCULATE CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

normal_count = len(
    os.listdir(os.path.join(TRAIN_DIR, "NORMAL"))
)

pneumonia_count = len(
    os.listdir(os.path.join(TRAIN_DIR, "PNEUMONIA"))
)

total = normal_count + pneumonia_count

class_weight = {
    0: total / (2 * normal_count),
    1: total / (2 * pneumonia_count)
}

print("NORMAL images:", normal_count)
print("PNEUMONIA images:", pneumonia_count)

print("Class weights:", class_weight)


# =========================================================
# 4. DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")


# =========================================================
# 5. EFFICIENTNETB0
# =========================================================

print("\nBuilding EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# =========================================================
# 6. BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)


# =========================================================
# 7. COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# =========================================================
# 8. MODEL SUMMARY
# =========================================================

model.summary()


# =========================================================
# 9. CALLBACKS
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/best_chest_xray_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# =========================================================
# 10. CREATE VALIDATION SPLIT
# =========================================================

print("\nCreating validation dataset...")

full_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="both",
    seed=42,
    shuffle=True
)

train_data = full_train_ds[0]
val_data = full_train_ds[1]


# =========================================================
# 11. TRAIN MODEL
# =========================================================

print("\nStarting training...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# =========================================================
# 12. SAVE FINAL MODEL
# =========================================================

model.save(
    "models/chest_xray_model.keras"
)

print(
    "\nModel saved successfully!"
)


# =========================================================
# 13. TRAINING GRAPHS
# =========================================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()


# Loss
plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()


plt.tight_layout()

plt.savefig(
    "training_results.png"
)

plt.show()

print(
    "\nTraining graphs saved as training_results.png"
)