import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==============================
# 1. Paths
# ==============================

TRAIN_DIR = "dataset/tb_processed/train"
VAL_DIR = "dataset/tb_processed/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# ==============================
# 2. Load datasets
# ==============================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nClasses:", train_ds.class_names)

# ==============================
# 3. Performance optimization
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==============================
# 4. Class weights
# ==============================

# NORMAL = 0
# TUBERCULOSIS = 1

normal_count = 2450
tb_count = 489

classes = np.array([0, 1])
class_counts = np.array([normal_count, tb_count])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.repeat(classes, class_counts)
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("\nClass Weights:")
print("NORMAL:", class_weights[0])
print("TUBERCULOSIS:", class_weights[1])

# ==============================
# 5. Data augmentation
# ==============================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")

# ==============================
# 6. EfficientNetB0
# ==============================

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==============================
# 7. Build model
# ==============================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)

# ==============================
# 8. Compile
# ==============================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ==============================
# 9. Callbacks
# ==============================

os.makedirs("models", exist_ok=True)

checkpoint = keras.callbacks.ModelCheckpoint(
    "models/best_tb_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# ==============================
# 10. Train
# ==============================

print("\nStarting TB model training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# ==============================
# 11. Save final model
# ==============================

model.save("models/tb_model.keras")

print("\n================================")
print("TB MODEL TRAINING COMPLETED")
print("================================")

print("\nBest model:")
print("models/best_tb_model.keras")

print("\nFinal model:")
print("models/tb_model.keras")