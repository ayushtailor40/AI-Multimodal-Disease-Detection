
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

# Paths
TRAIN_DIR = "dataset/covid_processed/train"
VAL_DIR = "dataset/covid_processed/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
print("\nClass order:", class_names)

# Optimize data loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Class weights from actual training folders
counts = np.array([
    len([
        f for f in os.listdir(os.path.join(TRAIN_DIR, c))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    for c in class_names
])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.repeat(np.arange(len(class_names)), counts)
)

class_weights = {
    i: float(weight) for i, weight in enumerate(weights)
}

print("Training counts:", dict(zip(class_names, counts)))
print("Class weights:", class_weights)

# Data augmentation
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10)
], name="augmentation")

# Pretrained feature extractor
base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    len(class_names), activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save models separately
os.makedirs("models", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/best_covid_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save("models/covid_model.keras")

print("\nCOVID MODEL TRAINING COMPLETED")
print("Best model: models/best_covid_model.keras")
print("Final model: models/covid_model.keras")
print("Classes:", class_names)
