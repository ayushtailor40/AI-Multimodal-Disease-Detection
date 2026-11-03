import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = 224
BATCH_SIZE = 32

TEST_DIR = "dataset/chest_xray/test"

MODEL_PATH = "models/best_chest_xray_model.keras"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD TEST DATA
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:", class_names)


# ==========================================
# GET TRUE LABELS
# ==========================================

y_true = []

for images, labels in test_ds:
    y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true).astype(int)


# ==========================================
# PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_probability = model.predict(test_ds)

# Convert probability to class
y_pred = (y_probability >= 0.5).astype(int).flatten()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_true, y_pred)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title("Chest X-Ray Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")