
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "dataset" / "lung_cancer_processed" / "test"
TRAIN_DIR = ROOT / "dataset" / "lung_cancer_processed" / "train"
MODEL_PATH = ROOT / "models" / "best_lung_cancer_model.keras"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Keep class order consistent with training
class_names = sorted(
    folder.name for folder in TRAIN_DIR.iterdir() if folder.is_dir()
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=16,
    label_mode="int",
    shuffle=False,
    class_names=class_names,
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print("\n========== LUNG CANCER TEST RESULTS ==========")
print(f"Test images: {len(y_true)}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
))

# Save numerical results
with open(RESULTS_DIR / "lung_cancer_classification_report.json",
          "w", encoding="utf-8") as f:
    json.dump({
        "test_accuracy": float(accuracy),
        "test_images": int(len(y_true)),
        "class_names": class_names,
        "classification_report": report,
        "note": (
            "Research-only evaluation. Image-level split; "
            "patient-level separation is not verified."
        ),
    }, f, indent=2)

# Save confusion matrix
cm = confusion_matrix(
    y_true, y_pred, labels=list(range(len(class_names)))
)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
display.plot(cmap="Blues", values_format="d")
plt.title("Lung Cancer Model - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "lung_cancer_confusion_matrix.png",
    dpi=200,
)
plt.show()

print("\nSaved report:",
      RESULTS_DIR / "lung_cancer_classification_report.json")
print("Saved confusion matrix:",
      RESULTS_DIR / "lung_cancer_confusion_matrix.png")
