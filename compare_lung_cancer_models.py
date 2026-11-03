
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "lung_cancer_processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant", "Normal"]
MODEL_PATHS = {
    "V1": MODEL_DIR / "best_lung_cancer_model.keras",
    "V2": MODEL_DIR / "best_lung_cancer_model_v2.keras",
}

# Use exactly the same untouched test set for both models
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "test",
    image_size=(224, 224),
    batch_size=16,
    label_mode="int",
    shuffle=False,
    class_names=CLASS_NAMES,
)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
results = {}

for version, model_path in MODEL_PATHS.items():
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n========== {version} EVALUATION ==========")
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(report_text)

    results[version] = {
        "accuracy": float(accuracy),
        "classification_report": report_dict,
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    display.plot(values_format="d")
    plt.title(f"Lung Cancer {version} - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"lung_cancer_{version.lower()}_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

# Save both reports for later reference
output_path = RESULTS_DIR / "lung_cancer_v1_v2_comparison.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n========== FINAL COMPARISON ==========")
for version, result in results.items():
    print(f"{version} accuracy: {result['accuracy'] * 100:.2f}%")

print("\nComparison saved to:", output_path)
print("Confusion matrices saved in:", RESULTS_DIR)
print(
    "\nResearch-only evaluation. Patient-level separation "
    "has not been verified."
)
