
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

MODEL_PATH = "models/best_covid_model.keras"
TEST_DIR = "dataset/covid_processed/test"

# Load the best model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test images
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
print("\nClasses:", class_names)

# Actual labels
y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
]).astype(int)

# Predict probabilities and classes
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Test accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n========== COVID MODEL TEST RESULTS ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Per-class metrics
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("Confusion Matrix:")
print(cm)

# Save confusion matrix
os.makedirs("results", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(values_format="d", xticks_rotation=30)
plt.title("COVID Model - Test Confusion Matrix")
plt.tight_layout()
plt.savefig("results/covid_confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved: results/covid_confusion_matrix.png")
