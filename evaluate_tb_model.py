
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# 1. Load the best saved model
MODEL_PATH = "models/best_tb_model.keras"
TEST_DIR = "dataset/tb_processed/test"

model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load the test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names
print("\nClasses:", class_names)

# 3. Actual labels
y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

# 4. Model predictions
predictions = model.predict(test_ds, verbose=1).reshape(-1)
y_pred = (predictions >= 0.5).astype(int)

# 5. Evaluation
accuracy = accuracy_score(y_true, y_pred)

print("\n========== TB MODEL TEST RESULTS ==========")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0,
    digits=4
))

# 6. Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

os.makedirs("results", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(values_format="d")
plt.title("TB Model - Test Confusion Matrix")
plt.tight_layout()
plt.savefig("results/tb_confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved: results/tb_confusion_matrix.png")
