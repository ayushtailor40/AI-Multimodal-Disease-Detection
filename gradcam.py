import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "models/best_chest_xray_model.keras"

IMAGE_DIR = "dataset/chest_xray/test/PNEUMONIA"

IMG_SIZE = 224


# =====================================================
# FIND IMAGE
# =====================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise FileNotFoundError("No X-ray image found!")

image_file = image_files[0]

image_path = os.path.join(
    IMAGE_DIR,
    image_file
)

print("Using image:")
print(image_path)


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# =====================================================
# LOAD IMAGE
# =====================================================

original_image = Image.open(
    image_path
).convert("RGB")

resized_image = original_image.resize(
    (IMG_SIZE, IMG_SIZE)
)

image_array = np.array(
    resized_image
).astype("float32")

image_array = np.expand_dims(
    image_array,
    axis=0
)


# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]

if prediction >= 0.5:

    predicted_class = "PNEUMONIA"

    confidence = prediction

else:

    predicted_class = "NORMAL"

    confidence = 1 - prediction


print("\nPrediction:")
print(predicted_class)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)


# =====================================================
# GET EFFICIENTNET MODEL
# =====================================================

base_model = model.get_layer(
    "efficientnetb0"
)

print(
    "\nEfficientNet model found:"
)

print(
    base_model.name
)


# =====================================================
# FIND LAST CONVOLUTIONAL LAYER
# =====================================================

last_conv_layer = None

for layer in reversed(base_model.layers):

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D
        )
    ):

        last_conv_layer = layer

        break


if last_conv_layer is None:

    raise ValueError(
        "Could not find convolutional layer."
    )


print(
    "\nLast convolutional layer:"
)

print(
    last_conv_layer.name
)


# =====================================================
# BUILD CONNECTED GRAD-CAM GRAPH
# =====================================================

input_tensor = model.input

# Data augmentation layer
augmentation_layer = model.get_layer(
    "data_augmentation"
)

x = augmentation_layer(
    input_tensor,
    training=False
)

# Run through EfficientNet
conv_outputs = base_model(
    x,
    training=False
)


# =====================================================
# CONNECT CLASSIFICATION HEAD
# =====================================================

# Get layers after EfficientNet

base_index = model.layers.index(
    base_model
)

post_layers = model.layers[
    base_index + 1:
]

x = conv_outputs

for layer in post_layers:

    x = layer(x)


final_output = x


# =====================================================
# CREATE GRADIENT MODEL
# =====================================================

grad_model = tf.keras.models.Model(
    inputs=input_tensor,
    outputs=[
        conv_outputs,
        final_output
    ]
)


# =====================================================
# CALCULATE GRADIENT
# =====================================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        image_array
    )

    pneumonia_probability = predictions[:, 0]

    if predicted_class == "PNEUMONIA":

        class_score = pneumonia_probability

    else:

        class_score = 1 - pneumonia_probability


grads = tape.gradient(
    class_score,
    conv_outputs
)


# =====================================================
# GLOBAL AVERAGE POOLING
# =====================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


# =====================================================
# CREATE HEATMAP
# =====================================================

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(
    conv_outputs *
    pooled_grads[
        tf.newaxis,
        tf.newaxis,
        :
    ],
    axis=-1
)


# Remove negative values

heatmap = tf.maximum(
    heatmap,
    0
)


# Normalize

max_value = tf.reduce_max(
    heatmap
)

if max_value > 0:

    heatmap = heatmap / max_value


heatmap = heatmap.numpy()


# =====================================================
# RESIZE HEATMAP
# =====================================================

heatmap_image = Image.fromarray(
    np.uint8(
        heatmap * 255
    )
)

heatmap_image = heatmap_image.resize(
    (
        original_image.width,
        original_image.height
    )
)

heatmap_array = np.array(
    heatmap_image
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

plt.figure(
    figsize=(15, 5)
)


# -----------------------------------------------------
# ORIGINAL IMAGE
# -----------------------------------------------------

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    original_image,
    cmap="gray"
)

plt.title(
    f"Original X-Ray\n{predicted_class}"
)

plt.axis("off")


# -----------------------------------------------------
# HEATMAP
# -----------------------------------------------------

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap_array,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")


# -----------------------------------------------------
# OVERLAY
# -----------------------------------------------------

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    original_image,
    cmap="gray"
)

plt.imshow(
    heatmap_array,
    cmap="jet",
    alpha=0.4
)

plt.title(
    "Grad-CAM Overlay"
)

plt.axis("off")


plt.tight_layout()


# =====================================================
# SAVE
# =====================================================

output_path = "gradcam_result.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nGrad-CAM completed successfully!"
)

print(
    "Saved as:",
    output_path
)