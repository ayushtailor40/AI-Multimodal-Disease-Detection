import os
import random
from PIL import Image
import matplotlib.pyplot as plt

base_path = "dataset/chest_xray/train"

classes = ["NORMAL", "PNEUMONIA"]

print("Image Information")
print("=" * 40)

for class_name in classes:

    folder_path = os.path.join(base_path, class_name)

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n{class_name}")
    print("Number of images:", len(images))

    # Check first image size
    first_image = os.path.join(folder_path, images[0])

    img = Image.open(first_image)

    print("Example image size:", img.size)
    print("Image mode:", img.mode)


# -----------------------------
# Display sample images
# -----------------------------

plt.figure(figsize=(12, 8))

for i, class_name in enumerate(classes):

    folder_path = os.path.join(base_path, class_name)

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_images = random.sample(images, 4)

    for j, image_name in enumerate(selected_images):

        image_path = os.path.join(folder_path, image_name)

        img = Image.open(image_path)

        position = i * 4 + j + 1

        plt.subplot(2, 4, position)

        plt.imshow(img, cmap="gray")

        plt.title(class_name)

        plt.axis("off")


plt.tight_layout()

plt.savefig("dataset_samples.png")

plt.show()

print("\nSample images saved as dataset_samples.png")