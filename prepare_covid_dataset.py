
import os
import shutil
import random

SOURCE = "dataset/covid"
DEST = "dataset/covid_processed"

random.seed(42)

classes = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

extensions = (".png", ".jpg", ".jpeg")

for class_name in classes:
    source_folder = os.path.join(SOURCE, class_name, "images")

    if not os.path.isdir(source_folder):
        raise FileNotFoundError(
            f"Image folder not found: {source_folder}"
        )

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(extensions)
    ]

    if not images:
        raise ValueError(f"No images found for {class_name}")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, split_images in splits.items():
        destination_folder = os.path.join(
            DEST, split, class_name
        )
        os.makedirs(destination_folder, exist_ok=True)

        for filename in split_images:
            shutil.copy2(
                os.path.join(source_folder, filename),
                os.path.join(destination_folder, filename)
            )

    print(f"\n{class_name}: Total {total}")
    for split, split_images in splits.items():
        print(f"  {split}: {len(split_images)}")

print("\nCOVID dataset preparation completed!")
print("Original dataset remains unchanged.")
