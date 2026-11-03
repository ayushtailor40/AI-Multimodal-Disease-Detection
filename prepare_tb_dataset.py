import os
import shutil
import random

SOURCE = "dataset/tb/TB_Chest_Radiography_Database"
DEST = "dataset/tb_processed"

random.seed(42)

classes = {
    "Normal": "NORMAL",
    "Tuberculosis": "TUBERCULOSIS"
}

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

for source_class, target_class in classes.items():

    source_folder = os.path.join(SOURCE, source_class)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split, split_images in splits.items():

        destination_folder = os.path.join(
            DEST,
            split,
            target_class
        )

        os.makedirs(destination_folder, exist_ok=True)

        for image in split_images:

            source_path = os.path.join(
                source_folder,
                image
            )

            destination_path = os.path.join(
                destination_folder,
                image
            )

            shutil.copy2(source_path, destination_path)

    print(f"{target_class}")
    print(f"  Total : {total}")
    print(f"  Train : {len(train_images)}")
    print(f"  Val   : {len(val_images)}")
    print(f"  Test  : {len(test_images)}")
    print()

print("TB dataset preparation completed!")