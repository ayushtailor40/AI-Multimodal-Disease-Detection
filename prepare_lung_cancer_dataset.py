
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset" / "lung_cancer" / "The IQ-OTHNCCD lung cancer dataset"
DEST = ROOT / "dataset" / "lung_cancer_processed"

random.seed(42)

# Source folder names and standardized class names
CLASS_MAP = {
    "Bengin cases": "Benign",
    "Malignant cases": "Malignant",
    "Normal cases": "Normal",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

if not SOURCE.exists():
    raise FileNotFoundError(f"Source folder not found: {SOURCE}")

# Prevent accidental overwriting of a previous prepared dataset
if DEST.exists():
    raise FileExistsError(
        f"{DEST} already exists. Rename or inspect it before running again."
    )

# Validate all source classes before creating the destination
class_files = {}

for source_name, class_name in CLASS_MAP.items():
    folder = SOURCE / source_name

    if not folder.is_dir():
        raise FileNotFoundError(f"Class folder not found: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise ValueError(f"No images found in: {folder}")

    class_files[class_name] = files

# Stratified image-level split within each class
for class_name, files in class_files.items():
    random.shuffle(files)

    total = len(files)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }

    for split_name, selected_files in split_files.items():
        output_folder = DEST / split_name / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for source_file in selected_files:
            # Prefix filenames with class to avoid collisions
            target = output_folder / f"{class_name}_{source_file.name}"
            shutil.copy2(source_file, target)

        print(f"{split_name:5} | {class_name:9} | {len(selected_files):4} images")

print("\nDataset preparation complete.")
print("Original images were not moved or deleted.")
print(f"Prepared dataset: {DEST}")
print("IMPORTANT: This is an image-level split, not a patient-level split.")
