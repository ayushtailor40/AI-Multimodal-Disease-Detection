
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "lung_cancer"

print("\n========== LUNG CANCER DATASET CHECK ==========")
print("Dataset path:", DATASET)
print("Dataset exists:", DATASET.exists())

if not DATASET.exists():
    print("\nDataset folder nahi mila. Path check karo.")
else:
    for folder in DATASET.rglob("*"):
        if folder.is_dir() and folder.name.lower() in {
            "bengin cases", "benign cases",
            "malignant cases", "normal cases"
        }:
            images = [
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
                }
            ]
            print(f"\nClass folder: {folder}")
            print("Image count:", len(images))
            print("Sample files:", [p.name for p in images[:3]])

print("\n========== CHECK COMPLETE ==========")
