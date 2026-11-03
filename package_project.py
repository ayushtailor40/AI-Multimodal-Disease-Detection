"""
Automated packaging script to create the deliverable ZIP archive.
Excludes venv, pycache, temporary files, and huge raw training sets (>4GB)
while including all code, templates, static files, models, results, tests,
and test samples.
"""

import os
import zipfile
from pathlib import Path

ROOT = Path("E:/SEMESTER 3/minor project/AI_Multimodal_Disease_Detection")
OUTPUT_ZIP = Path("E:/SEMESTER 3/minor project/AI_Multimodal_Disease_Detection_Deliverable.zip")

print("Building deliverable ZIP archive at:", OUTPUT_ZIP)

# Directories to include fully
FULL_INCLUDE_DIRS = [
    "templates",
    "static",
    "models",
    "results",
    "tests",
    "backup_original",
    "dataset/clinical",
    "dataset/symptoms",
    "database",
]

# Root files to include
ROOT_FILES = [
    "app.py",
    "database.py",
    "report_generator.py",
    "run.bat",
    "requirements.txt",
    "README.md",
    "clinical_model.py",
    "multimodal_fusion.py",
    "gradcam.py",
    "compare_lung_cancer_models.py",
    "evaluate_model.py",
    "evaluate_covid_model.py",
    "evaluate_tb_model.py",
    "evaluate_lung_cancer.py",
    "train_image.py",
    "train_covid.py",
    "train_tb.py",
    "train_lung_cancer.py",
    "train_lung_cancer_v2.py",
    "train_nlp.py",
    "test_nlp.py",
]

# Sample test images to include for out-of-the-box testing
SAMPLE_TEST_FILES = [
    "dataset/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg",
    "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
    "dataset/covid_processed/test/COVID/COVID-1002.png",
    "dataset/covid_processed/test/Normal/Normal-1001.png",
    "dataset/tb_processed/test/TUBERCULOSIS/Tuberculosis-1.png",
    "dataset/tb_processed/test/NORMAL/Normal-1.png",
    "dataset/lung_cancer_processed/test/Malignant/Malignant_Malignant case (1).jpg",
    "dataset/lung_cancer_processed/test/Benign/Benign_Benign case (1).jpg",
    "dataset/lung_cancer_processed/test/Normal/Normal_Normal case (1).jpg",
]

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zip_file:
    # Add root files
    for filename in ROOT_FILES:
        filepath = ROOT / filename
        if filepath.exists():
            zip_file.write(filepath, arcname=f"AI_Multimodal_Disease_Detection/{filename}")
            print(f"Added file: {filename}")

    # Add directories
    for dir_rel in FULL_INCLUDE_DIRS:
        dir_path = ROOT / dir_rel
        if dir_path.exists():
            for root_folder, _, files in os.walk(dir_path):
                if "__pycache__" in root_folder:
                    continue
                for f in files:
                    full_p = Path(root_folder) / f
                    arc_name = "AI_Multimodal_Disease_Detection" / full_p.relative_to(ROOT)
                    zip_file.write(full_p, arcname=str(arc_name))
            print(f"Added directory: {dir_rel}")

    # Add sample test files
    for sample_rel in SAMPLE_TEST_FILES:
        sample_path = ROOT / sample_rel
        if sample_path.exists():
            zip_file.write(sample_path, arcname=f"AI_Multimodal_Disease_Detection/{sample_rel}")
            print(f"Added test sample: {sample_rel}")

print("\nZIP Archive created successfully!")
print("Size in MB:", OUTPUT_ZIP.stat().st_size / (1024 * 1024))
