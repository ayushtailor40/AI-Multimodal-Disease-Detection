# AI-Based Multimodal Approach for Early Disease Detection

**Master of Computer Applications (MCA) Final-Year Capstone Project**  
*A Clinical AI Decision-Support Dashboard Integrating Medical Imaging, Symptom NLP, and Physiological Vital Signs with Explainable AI (Grad-CAM).*

---

## 1. Project Overview

Early detection of respiratory and pulmonary diseases is essential for reducing clinical mortality. Conventional deep learning approaches typically examine medical imagery in isolation from patient symptoms and vital signs.

This project delivers a **multimodal diagnostic web dashboard** built with Python and Flask. It integrates:
1. **Tri-Modal Pneumonia Detection:** Fusing Chest X-Ray radiographs (40%), free-text clinical symptom narratives (30%), and physiological vital signs (30%) via late-fusion consensus.
2. **Explainable AI (Grad-CAM):** Visualizing convolutional attention heatmaps over chest radiographs to highlight regions of interest (e.g. consolidations, infiltrates).
3. **Independent Disease Screening Engines:**
   - **COVID-19 Radiography:** 4-class classifier (`COVID`, `Lung_Opacity`, `Normal`, `Viral Pneumonia`).
   - **Tuberculosis (TB) Radiography:** High-sensitivity binary screening classifier (`NORMAL` vs `TUBERCULOSIS` with 99.21% test accuracy).
   - **Lung Cancer CT Scan Analysis:** 3-class classifier (`Benign`, `Malignant`, `Normal`) specifically for cross-sectional axial CT slices.
4. **Interactive Model Evaluation Dashboard:** Transparent, authentic metrics reporting test accuracy, precision, recall, F1-scores, dataset split statistics, and real confusion matrices.
5. **Patient Diagnostic History & SQLite Database:** Persistent clinical log tracking patient ID, vitals, symptoms, prediction, and confidence score with instant search, filter, and CSV audit export.
6. **Clinical Diagnostic PDF Report Generator:** One-click download of hospital-grade laboratory reports featuring patient demographics, modality breakdown tables, uploaded scans, and Grad-CAM attention heatmaps.

---

## 2. Technology Stack

- **Backend:** Python 3.13, Flask 3.1, Werkzeug
- **Deep Learning:** TensorFlow 2.x, Keras 3.x (EfficientNet-B0 transfer learning backbones)
- **Machine Learning & NLP:** Scikit-learn (LogisticRegression, TfidfVectorizer, StandardScaler)
- **Computer Vision & Image Processing:** OpenCV (`cv2`), Pillow (`PIL`), Matplotlib
- **Frontend:** HTML5, CSS3 Custom Medical Design System, Vanilla JavaScript (Drag-and-Drop file uploads, instant thumbnail preview, demo quick-fill profiles, responsive mobile drawer)

---

## 3. Directory Structure

```text
AI_Multimodal_Disease_Detection/
├── app.py                         # Master Flask application controller
├── requirements.txt               # Project Python dependencies
├── README.md                      # Comprehensive project documentation
├── backup_original/               # Backup archive of initial project files
├── templates/                     # Jinja2 presentation templates
│   ├── base.html                  # Global layout, sidebar navigation & disclaimer
│   ├── index.html                 # Overview landing dashboard & pipeline diagram
│   ├── pneumonia.html             # Tri-modal Pneumonia detection + Grad-CAM
│   ├── covid.html                 # COVID-19 4-class chest X-ray classifier
│   ├── tb.html                    # Tuberculosis binary chest X-ray classifier
│   ├── lung_cancer.html           # Lung Cancer 3-class CT scan classifier
│   ├── evaluation.html            # Real benchmarking metrics & confusion matrices
│   └── about.html                 # Academic project documentation & formulas
├── static/
│   ├── css/
│   │   └── dashboard.css          # Medical UI design system stylesheet
│   ├── js/
│   │   └── main.js                # Drag-and-drop uploads, preview & tabs
│   ├── images/                    # Verified confusion matrices & plots
│   │   ├── pneumonia_confusion_matrix.png
│   │   ├── covid_confusion_matrix.png
│   │   ├── tb_confusion_matrix.png
│   │   ├── lung_cancer_v1_confusion_matrix.png
│   │   └── lung_cancer_v2_confusion_matrix.png
│   └── uploads/                   # Temporary directory for uploaded scans & heatmaps
├── models/                        # Pre-trained deep learning & ML model artifacts
│   ├── best_chest_xray_model.keras
│   ├── best_covid_model.keras
│   ├── best_tb_model.keras
│   ├── best_lung_cancer_model.keras
│   ├── best_lung_cancer_model_v2.keras
│   ├── clinical_model.pkl
│   ├── clinical_scaler.pkl
│   ├── nlp_model.pkl
│   └── tfidf_vectorizer.pkl
├── dataset/                       # Clinical, radiography & CT datasets
│   ├── chest_xray/                # Kaggle Chest X-Ray dataset (train/val/test)
│   ├── covid_processed/           # COVID-19 Radiography Database
│   ├── tb_processed/              # TB Chest Radiography Database
│   ├── lung_cancer_processed/     # IQ-OTHNCCD Lung Cancer CT dataset
│   ├── clinical/                  # Clinical tabular features & CSV
│   └── symptoms/                  # Pulmonary symptom narratives & CSV
├── results/                       # Authentic evaluation JSON reports & figures
│   ├── pneumonia_evaluation_report.json
│   ├── covid_evaluation_report.json
│   ├── tb_evaluation_report.json
│   └── lung_cancer_v1_v2_comparison.json
└── tests/
    ├── test_application.py        # Automated test suite (15 unit & integration tests)
    └── TEST_REPORT.md             # Detailed test verification report
```

---

## 4. Setup & Installation Instructions (Windows)

### Step 1: Open Terminal in the Project Directory
In **VS Code**, open the project folder `AI_Multimodal_Disease_Detection` and launch PowerShell:
```powershell
Set-Location "E:\SEMESTER 3\minor project\AI_Multimodal_Disease_Detection"
```

### Step 2: Activate the Virtual Environment
An existing pre-configured virtual environment is located at `venv`:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

If creating a new virtual environment from scratch:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 5. Running the Application

Start the Flask development server:
```powershell
python app.py
```

Once started, open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 6. Testing Each Disease Module

You can test each module through the web UI using the sample test images in `dataset/`:

### A. Pneumonia Multimodal Detection (`/pneumonia`)
1. Click **Pneumonia Detection** in the sidebar.
2. Click the **"⚡ Sample Pneumonia Case"** or **"⚡ Sample Normal Case"** quick-fill button at the top right to instantly populate symptoms and vitals.
3. Upload an X-ray from:  
   `dataset/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg`
4. Click **Run Tri-Modal Multimodal Fusion Analysis**.
5. Observe the 3-modality breakdown, fusion decision, and Grad-CAM attention heatmap overlay.

### B. COVID-19 Detection (`/covid`)
1. Navigate to **COVID-19 Detection**.
2. Upload a sample radiograph from:  
   `dataset/covid_processed/test/COVID/COVID-1002.png`
3. View the 4-class confidence distribution.

### C. Tuberculosis Detection (`/tb`)
1. Navigate to **Tuberculosis (TB)**.
2. Upload a sample radiograph from:  
   `dataset/tb_processed/test/TUBERCULOSIS/Tuberculosis-1.png`
3. View the binary screening classification result.

### D. Lung Cancer CT Scan Detection (`/lung-cancer`)
1. Navigate to **Lung Cancer CT**.
2. **Note:** This module strictly requires axial CT slices, NOT chest X-rays.
3. Upload a sample CT slice from:  
   `dataset/lung_cancer_processed/test/Malignant/Malignant_Malignant case (1).jpg`
4. View the 3-class classification (`Benign`, `Malignant`, `Normal`).

---

## 7. Running the Automated Test Suite

To execute the automated regression test suite:
```powershell
python tests/test_application.py
```
This runs 15 tests covering all routes, input validations, security boundaries, and end-to-end neural network inferences.

---

## 8. Model Limitations & Medical Research Disclaimer

- **Educational Prototype:** This system is an academic MCA final-year project intended strictly for demonstration and research.
- **Uncalibrated Scores:** Model percentage outputs represent uncalibrated softmax scores and do not represent absolute clinical diagnostic probabilities.
- **CT Dataset Partitioning:** The IQ-OTHNCCD lung cancer dataset uses an image-level split; patient-level separation is not verified in the public benchmark.
- **No Clinical Decisions:** Never make medical decisions based on this prototype. Always consult a licensed medical professional for clinical imaging interpretation.
