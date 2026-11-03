# Test & Validation Report: AI Multimodal Disease Detection

**Project Title:** An AI-Based Multimodal Approach for Early Disease Detection  
**Degree / Academic Level:** Master of Computer Applications (MCA) Final-Year Project  
**Date of Verification:** October 28, 2026  
**Environment:** Python 3.13.6 on Windows (PowerShell / VS Code Workspace)  
**Test Framework:** Python `unittest` Automated Test Suite (`tests/test_application.py`)

---

## 1. Executive Summary

A comprehensive automated test suite consisting of **19 unit and integration test cases** was executed against the Flask backend, template rendering engine, validation pipelines, deep learning inference modules, SQLite patient database, and ReportLab PDF generator.

**Overall Result:** `ALL 19 TESTS PASSED` (Execution Time: 54.12s)

```text
Ran 19 tests in 54.117s
OK
```

---

## 2. Test Execution Details

| Test Case | Description | Tested Component | Status |
|:---|:---|:---|:---:|
| `test_01_home_dashboard_get` | HTTP GET `/` Overview Dashboard | Landing page & navigation sidebar | **PASS** |
| `test_02_pneumonia_get` | HTTP GET `/pneumonia` Multimodal Page | Tri-modal form UI & demo buttons | **PASS** |
| `test_03_covid_get` | HTTP GET `/covid` COVID-19 Radiography | 4-class classifier interface | **PASS** |
| `test_04_tb_get` | HTTP GET `/tb` Tuberculosis Screening | Binary radiographic screener UI | **PASS** |
| `test_05_lung_cancer_get` | HTTP GET `/lung-cancer` CT Nodule Module | CT-specific requirements interface | **PASS** |
| `test_06_evaluation_get` | HTTP GET `/evaluation` Benchmarking | Metrics tabs & confusion matrices | **PASS** |
| `test_07_about_get` | HTTP GET `/about` Documentation | Architecture & mathematical formulations | **PASS** |
| `test_08_allowed_image_extensions` | Allowed extension validation (`.jpg`, `.png`, etc.) | Input security & MIME validation | **PASS** |
| `test_09_missing_file_upload` | Missing image in POST payload | Graceful error handling & feedback | **PASS** |
| `test_10_unsupported_file_format` | Disallowed file extension (`.txt`, `.pdf`) | Extension security rejection | **PASS** |
| `test_11_corrupted_image_content` | Corrupted binary file masked as `.jpg` | Pillow `Image.verify()` validation | **PASS** |
| `test_12_covid_prediction` | End-to-end inference on COVID test radiograph | `best_covid_model.keras` | **PASS** |
| `test_13_tb_prediction` | End-to-end inference on TB test radiograph | `best_tb_model.keras` (Single-Sigmoid) | **PASS** |
| `test_14_lung_cancer_prediction` | End-to-end inference on CT scan slice | `best_lung_cancer_model.keras` | **PASS** |
| `test_15_pneumonia_multimodal_prediction`| Tri-modal fusion (CXR + NLP + Vitals + Grad-CAM)| Late fusion consensus engine & XAI | **PASS** |
| `test_16_patient_history_get` | HTTP GET `/history` Patient Records | SQLite patient records & KPI metrics | **PASS** |
| `test_17_export_csv` | HTTP GET `/export-csv` Audit Stream | CSV header & data serialization | **PASS** |
| `test_18_pdf_report_download` | HTTP GET `/download-report/<id>` PDF Report | ReportLab PDF compilation & stream | **PASS** |
| `test_19_delete_record` | HTTP POST `/delete-record/<id>` Row Deletion | SQLite row deletion & redirection | **PASS** |

---

## 3. Database & PDF Reporting Architecture

### A. Patient History Database (SQLite)
- **Module:** `database.py`
- **Database File:** `database/patient_records.db`
- **Stored Attributes:** Patient ID, Name, Age, Gender, Disease Module, Imaging Modality, AI Finding, Confidence Score, Scores JSON, Vitals JSON, Symptoms Narrative, Uploaded Scan Path, Grad-CAM Heatmap Path, and Timestamp.
- **Audit Capabilities:** Instant keyword search, disease-module filtering, and CSV export streaming (`/export-csv`).

### B. Clinical Diagnostic PDF Report Generator
- **Module:** `report_generator.py`
- **Output:** Hospital-grade diagnostic laboratory reports compiled using ReportLab.
- **Embedded Elements:** Official laboratory banner, patient demographic table, primary diagnostic finding badge, vital signs grid, differential diagnosis breakdown, high-resolution original radiograph / CT image, Grad-CAM attention heatmap overlay, research disclaimer, and consulting physician signature block.
