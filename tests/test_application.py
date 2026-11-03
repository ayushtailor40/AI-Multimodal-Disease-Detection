"""
=============================================================================
COMPREHENSIVE TEST SUITE - AI MULTIMODAL DISEASE DETECTION
Automated Unit, Route, Template & End-to-End Prediction Verification
=============================================================================
"""

import io
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app, allowed_image_file, verify_image


class DiseaseDetectionTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.root = PROJECT_ROOT

        # Find sample images
        cls.sample_cxr = cls.root / "dataset" / "chest_xray" / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg"
        cls.sample_covid = cls.root / "dataset" / "covid_processed" / "test" / "COVID" / "COVID-1002.png"
        cls.sample_tb = cls.root / "dataset" / "tb_processed" / "test" / "TUBERCULOSIS" / "Tuberculosis-1.png"
        cls.sample_lung = cls.root / "dataset" / "lung_cancer_processed" / "test" / "Malignant" / "Malignant_Malignant case (1).jpg"

    # =========================================================================
    # 1. GET ROUTE & TEMPLATE RENDERING TESTS
    # =========================================================================

    def test_01_home_dashboard_get(self):
        """Test GET / (Overview Dashboard) renders properly."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Overview Dashboard", response.data)
        self.assertIn(b"Autonomous Disease Detection Modules", response.data)

    def test_02_pneumonia_get(self):
        """Test GET /pneumonia renders properly."""
        response = self.client.get("/pneumonia")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Pneumonia Multimodal Diagnostic Engine", response.data)
        self.assertIn(b"Modality 1:", response.data)

    def test_03_covid_get(self):
        """Test GET /covid renders properly."""
        response = self.client.get("/covid")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"COVID-19 Pulmonary Radiography Classifier", response.data)

    def test_04_tb_get(self):
        """Test GET /tb renders properly."""
        response = self.client.get("/tb")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tuberculosis (TB) Screening Classifier", response.data)

    def test_05_lung_cancer_get(self):
        """Test GET /lung-cancer renders properly."""
        response = self.client.get("/lung-cancer")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Lung Cancer CT Histopathology", response.data)
        self.assertIn(b"Computed Tomography (CT) Only", response.data)

    def test_06_evaluation_get(self):
        """Test GET /evaluation renders properly."""
        response = self.client.get("/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model Benchmarking & Evaluation Dashboard", response.data)
        self.assertIn(b"99.21%", response.data)

    def test_07_about_get(self):
        """Test GET /about renders properly."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About the Research Project", response.data)
        self.assertIn(b"Grad-CAM", response.data)

    # =========================================================================
    # 2. VALIDATION & SECURITY TESTS
    # =========================================================================

    def test_08_allowed_image_extensions(self):
        """Test file extension validation."""
        self.assertTrue(allowed_image_file("patient_scan.jpg"))
        self.assertTrue(allowed_image_file("chest_xray.png"))
        self.assertTrue(allowed_image_file("nodule.tiff"))
        self.assertFalse(allowed_image_file("exploit.exe"))
        self.assertFalse(allowed_image_file("script.py"))
        self.assertFalse(allowed_image_file("notes.pdf"))

    def test_09_missing_file_upload(self):
        """Test missing file upload returns validation error."""
        response = self.client.post("/covid", data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please select a medical image first", response.data)

    def test_10_unsupported_file_format(self):
        """Test uploading a disallowed file type returns error."""
        data = {
            "scan": (io.BytesIO(b"fake text content"), "report.txt")
        }
        response = self.client.post("/covid", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported image file type", response.data)

    def test_11_corrupted_image_content(self):
        """Test uploading corrupted bytes masked as .jpg is caught."""
        data = {
            "scan": (io.BytesIO(b"corrupted binary non-image data"), "corrupt.jpg")
        }
        response = self.client.post("/tb", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"could not be verified as a valid image", response.data)

    # =========================================================================
    # 3. END-TO-END PREDICTION INFERENCE TESTS (USING REAL TEST SAMPLES)
    # =========================================================================

    def test_12_covid_prediction(self):
        """Test end-to-end inference on COVID-19 module."""
        if not self.sample_covid.exists():
            self.skipTest(f"Sample COVID image not found: {self.sample_covid}")

        with open(self.sample_covid, "rb") as img:
            data = {"scan": (img, self.sample_covid.name)}
            response = self.client.post("/covid", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Radiographic Classification Result", response.data)
        self.assertIn(b"Class Confidence Distribution", response.data)

    def test_13_tb_prediction(self):
        """Test end-to-end inference on Tuberculosis module (single-sigmoid fix)."""
        if not self.sample_tb.exists():
            self.skipTest(f"Sample TB image not found: {self.sample_tb}")

        with open(self.sample_tb, "rb") as img:
            data = {"scan": (img, self.sample_tb.name)}
            response = self.client.post("/tb", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tuberculosis Screening Result", response.data)
        self.assertIn(b"TUBERCULOSIS", response.data)

    def test_14_lung_cancer_prediction(self):
        """Test end-to-end inference on Lung Cancer CT module."""
        if not self.sample_lung.exists():
            self.skipTest(f"Sample Lung CT image not found: {self.sample_lung}")

        with open(self.sample_lung, "rb") as img:
            data = {"scan": (img, self.sample_lung.name)}
            response = self.client.post("/lung-cancer", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CT Nodule Classification Result", response.data)
        self.assertIn(b"Malignant", response.data)

    def test_15_pneumonia_multimodal_prediction(self):
        """Test end-to-end tri-modal multimodal fusion pipeline with Grad-CAM and DB record creation."""
        if not self.sample_cxr.exists():
            self.skipTest(f"Sample CXR image not found: {self.sample_cxr}")

        with open(self.sample_cxr, "rb") as img:
            data = {
                "patient_name": "Test Patient Alpha",
                "patient_gender": "Male",
                "xray": (img, self.sample_cxr.name),
                "symptoms": "High fever, productive cough with greenish phlegm, pleuritic chest pain and dyspnea.",
                "age": "55",
                "temperature": "38.8",
                "heart_rate": "102",
                "respiratory_rate": "26",
                "spo2": "92",
                "cough": "1",
                "breathing_difficulty": "1",
            }
            response = self.client.post("/pneumonia", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tri-Modal Late-Fusion Diagnosis", response.data)
        self.assertIn(b"Individual Modality Confidence Breakdown", response.data)
        self.assertIn(b"Grad-CAM Heatmap Overlay Attention Map", response.data)
        self.assertIn(b"Download PDF Report", response.data)

    # =========================================================================
    # 4. PATIENT RECORDS DATABASE & PDF REPORT GENERATION TESTS
    # =========================================================================

    def test_16_patient_history_get(self):
        """Test GET /history renders patient records and statistics."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Patient Diagnostic Records", response.data)
        self.assertIn(b"Export All to CSV", response.data)

    def test_17_export_csv(self):
        """Test GET /export-csv streams valid CSV content."""
        response = self.client.get("/export-csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"text/csv", response.content_type.encode())
        self.assertIn(b"Patient ID,Patient Name", response.data)

    def test_18_pdf_report_download(self):
        """Test GET /download-report/<id> generates and serves a valid PDF."""
        import database
        records = database.get_all_records(limit=1)
        if not records:
            self.skipTest("No records in database to download PDF.")

        rec_id = records[0]["id"]
        response = self.client.get(f"/download-report/{rec_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/pdf")
        self.assertTrue(response.data.startswith(b"%PDF"))

    def test_19_delete_record(self):
        """Test POST /delete-record/<id> removes a record and redirects."""
        import database
        # Insert a temporary test record
        rec_id = database.insert_patient_record(
            patient_id="PT-DEL-TEMP",
            disease_module="COVID-19 Radiography",
            imaging_modality="Chest X-Ray",
            prediction="NORMAL",
            confidence_score=95.0,
            patient_name="Temporary Record"
        )
        response = self.client.post(f"/delete-record/{rec_id}", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(database.get_record_by_id(rec_id))


if __name__ == "__main__":
    unittest.main(verbosity=2)
