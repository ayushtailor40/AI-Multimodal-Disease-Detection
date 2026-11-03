"""
=============================================================================
AI MULTIMODAL EARLY DISEASE DETECTION - MASTER FLASK BACKEND
MCA Final-Year Capstone Project
=============================================================================
Architecture:
  - Multimodal Pneumonia Fusion (Chest X-Ray CNN + Symptom NLP + Clinical Vitals)
  - Explainable AI (Grad-CAM Saliency Maps)
  - COVID-19 Pulmonary Radiography Classifier (4-Class)
  - Tuberculosis Radiographic Screener (Binary)
  - Lung Cancer CT Histopathology Classifier (3-Class)
  - Comprehensive Dataset & Model Evaluation Dashboard
  - Patient Diagnostic History Database (SQLite)
  - Official Clinical Diagnostic PDF Report Generator
  - Automated CSV Export
=============================================================================
"""

import csv
import io
import os
import pickle
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_file, Response
from werkzeug.utils import secure_filename

import database
import report_generator

# =============================================================================
# BASE PATHS & FLASK SETUP
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload ceiling
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

print("\n" + "=" * 65)
print("     AI MULTIMODAL EARLY DISEASE DETECTION - MEDISCAN AI")
print("=" * 65)
print("Loading trained deep learning and machine learning models...")

# =============================================================================
# 1. LOAD TRAINED MODELS & ARTIFACTS
# =============================================================================

# A. Chest X-Ray Model (Pneumonia)
xray_model_path = MODELS_DIR / "best_chest_xray_model.keras"
if not xray_model_path.exists():
    xray_model_path = MODELS_DIR / "chest_xray_model.keras"
image_model = tf.keras.models.load_model(str(xray_model_path))
print(" -> Loaded Chest X-Ray Pneumonia Model:", xray_model_path.name)

# B. Symptom NLP Model & TF-IDF Vectorizer
with open(MODELS_DIR / "nlp_model.pkl", "rb") as f:
    nlp_model = pickle.load(f)
with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
print(" -> Loaded Symptom NLP Model & TF-IDF Vectorizer")

# C. Clinical Tabular Model & Scaler
with open(MODELS_DIR / "clinical_model.pkl", "rb") as f:
    clinical_model = pickle.load(f)
with open(MODELS_DIR / "clinical_scaler.pkl", "rb") as f:
    clinical_scaler = pickle.load(f)
print(" -> Loaded Clinical Vitals Classifier & StandardScaler")

# D. Separate Disease Models (Zero-leakage, completely independent)
covid_model_path = MODELS_DIR / "best_covid_model.keras"
if not covid_model_path.exists():
    covid_model_path = MODELS_DIR / "covid_model.keras"
covid_model = tf.keras.models.load_model(str(covid_model_path))

tb_model_path = MODELS_DIR / "best_tb_model.keras"
if not tb_model_path.exists():
    tb_model_path = MODELS_DIR / "tb_model.keras"
tb_model = tf.keras.models.load_model(str(tb_model_path))

lung_cancer_model_path = MODELS_DIR / "best_lung_cancer_model.keras"
if not lung_cancer_model_path.exists():
    lung_cancer_model_path = MODELS_DIR / "lung_cancer_model.keras"
lung_cancer_model = tf.keras.models.load_model(str(lung_cancer_model_path))

# Class ordering matching training directory structures
covid_classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
tb_classes = ["NORMAL", "TUBERCULOSIS"]
lung_cancer_classes = ["Benign", "Malignant", "Normal"]

print(" -> Loaded Separate COVID, TB, and Lung Cancer Deep Learning Models")
print("All AI diagnostic models initialized successfully!\n" + "=" * 65 + "\n")


# =============================================================================
# HELPER FUNCTIONS & VALIDATION
# =============================================================================

def allowed_image_file(filename: str) -> bool:
    """Validate file extension against permitted medical image formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def verify_image(image_path: str) -> bool:
    """Verify that file is a readable image and not corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


# =============================================================================
# MULTIMODAL PREDICTION WORKFLOW (PNEUMONIA)
# =============================================================================

def predict_xray(image_path: str):
    """
    Inference on Chest X-Ray image.
    EfficientNetB0 binary classifier (0 = NORMAL, 1 = PNEUMONIA).
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = image_model.predict(image_array, verbose=0)[0][0]
    pneumonia_probability = float(prediction)
    normal_probability = 1.0 - pneumonia_probability

    return normal_probability, pneumonia_probability


def predict_symptoms(symptoms_text: str):
    """
    Inference on symptom narrative using TF-IDF + NLP Classifier.
    """
    text_vector = vectorizer.transform([symptoms_text])
    probabilities = nlp_model.predict_proba(text_vector)[0]
    classes = nlp_model.classes_

    result = dict(zip(classes, probabilities))
    normal_prob = float(result.get("NORMAL", 0.0))
    pneumonia_prob = float(result.get("PNEUMONIA", 0.0))

    return normal_prob, pneumonia_prob


def predict_clinical(age, temp, hr, rr, spo2, cough, breathing_diff):
    """
    Inference on 7 clinical vitals using StandardScaler + Classifier.
    """
    clinical_df = pd.DataFrame([{
        "age": float(age),
        "temperature": float(temp),
        "heart_rate": float(hr),
        "respiratory_rate": float(rr),
        "spo2": float(spo2),
        "cough": int(cough),
        "breathing_difficulty": int(breathing_diff)
    }])

    clinical_data_scaled = clinical_scaler.transform(clinical_df)
    probabilities = clinical_model.predict_proba(clinical_data_scaled)[0]
    classes = clinical_model.classes_

    result = dict(zip(classes, probabilities))
    normal_prob = float(result.get("NORMAL", 0.0))
    pneumonia_prob = float(result.get("PNEUMONIA", 0.0))

    return normal_prob, pneumonia_prob


def generate_gradcam(image_path: str, output_path: str) -> str:
    """
    Generate Grad-CAM activation heatmap overlay for Chest X-Ray radiograph.
    Extracts gradient of output class with respect to 'top_conv' layer in EfficientNetB0.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    efficientnet = image_model.get_layer("efficientnetb0")
    last_conv_layer = efficientnet.get_layer("top_conv")

    grad_model = tf.keras.models.Model(
        inputs=efficientnet.input,
        outputs=last_conv_layer.output
    )

    efficientnet_index = image_model.layers.index(efficientnet)

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array, training=False)
        x = conv_outputs
        for layer in image_model.layers[efficientnet_index + 1:]:
            x = layer(x, training=False)
        prediction = x[:, 0]

        # Target class: PNEUMONIA if >= 0.5, else NORMAL
        if float(prediction[0]) >= 0.5:
            loss = prediction[0]
        else:
            loss = 1.0 - prediction[0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient computation returned None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = float(tf.reduce_max(heatmap).numpy())
    if max_value > 0:
        heatmap = heatmap / max_value
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(image)
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original, 0.60, heatmap_color, 0.40, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    output_path = os.path.splitext(output_path)[0] + ".jpg"
    success = cv2.imwrite(output_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError(f"Could not save Grad-CAM image at: {output_path}")

    return output_path


def execute_pneumonia_multimodal_pipeline(request_obj):
    """
    Executes the complete tri-modal multimodal fusion pipeline:
      - Validates and saves X-ray
      - Evaluates image, text, and clinical models
      - Computes weighted late-fusion decision (40% image, 30% text, 30% clinical)
      - Generates Grad-CAM explainability heatmap
      - Logs patient record to SQLite database for PDF generation & history tracking
    """
    image_file = request_obj.files.get("xray")
    if not image_file or image_file.filename == "":
        raise ValueError("Please select a Chest X-Ray radiograph image.")

    orig_name = secure_filename(image_file.filename)
    if not orig_name or not allowed_image_file(orig_name):
        raise ValueError("Unsupported image file. Use JPG, JPEG, PNG, BMP, or TIFF.")

    unique_id = uuid.uuid4().hex[:8]
    filename = f"pneumonia_{unique_id}_{orig_name}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(image_path)

    if not verify_image(image_path):
        os.remove(image_path)
        raise ValueError("The uploaded file could not be parsed as a valid image.")

    gradcam_filename = f"gradcam_{unique_id}.jpg"
    gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)

    symptoms = request_obj.form.get("symptoms", "").strip()
    if not symptoms:
        raise ValueError("Please provide patient symptom observations for the NLP model.")

    patient_name = request_obj.form.get("patient_name", "").strip() or "Anonymous Patient"
    patient_gender = request_obj.form.get("patient_gender", "Unspecified")
    patient_id = f"PT-PNEU-{uuid.uuid4().hex[:6].upper()}"

    age = float(request_obj.form.get("age", 45))
    temp = float(request_obj.form.get("temperature", 37.0))
    hr = float(request_obj.form.get("heart_rate", 75))
    rr = float(request_obj.form.get("respiratory_rate", 16))
    spo2 = float(request_obj.form.get("spo2", 98))
    cough = int(request_obj.form.get("cough", 0))
    breathing_diff = int(request_obj.form.get("breathing_difficulty", 0))

    # Inference across the 3 independent modalities
    img_norm, img_pneu = predict_xray(image_path)
    txt_norm, txt_pneu = predict_symptoms(symptoms)
    cli_norm, cli_pneu = predict_clinical(age, temp, hr, rr, spo2, cough, breathing_diff)

    # Late Fusion Consensus (Weights: 40% Image, 30% Text, 30% Clinical)
    w_img, w_txt, w_cli = 0.40, 0.30, 0.30
    final_norm = (w_img * img_norm) + (w_txt * txt_norm) + (w_cli * cli_norm)
    final_pneu = (w_img * img_pneu) + (w_txt * txt_pneu) + (w_cli * cli_pneu)

    prediction = "PNEUMONIA" if final_pneu >= final_norm else "NORMAL"
    confidence = round(final_pneu * 100.0, 2) if prediction == "PNEUMONIA" else round(final_norm * 100.0, 2)

    # Generate Grad-CAM Saliency Map
    generate_gradcam(image_path, gradcam_path)

    vitals_dict = {
        "temperature": temp,
        "heart_rate": hr,
        "respiratory_rate": rr,
        "spo2": spo2,
        "cough": cough,
        "breathing_difficulty": breathing_diff,
    }

    scores_dict = {
        "NORMAL (Fusion)": round(final_norm * 100.0, 2),
        "PNEUMONIA (Fusion)": round(final_pneu * 100.0, 2),
        "Image Normal": round(img_norm * 100.0, 2),
        "Image Pneumonia": round(img_pneu * 100.0, 2),
        "Symptom NLP Normal": round(txt_norm * 100.0, 2),
        "Symptom NLP Pneumonia": round(txt_pneu * 100.0, 2),
        "Clinical Vitals Normal": round(cli_norm * 100.0, 2),
        "Clinical Vitals Pneumonia": round(cli_pneu * 100.0, 2),
    }

    # Store record into SQLite database
    record_id = database.insert_patient_record(
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=age,
        patient_gender=patient_gender,
        disease_module="Pneumonia Multimodal",
        imaging_modality="Chest X-Ray (PA View)",
        prediction=prediction,
        confidence_score=confidence,
        scores_dict=scores_dict,
        vitals_dict=vitals_dict,
        symptoms_text=symptoms,
        image_file=filename,
        gradcam_file=gradcam_filename,
    )

    return {
        "record_id": record_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "prediction": prediction,
        "final_pneumonia": round(final_pneu * 100.0, 2),
        "final_normal": round(final_norm * 100.0, 2),
        "image_pneumonia": round(img_pneu * 100.0, 2),
        "image_normal": round(img_norm * 100.0, 2),
        "text_pneumonia": round(txt_pneu * 100.0, 2),
        "text_normal": round(txt_norm * 100.0, 2),
        "clinical_pneumonia": round(cli_pneu * 100.0, 2),
        "clinical_normal": round(cli_norm * 100.0, 2),
        "image_file": filename,
        "gradcam_file": gradcam_filename,
    }


# =============================================================================
# GENERIC SINGLE-IMAGE CLASSIFIER (COVID, TB, LUNG CANCER)
# =============================================================================

def predict_single_image(image_path: str, model, class_names):
    """
    Predict single medical image.
    Handles both binary sigmoid models (e.g. TB) and multi-class softmax models (COVID, Lung Cancer).
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    raw_output = model.predict(image_array, verbose=0)[0]

    # Binary classification with 1 sigmoid output (e.g. TB Model: 0=NORMAL, 1=TUBERCULOSIS)
    if len(raw_output) == 1 and len(class_names) == 2:
        pos_prob = float(raw_output[0])
        neg_prob = 1.0 - pos_prob
        best_index = 1 if pos_prob >= 0.5 else 0
        scores = {
            class_names[0]: round(neg_prob * 100.0, 2),
            class_names[1]: round(pos_prob * 100.0, 2),
        }
        best_score = round((pos_prob if best_index == 1 else neg_prob) * 100.0, 2)
        return {
            "label": class_names[best_index],
            "score": best_score,
            "scores": scores,
        }

    # Multi-class softmax output (e.g. COVID 4-class, Lung Cancer 3-class)
    if len(raw_output) != len(class_names):
        raise ValueError(
            f"Model output dimension ({len(raw_output)}) does not match configured class names ({len(class_names)})."
        )

    best_index = int(np.argmax(raw_output))
    return {
        "label": class_names[best_index],
        "score": round(float(raw_output[best_index]) * 100.0, 2),
        "scores": {
            label: round(float(raw_output[i]) * 100.0, 2)
            for i, label in enumerate(class_names)
        },
    }


def handle_disease_image_upload(model, class_names, prefix, disease_name, modality_name):
    """Handles image upload, prediction, and DB logging for separate disease modules."""
    result = None
    error = None

    if request.method == "POST":
        try:
            uploaded = request.files.get("scan")
            if not uploaded or uploaded.filename == "":
                raise ValueError("Please select a medical image first.")

            original_name = secure_filename(uploaded.filename)
            if not original_name or not allowed_image_file(original_name):
                raise ValueError("Unsupported image file type. Please use JPG, JPEG, PNG, BMP, or TIFF.")

            filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{original_name}"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded.save(image_path)

            if not verify_image(image_path):
                os.remove(image_path)
                raise ValueError("The uploaded file could not be verified as a valid image.")

            patient_name = request.form.get("patient_name", "").strip() or "Anonymous Patient"
            patient_gender = request.form.get("patient_gender", "Unspecified")
            patient_age_str = request.form.get("patient_age", "").strip()
            patient_age = float(patient_age_str) if patient_age_str else None
            patient_id = f"PT-{prefix.upper()[:4]}-{uuid.uuid4().hex[:6].upper()}"

            prediction = predict_single_image(image_path, model, class_names)

            # Store in SQLite database
            record_id = database.insert_patient_record(
                patient_id=patient_id,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                disease_module=disease_name,
                imaging_modality=modality_name,
                prediction=prediction["label"],
                confidence_score=prediction["score"],
                scores_dict=prediction["scores"],
                image_file=filename,
            )

            result = {
                "record_id": record_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "image_file": filename,
                "prediction": prediction["label"],
                "score": prediction["score"],
                "scores": prediction["scores"],
            }
        except Exception as exc:
            app.logger.exception("%s inference exception", prefix)
            error = str(exc) if isinstance(exc, ValueError) else "Image processing failed. Please check the file."

    return result, error


# =============================================================================
# FLASK ROUTE CONTROLLERS
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Overview Landing Dashboard.
    Supports POST for backward compatibility with legacy multimodal forms.
    """
    if request.method == "POST":
        result = None
        error = None
        try:
            result = execute_pneumonia_multimodal_pipeline(request)
        except Exception as exc:
            error = str(exc)
        return render_template(
            "pneumonia.html",
            active_page="pneumonia",
            result=result,
            error=error
        )

    return render_template(
        "index.html",
        active_page="dashboard"
    )


@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia_detection():
    """Pneumonia Tri-Modal Detection Pipeline."""
    result = None
    error = None
    if request.method == "POST":
        try:
            result = execute_pneumonia_multimodal_pipeline(request)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "pneumonia.html",
        active_page="pneumonia",
        result=result,
        error=error
    )


@app.route("/covid", methods=["GET", "POST"])
def covid_detection():
    """COVID-19 Chest X-Ray Research Module."""
    result, error = handle_disease_image_upload(
        covid_model, covid_classes, "covid", "COVID-19 Radiography", "Chest X-Ray (PA/AP)"
    )
    return render_template(
        "covid.html",
        active_page="covid",
        result=result,
        error=error,
        class_names=covid_classes,
    )


@app.route("/tb", methods=["GET", "POST"])
def tb_detection():
    """Tuberculosis Chest X-Ray Research Module."""
    result, error = handle_disease_image_upload(
        tb_model, tb_classes, "tb", "Tuberculosis Screening", "Chest X-Ray (PA View)"
    )
    return render_template(
        "tb.html",
        active_page="tb",
        result=result,
        error=error,
        class_names=tb_classes,
    )


@app.route("/lung-cancer", methods=["GET", "POST"])
def lung_cancer_detection():
    """Lung Cancer CT Research Module."""
    result, error = handle_disease_image_upload(
        lung_cancer_model, lung_cancer_classes, "lung_ct", "Lung Cancer CT", "Axial Chest CT Scan"
    )
    return render_template(
        "lung_cancer.html",
        active_page="lung_cancer",
        result=result,
        error=error,
        class_names=lung_cancer_classes,
    )


@app.route("/evaluation", methods=["GET"])
def evaluation_dashboard():
    """Dataset & Model Evaluation Benchmarking Dashboard."""
    return render_template(
        "evaluation.html",
        active_page="evaluation"
    )


@app.route("/about", methods=["GET"])
def about_project():
    """MCA Capstone Project Documentation & Architecture."""
    return render_template(
        "about.html",
        active_page="about"
    )


# =============================================================================
# PATIENT HISTORY, PDF REPORT & CSV EXPORT ROUTES
# =============================================================================

@app.route("/history", methods=["GET"])
def patient_history():
    """Patient Diagnostic Records & Historical Logs."""
    search_query = request.args.get("search", "").strip()
    disease_filter = request.args.get("disease", "").strip()

    records = database.get_all_records(search=search_query, disease_filter=disease_filter)
    stats = database.get_statistics()

    return render_template(
        "history.html",
        active_page="history",
        records=records,
        stats=stats,
        search_query=search_query,
        disease_filter=disease_filter,
    )


@app.route("/download-report/<int:record_id>", methods=["GET"])
def download_report(record_id):
    """Download official clinical PDF report for a patient record."""
    row = database.get_record_by_id(record_id)
    if not row:
        return redirect(url_for("patient_history"))

    record_dict = dict(row)
    pdf_stream = report_generator.generate_diagnostic_pdf(record_dict)

    patient_id = record_dict.get("patient_id", f"REC-{record_id}")
    filename = f"Diagnostic_Report_{patient_id}.pdf"

    return send_file(
        pdf_stream,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/export-csv", methods=["GET"])
def export_csv():
    """Export all patient diagnostic records to a downloadable CSV file."""
    records = database.get_all_records(limit=1000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID",
        "Patient ID",
        "Patient Name",
        "Age",
        "Gender",
        "Disease Module",
        "Imaging Modality",
        "AI Finding",
        "Confidence Score (%)",
        "Reported Symptoms",
        "Date and Time"
    ])

    for r in records:
        writer.writerow([
            r["id"],
            r["patient_id"],
            r["patient_name"],
            r["patient_age"] if r["patient_age"] else "N/A",
            r["patient_gender"],
            r["disease_module"],
            r["imaging_modality"],
            r["prediction"],
            f"{r['confidence_score']:.2f}",
            r["symptoms_text"] or "",
            r["created_at"],
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=Patient_Diagnostic_Records.csv"},
    )


@app.route("/delete-record/<int:record_id>", methods=["POST"])
def delete_patient_record(record_id):
    """Delete a patient record by ID."""
    database.delete_record(record_id)
    return redirect(url_for("patient_history"))


@app.errorhandler(413)
def upload_too_large(_error):
    return "The uploaded medical image exceeds the maximum permitted limit of 10 MB.", 413


@app.errorhandler(404)
def not_found(_error):
    return redirect(url_for("home"))


# =============================================================================
# APPLICATION ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
