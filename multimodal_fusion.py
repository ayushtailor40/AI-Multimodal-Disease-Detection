import os
import pickle
import numpy as np
import tensorflow as tf

from PIL import Image


# =========================================================
# 1. LOAD ALL MODELS
# =========================================================

print("Loading models...")

# X-ray model
image_model = tf.keras.models.load_model(
    "models/best_chest_xray_model.keras"
)

# NLP model
with open("models/nlp_model.pkl", "rb") as f:
    nlp_model = pickle.load(f)

# TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clinical model
with open("models/clinical_model.pkl", "rb") as f:
    clinical_model = pickle.load(f)

# Clinical scaler
with open("models/clinical_scaler.pkl", "rb") as f:
    clinical_scaler = pickle.load(f)

print("All models loaded successfully!")


# =========================================================
# 2. X-RAY PREDICTION
# =========================================================

def predict_xray(image_path):

    if not os.path.exists(image_path):
        print("\nERROR: X-ray image not found!")
        return None

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    probability = image_model.predict(
        image_array,
        verbose=0
    )[0][0]

    pneumonia_probability = float(probability)
    normal_probability = 1 - pneumonia_probability

    return normal_probability, pneumonia_probability


# =========================================================
# 3. NLP / SYMPTOM PREDICTION
# =========================================================

def predict_symptoms(symptoms):

    text_vector = vectorizer.transform([symptoms])

    probabilities = nlp_model.predict_proba(
        text_vector
    )[0]

    classes = nlp_model.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    normal_probability = float(
        probability_dict.get("NORMAL", 0)
    )

    pneumonia_probability = float(
        probability_dict.get("PNEUMONIA", 0)
    )

    return normal_probability, pneumonia_probability


# =========================================================
# 4. CLINICAL DATA PREDICTION
# =========================================================

def predict_clinical(
    age,
    temperature,
    heart_rate,
    respiratory_rate,
    spo2,
    cough,
    breathing_difficulty
):

    clinical_data = np.array([[
        age,
        temperature,
        heart_rate,
        respiratory_rate,
        spo2,
        cough,
        breathing_difficulty
    ]])

    clinical_data_scaled = clinical_scaler.transform(
        clinical_data
    )

    probabilities = clinical_model.predict_proba(
        clinical_data_scaled
    )[0]

    classes = clinical_model.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    normal_probability = float(
        probability_dict.get("NORMAL", 0)
    )

    pneumonia_probability = float(
        probability_dict.get("PNEUMONIA", 0)
    )

    return normal_probability, pneumonia_probability


# =========================================================
# 5. THREE-MODALITY FUSION
# =========================================================

def multimodal_prediction(
    image_path,
    symptoms,
    age,
    temperature,
    heart_rate,
    respiratory_rate,
    spo2,
    cough,
    breathing_difficulty
):

    # -----------------------------------------
    # X-ray
    # -----------------------------------------

    xray_result = predict_xray(image_path)

    if xray_result is None:
        return

    image_normal, image_pneumonia = xray_result


    # -----------------------------------------
    # Symptoms
    # -----------------------------------------

    text_normal, text_pneumonia = predict_symptoms(
        symptoms
    )


    # -----------------------------------------
    # Clinical Data
    # -----------------------------------------

    clinical_normal, clinical_pneumonia = predict_clinical(
        age,
        temperature,
        heart_rate,
        respiratory_rate,
        spo2,
        cough,
        breathing_difficulty
    )


    # =====================================================
    # FUSION WEIGHTS
    # =====================================================

    image_weight = 0.40
    text_weight = 0.30
    clinical_weight = 0.30


    # =====================================================
    # FINAL FUSION
    # =====================================================

    final_normal = (
        image_weight * image_normal
        + text_weight * text_normal
        + clinical_weight * clinical_normal
    )

    final_pneumonia = (
        image_weight * image_pneumonia
        + text_weight * text_pneumonia
        + clinical_weight * clinical_pneumonia
    )


    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    if final_pneumonia >= final_normal:
        final_prediction = "PNEUMONIA"
    else:
        final_prediction = "NORMAL"


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n")
    print("=" * 60)
    print("          AI MULTIMODAL DISEASE DETECTION")
    print("=" * 60)


    print("\n1. X-RAY MODEL")
    print("-" * 60)

    print(
        f"NORMAL       : {image_normal * 100:.2f}%"
    )

    print(
        f"PNEUMONIA    : {image_pneumonia * 100:.2f}%"
    )


    print("\n2. SYMPTOM / NLP MODEL")
    print("-" * 60)

    print(
        f"NORMAL       : {text_normal * 100:.2f}%"
    )

    print(
        f"PNEUMONIA    : {text_pneumonia * 100:.2f}%"
    )


    print("\n3. CLINICAL DATA MODEL")
    print("-" * 60)

    print(
        f"NORMAL       : {clinical_normal * 100:.2f}%"
    )

    print(
        f"PNEUMONIA    : {clinical_pneumonia * 100:.2f}%"
    )


    print("\n4. THREE-MODALITY FUSION")
    print("-" * 60)

    print(
        f"NORMAL       : {final_normal * 100:.2f}%"
    )

    print(
        f"PNEUMONIA    : {final_pneumonia * 100:.2f}%"
    )


    print("\n5. FINAL PREDICTION")
    print("-" * 60)

    print(
        f"             {final_prediction}"
    )

    print("=" * 60)

    print("\nFusion Weights:")
    print("X-ray    : 40%")
    print("Symptoms : 30%")
    print("Clinical : 30%")

    print(
        "\nNOTE: This is a student research prototype, "
        "not a clinical diagnostic system."
    )


# =========================================================
# 6. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("AI MULTIMODAL DISEASE DETECTION")
    print("-" * 40)


    # X-ray
    image_path = input(
        "\nEnter X-ray image path: "
    ).strip()


    # Symptoms
    symptoms = input(
        "Enter patient symptoms: "
    ).strip()


    # Clinical information
    print("\nEnter Clinical Information")
    print("-" * 40)

    age = float(
        input("Age: ")
    )

    temperature = float(
        input("Temperature (°C): ")
    )

    heart_rate = float(
        input("Heart Rate (bpm): ")
    )

    respiratory_rate = float(
        input("Respiratory Rate: ")
    )

    spo2 = float(
        input("SpO2 (%): ")
    )

    cough = int(
        input("Cough? (1 = Yes, 0 = No): ")
    )

    breathing_difficulty = int(
        input(
            "Breathing Difficulty? (1 = Yes, 0 = No): "
        )
    )


    # Run multimodal system
    multimodal_prediction(
        image_path,
        symptoms,
        age,
        temperature,
        heart_rate,
        respiratory_rate,
        spo2,
        cough,
        breathing_difficulty
    )