import pickle

# Load model
with open("models/nlp_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("NLP Symptom Prediction")
print("----------------------")

while True:
    text = input("\nEnter patient symptoms (or type 'exit'): ")

    if text.lower() == "exit":
        break

    # Convert text to TF-IDF
    text_tfidf = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_tfidf)[0]

    # Probability
    probability = model.predict_proba(text_tfidf)[0]

    classes = model.classes_

    print("\nPrediction:", prediction)

    print("Probabilities:")

    for class_name, prob in zip(classes, probability):
        print(f"{class_name}: {prob * 100:.2f}%")