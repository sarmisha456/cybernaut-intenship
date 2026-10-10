import os

import joblib
from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_classifier.joblib")
MAX_MESSAGE_LENGTH = 10_000

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Enter a message to check."), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify(
            error=f"Message must be {MAX_MESSAGE_LENGTH:,} characters or fewer."
        ), 400

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = float(max(probabilities)) * 100

    return jsonify(
        label=prediction,
        confidence=round(confidence, 2),
        is_spam=prediction == "spam",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
