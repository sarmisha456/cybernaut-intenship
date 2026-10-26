import os
import threading
import webbrowser

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "spam_classifier.joblib"
)
MAX_MESSAGE_LENGTH = 10_000

app = Flask(__name__)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(error="Invalid JSON request."), 400

    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Please enter an email message."), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify(
            error="Message must be 10,000 characters or fewer."
        ), 400

    try:
        prediction = str(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]

        
        classes = [str(c).strip().lower() for c in model.classes_]

        spam_probability = 0.0
        not_spam_probability = 0.0

        for class_name, probability in zip(classes, probabilities):
            if class_name in ("spam", "junk"):
                spam_probability += float(probability)
            else:
                not_spam_probability += float(probability)

        is_spam = prediction.strip().lower() in ("spam", "junk")
        confidence = max(probabilities) * 100

        return jsonify(
            label="Spam" if is_spam else "Not Spam",
            confidence=round(float(confidence), 2),
            spam_probability=round(spam_probability * 100, 2),
            not_spam_probability=round(not_spam_probability * 100, 2),
            is_spam=is_spam
        )

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify(
            error="Prediction failed. Check the trained model."
        ), 500


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"

    threading.Timer(
        2.0, lambda: webbrowser.open_new_tab(url)
    ).start()

    print("EMAIL SPAM CLASSIFIER")
    print("Website:", url)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
