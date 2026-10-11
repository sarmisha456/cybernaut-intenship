
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


# Load the trained spam classifier model
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}\n"
        "Please run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


# Homepage
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Email spam prediction API
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(
            error="Invalid request. Please send JSON data."
        ), 400

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
        confidence = float(max(probabilities)) * 100

        return jsonify(
            label=prediction,
            confidence=round(confidence, 2),
            is_spam=prediction.lower() == "spam"
        )

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify(
            error="Prediction failed. Please check the trained model."
        ), 500


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"

    # Open browser automatically after server starts
    threading.Timer(
        2.0,
        lambda: webbrowser.open_new_tab(url)
    ).start()

    print("\n" + "=" * 45)
    print("       EMAIL SPAM CLASSIFIER")
    print("=" * 45)
    print("Starting the Flask server...")
    print("Website: " + url)
    print("Keep this terminal open while using the website.\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
