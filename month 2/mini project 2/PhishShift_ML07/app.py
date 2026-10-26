
from pathlib import Path
import threading
import webbrowser
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tfidf_logistic.joblib"

app = Flask(__name__)

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

# Recent scans are kept while the app is running.
scan_history = []
MAX_HISTORY = 100


def explain(text):
    s = text.lower()

    cue_map = {
        "Credential request": [
            "password", "credentials", "login details",
            "one-time code", "verification code"
        ],
        "Urgency or threat": [
            "immediately", "urgent", "right away",
            "account suspension", "lockout"
        ],
        "Unusual link or sign-in": [
            "link", "sign in", "verify your account", "shortened"
        ],
        "Payment or reward lure": [
            "refund", "gift-card", "bonus",
            "banking information", "transfer"
        ],
        "Secrecy or process bypass": [
            "confidential", "do not contact",
            "bypass normal review"
        ]
    }

    found = []
    for label, terms in cue_map.items():
        hits = [term for term in terms if term in s]
        if hits:
            found.append({
                "cue": label,
                "evidence": ", ".join(hits[:3])
            })

    if not found:
        found.append({
            "cue": "No configured high-risk phrase matched",
            "evidence": "This does not prove the message is safe."
        })

    return found


def classify(text):
    if model is None:
        return {
            "error": "Model file is missing. Run train_model.py first."
        }, 500

    if not isinstance(text, str) or not text.strip():
        return {"error": "Enter a message before scanning."}, 400

    if len(text) > 10000:
        return {
            "error": "Message is too long. Maximum length is 10,000 characters."
        }, 400

    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)

    if "phishing" not in classes:
        return {"error": "Phishing class not found in the model."}, 500

    risk = float(probabilities[classes.index("phishing")])

    if risk >= 0.60:
        decision = "High risk — review as suspicious"
        category = "Phishing"
    elif risk <= 0.40:
        decision = "Lower model-estimated risk — verify independently"
        category = "Lower Risk"
    else:
        decision = "Abstain — manual review recommended"
        category = "Manual Review"

    return {
        "decision": decision,
        "category": category,
        "risk_probability": round(risk, 4),
        "risk_percent": round(risk * 100, 1),
        "risk_cues": explain(text),
        "notice": (
            "Decision support only, not a guarantee of safety. "
            "Do not open links or provide credentials based on this result."
        ),
        "model_version": "ML-07 TF-IDF + Logistic Regression v1"
    }, 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    result, status = classify(message)

    if status == 200:
        record = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message[:120],
            "category": result["category"],
            "risk_percent": result["risk_percent"],
            "decision": result["decision"]
        }

        scan_history.insert(0, record)
        del scan_history[MAX_HISTORY:]

    return jsonify(result), status


@app.route("/api/dashboard")
def dashboard():
    phishing_count = sum(
        1 for item in scan_history if item["category"] == "Phishing"
    )
    lower_risk_count = sum(
        1 for item in scan_history if item["category"] == "Lower Risk"
    )
    review_count = sum(
        1 for item in scan_history if item["category"] == "Manual Review"
    )

    return jsonify({
        "total_scans": len(scan_history),
        "phishing_count": phishing_count,
        "lower_risk_count": lower_risk_count,
        "manual_review_count": review_count,
        "history": scan_history
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "project": "PhishShift ML-07"
    })


if __name__ == "__main__":
    threading.Timer(
        1.2,
        lambda: webbrowser.open("http://127.0.0.1:5000")
    ).start()

    app.run(host="127.0.0.1", port=5000, debug=False)
