from pathlib import Path
import threading, webbrowser, os, re, json
from flask import Flask, render_template, request, jsonify
import joblib

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/"models"/"tfidf_logistic.joblib"
app=Flask(__name__)
model=None
if MODEL_PATH.exists():
    model=joblib.load(MODEL_PATH)

def explain(text):
    s=text.lower()
    cue_map={
      "credential request":["password","credentials","login details","one-time code","verification code"],
      "urgency or threat":["immediately","urgent","right away","within the next hour","account suspension","lockout","cancelled today","paused"],
      "unusual link or sign-in":["link","sign in","verify your account","shortened"],
      "payment or reward lure":["payroll adjustment","refund","gift-card","bonus","banking information","transfer"],
      "secrecy or process bypass":["confidential","keep this confidential","do not contact","bypass normal review"]
    }
    found=[]
    for label,terms in cue_map.items():
        hits=[term for term in terms if term in s]
        if hits: found.append({"cue":label,"evidence":", ".join(hits[:3])})
    if not found:
        found=[{"cue":"No configured high-risk phrase matched","evidence":"This does not prove the message is safe."}]
    return found

def classify(text):
    if model is None:
        return {"error":"Model file is missing. Run python train_model.py first."},500
    if not isinstance(text,str) or not text.strip():
        return {"error":"Enter a message before scanning."},400
    if len(text)>10000:
        return {"error":"Message is too long. Maximum length is 10,000 characters."},400
    probs=model.predict_proba([text])[0]
    classes=list(model.classes_)
    p=float(probs[classes.index("phishing")])
    if p>=0.60:
        decision="High risk — review as suspicious"
    elif p<=0.40:
        decision="Lower model-estimated risk — still verify independently"
    else:
        decision="Abstain — manual review recommended"
    return {"decision":decision,"risk_probability":round(p,4),"risk_percent":round(p*100,1),
            "risk_cues":explain(text),
            "notice":"Decision support only, not a guarantee of safety. Do not open links or provide credentials based on this result.",
            "model_version":"ML-07 TF-IDF + Logistic Regression v1"},200

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/api/predict",methods=["POST"])
def predict():
    payload=request.get_json(silent=True) or {}
    result,status=classify(payload.get("message",""))
    return jsonify(result),status
@app.route("/health")
def health():
    return jsonify({"status":"ok","model_loaded":model is not None,"project":"PhishShift ML-07"})
if __name__=="__main__":
    # Open browser automatically when launched from the terminal.
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1",port=5000,debug=False)
