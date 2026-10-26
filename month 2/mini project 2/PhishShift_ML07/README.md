# PhishShift Message Risk Scanner — ML-07

Educational cybersecurity/NLP prototype. It classifies **synthetic training messages** as likely phishing, lower model-estimated risk, or abstain/manual review. It never opens URLs, fetches remote content, executes attachments, or asks for credentials.

## Run in VS Code (Windows)

1. Extract this ZIP to a folder.
2. Open the extracted `PhishShift_ML07` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and use that Python executable for the following commands.
5. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
6. Train and create the model/evidence files:
   ```powershell
   python train_model.py
   ```
   Wait for `Training complete.` before continuing.
7. Start the web app:
   ```powershell
   python app.py
   ```
8. Your browser should open automatically at **http://127.0.0.1:5000**. If not, copy that address into your browser. Keep the terminal open while using the app. Press `Ctrl+C` in the terminal to stop it.

## What is included

- `data/synthetic_messages.csv`: 720 locally authored synthetic examples with required schema fields, safe inert URL-feature labels, labels, campaign groups, and provenance.
- `train_model.py`: normalized duplicate removal, campaign-separated holdout, rule baseline, TF-IDF + Logistic Regression, TF-IDF + Random Forest, metrics, predictions, and robustness perturbation logs.
- `app.py`, `templates/`, `static/`: Flask scanner with risk-cue explanation, abstain zone, validation and automatic browser launch.
- `outputs/`: populated after training; evidence logs and comparison table.
- `docs/SRS.pdf`, `docs/final_report.pdf`: project report drafts. Replace student/mentor details and update claims/results using your actual run.
- `tests/test_acceptance.py`: basic API checks; app must be running in another terminal.

## Technical notes

- Group split uses `campaign_group`, and the script asserts no campaign overlap.
- The abstain rule is: probability ≥ 0.60 → suspicious; ≤ 0.40 → lower estimated risk; otherwise manual review. These are demo thresholds, not calibrated safety guarantees.
- `false_positive_rate = FP / (FP + TN)` for the phishing class.
- The included dataset is synthetic and is not a substitute for licensed public corpus evidence. The requirements request 400+ permitted examples with provenance; this package includes 720 generated examples and labels them as synthetic. Ask your mentor whether locally authored synthetic examples satisfy their source requirement.
- Results can vary by library versions and the generated data. Do not claim the prototype detects all phishing. Explain its limitations at review.
