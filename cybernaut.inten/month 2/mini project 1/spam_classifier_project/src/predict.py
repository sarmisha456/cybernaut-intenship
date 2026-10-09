"""Real-time classification.

    python -m src.predict "Free prize! Call now"
    python -m src.predict --interactive
"""
import argparse

import joblib
import numpy as np

from .config import MODEL_PATH
from .preprocess import clean_text


class SpamDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, message: str) -> dict:
        cleaned = clean_text(message)
        label = int(self.model.predict([cleaned])[0])
        try:
            prob = float(self.model.predict_proba([cleaned])[0][1])
        except AttributeError:   # e.g. LinearSVC
            prob = float(1 / (1 + np.exp(-2 * self.model.decision_function([cleaned])[0])))
        return {"label": "SPAM" if label else "HAM", "spam_confidence": round(prob, 4)}


def main():
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham")
    parser.add_argument("message", nargs="?", help="message text to classify")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run:  python -m src.train")
    detector = SpamDetector()

    if args.interactive:
        print("Type a message ('quit' to exit):")
        while True:
            msg = input("> ").strip()
            if msg.lower() in {"quit", "exit", "q"}:
                break
            if msg:
                print(detector.predict(msg))
    elif args.message:
        print(detector.predict(args.message))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
