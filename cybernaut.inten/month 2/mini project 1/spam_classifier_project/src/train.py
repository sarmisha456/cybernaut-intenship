"""Train + evaluate Naive Bayes, Logistic Regression and SVM, then save the best.

Run from the project root:
    python -m src.train
    python -m src.train --data data/spam.csv
"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import visualize as viz
from .config import (CV_FOLDS, DATA_PATH, MODEL_DIR, MODEL_PATH, OUTPUT_DIR,
                     RANDOM_STATE, TEST_SIZE)
from .data_loader import load_data
from .preprocess import clean_text


def build_models():
    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(C=10, max_iter=1000,
                                                                    class_weight="balanced"))]),
        "SVM (Linear)": Pipeline([("tfidf", tfidf()),
                                  ("clf", LinearSVC(C=1.0, class_weight="balanced"))]),
    }


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_and_evaluate(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=TEST_SIZE, stratify=df["target"],
        random_state=RANDOM_STATE)
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, roc_data, fitted = [], {}, {}

    for name, pipe in build_models().items():
        print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        auc = roc_auc_score(y_test, _scores(pipe, X_test))
        cv_f1 = cross_val_score(pipe, df["clean"], df["target"], cv=cv, scoring="f1").mean()

        print(classification_report(y_test, preds, target_names=["Ham", "Spam"], digits=4))
        print(f"False-positive rate : {fp / (fp + tn):.4f}  ({fp} ham marked as spam)")
        print(f"ROC-AUC             : {auc:.4f}")
        print(f"{CV_FOLDS}-fold CV F1        : {cv_f1:.4f}")

        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, preds),
                     "Precision": precision_score(y_test, preds),
                     "Recall": recall_score(y_test, preds),
                     "F1": f1_score(y_test, preds),
                     "ROC-AUC": auc, "CV F1": cv_f1,
                     "False Positives": fp, "False Negatives": fn})
        fpr, tpr, _ = roc_curve(y_test, _scores(pipe, X_test))
        roc_data[name] = (fpr, tpr, auc)
        fitted[name] = pipe
        viz.plot_confusion(cm, name, f"03_confusion_{name.split()[0].lower()}.png")

    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 60 + "\nMODEL COMPARISON\n" + "=" * 60)
    print(results.round(4).to_string(index=False))

    results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    viz.plot_model_comparison(results)
    viz.plot_roc(roc_data)
    viz.plot_top_words(fitted["Logistic Regression"])

    best = results.loc[0, "Model"]
    print(f"\nBest model (by F1): {best}")
    return fitted[best], best


def main():
    parser = argparse.ArgumentParser(description="Train the spam classifier")
    parser.add_argument("--data", default=str(DATA_PATH), help="path to spam.csv")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading data ...")
    df = load_data(args.data)
    print(f"Rows after cleaning/dedup: {len(df)}")
    print(df["label"].value_counts().to_string())

    df["msg_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    print("\nAverage length by class:")
    print(df.groupby("label")[["msg_length", "word_count"]].mean().round(1))
    viz.plot_class_distribution(df)
    viz.plot_length_distribution(df)

    print("\nPreprocessing text ...")
    df["clean"] = df["text"].apply(clean_text)

    best_model, best_name = train_and_evaluate(df)

    best_model.fit(df["clean"], df["target"])   # refit winner on all data for deployment
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved model ({best_name}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
