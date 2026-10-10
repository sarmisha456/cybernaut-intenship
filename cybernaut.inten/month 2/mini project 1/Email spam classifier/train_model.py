
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("          EMAIL SPAM CLASSIFIER")
print("=" * 55)

# Load dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please place spam.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Kaggle SMS Spam Collection has v1 (label) and v2 (message)
if not {"v1", "v2"}.issubset(df.columns):
    raise ValueError(
        "Unexpected CSV columns. Please use the Kaggle "
        "SMS Spam Collection dataset containing v1 and v2 columns."
    )

df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]
df = df.dropna()
df["label"] = df["label"].str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()
df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"] != ""]

if df["label"].nunique() < 2:
    raise ValueError("Dataset must contain both ham and spam messages.")

print(f"\nTotal messages: {len(df)}")
print("\nClass counts:")
print(df["label"].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Build and train model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", MultinomialNB())
])

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate on unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
model_path = os.path.join(MODEL_DIR, "spam_classifier.joblib")
joblib.dump(model, model_path)

# Save confusion matrix
labels = ["ham", "spam"]
matrix = confusion_matrix(y_test, predictions, labels=labels)
matrix_df = pd.DataFrame(
    matrix,
    index=["Actual ham", "Actual spam"],
    columns=["Predicted ham", "Predicted spam"]
)
matrix_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

print(f"\nModel saved to: {model_path}")
print(f"Confusion matrix saved to: {OUTPUT_DIR}")
print("\nTraining completed successfully!")
