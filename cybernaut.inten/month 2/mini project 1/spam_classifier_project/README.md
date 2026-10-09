# 📧 Email Spam Classifier

A Machine Learning project that classifies messages as **spam** or **ham (legitimate)**
using Natural Language Processing (NLP) and supervised learning.

- **Language:** Python
- **Libraries:** scikit-learn, pandas, NumPy, NLTK, Matplotlib, Seaborn, Streamlit
- **Algorithms:** Naive Bayes, Logistic Regression, Support Vector Machine (SVM)
- **Dataset:** UCI SMS Spam Collection (Kaggle) — 5,572 messages (5,169 after removing duplicates)

## Project structure

```
spam_classifier_project/
├── data/spam.csv               # dataset
├── src/
│   ├── config.py               # paths & constants
│   ├── data_loader.py          # load + clean the csv
│   ├── preprocess.py           # NLP text cleaning
│   ├── visualize.py            # all charts
│   ├── train.py                # train / evaluate 3 models, save best
│   └── predict.py              # real-time classifier (CLI + class)
├── app.py                      # Streamlit web app
├── tests/test_project.py       # unit tests
├── models/                     # saved model (.joblib)
├── outputs/                    # charts + model_results.csv
├── docs/PROJECT_REPORT.md      # report for submission
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage (run all commands from the project root folder)

```bash
python -m src.train                                   # train + evaluate + save model + charts
python -m src.predict "Free prize! Call now to claim" # classify one message
python -m src.predict --interactive                   # classify typed messages
streamlit run app.py                                  # launch the web app
pytest -q                                             # run tests
```

A trained model and charts are already included, so you can try the web app immediately.
If you see a scikit-learn version warning when loading the model, just run
`python -m src.train` once to rebuild it for your installed version.

## How it works

1. **Load & clean** – drop junk columns, null rows and duplicates; label spam=1, ham=0.
2. **NLP preprocessing** – lowercase, replace URLs / emails / phone numbers / numbers with
   tokens, remove punctuation and stop-words, stem with NLTK Porter stemmer.
3. **Feature extraction** – TF-IDF with unigrams + bigrams.
4. **Training** – 3 classifiers in scikit-learn Pipelines; 80/20 stratified split.
5. **Evaluation** – accuracy, precision, recall, F1, ROC-AUC, false-positive count and
   5-fold cross-validation.
6. **Deployment** – best model (by F1) is refit on all data and saved with joblib.

## Results (20% hold-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False positives |
|---|---|---|---|---|---|---|
| **Naive Bayes** | 98.84% | 98.37% | 92.37% | 0.953 | 0.994 | **2** |
| Logistic Regression | 98.55% | 95.31% | 93.13% | 0.942 | 0.996 | 6 |
| SVM (Linear) | 98.45% | 95.28% | 92.37% | 0.938 | 0.997 | 6 |

Exact numbers can vary slightly if NLTK stemming is available on your machine.

## Extending the project

- Retrain on email data (SpamAssassin) — just provide a csv with `label,text` columns.
- Add phishing / scam / advertisement classes (multi-class classification).
- Try deep learning (LSTM / BERT) or add sender / header features.
