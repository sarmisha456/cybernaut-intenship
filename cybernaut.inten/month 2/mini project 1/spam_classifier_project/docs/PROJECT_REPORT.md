# Email Spam Classifier — Project Report

## 1. Introduction
Spam messages flood inboxes with irrelevant or malicious content. This project builds a
Machine Learning system that automatically classifies messages as spam or ham by analysing
their text using NLP and supervised learning.

## 2. Objectives
- Detect whether a message is spam or legitimate (ham)
- Extract meaningful features using text processing
- Compare Naive Bayes, Logistic Regression and SVM
- Achieve high accuracy and precision with a low false-positive rate
- Provide real-time classification and keyword visualisation

## 3. Dataset
UCI SMS Spam Collection (Kaggle). 5,572 messages, labelled `ham` / `spam`.
After removing duplicates: 5,169 messages (about 87% ham, 13% spam), so the data is
**imbalanced**. Because of this, precision / recall / F1 are reported in addition to accuracy,
and class weights are used for Logistic Regression and SVM.

## 4. Methodology
**Preprocessing:** lowercasing; URLs, e-mails, phone numbers and digits replaced by tokens
(these are strong spam signals); punctuation and stop-words removed; Porter stemming.

**Features:** TF-IDF (unigrams + bigrams, min_df=2, max_df=0.95, sublinear TF).

**Models:** Multinomial Naive Bayes (alpha=0.1), Logistic Regression (C=10, balanced),
Linear SVM (C=1, balanced).

**Evaluation:** stratified 80/20 split; accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, 5-fold cross-validation.

## 5. Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 0.9884 | 0.9837 | 0.9237 | 0.9528 | 0.9941 | 2 | 10 |
| Logistic Regression | 0.9855 | 0.9531 | 0.9313 | 0.9421 | 0.9962 | 6 | 9 |
| SVM (Linear) | 0.9845 | 0.9528 | 0.9237 | 0.9380 | 0.9965 | 6 | 10 |

Naive Bayes had the best F1 and the fewest false positives (only 2 of 903 legitimate
messages wrongly marked as spam). Cross-validated F1 was 0.94-0.95 for all three models,
showing the results are stable.

Charts are in the `outputs/` folder (class distribution, message length, confusion matrices,
model comparison, ROC curves, top spam keywords).

## 6. Key findings
- Spam messages are longer on average and contain phone numbers, URLs and numbers.
- Top spam indicators: phone/number tokens, "mobile", "text", "reply", "claim", "won", "service".
- Top ham indicators: casual conversational words such as "ok", "hey", "home", "later".

## 7. Conclusion
The system classifies messages with about 98-99% accuracy and a very low false-positive rate.
It is lightweight (one saved model file), fast, and ready to integrate into a mail or messaging
system. It was built as a scalable prototype and strengthened understanding of text
classification and data preprocessing.

## 8. Future work
- Train on email data (SpamAssassin) and use header / sender features
- Detect phishing, scams and advertisements as separate classes
- Try deep learning (LSTM, BERT)
- Deploy as an API or mail-client plug-in
