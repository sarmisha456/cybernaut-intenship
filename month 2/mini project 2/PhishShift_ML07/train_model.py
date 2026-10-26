from pathlib import Path
import json, re, hashlib
import pandas as pd
import joblib
from difflib import SequenceMatcher
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_messages.csv"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\b(training|reference)\s+\w+\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def deduplicate(frame):
    # Exact normalized duplicate removal, followed by a conservative near-duplicate
    # check within each label. Keep the first record and log removed IDs.
    kept, removed, seen = [], [], {}
    for _, row in frame.iterrows():
        key = (row["label"], normalize(row["subject_or_title"] + " " + row["body_text"]))
        if key in seen:
            removed.append({"message_id": row["message_id"], "duplicate_of": seen[key], "method": "normalized_exact"})
        else:
            seen[key] = row["message_id"]; kept.append(row.to_dict())
    clean = pd.DataFrame(kept)
    # Similarity check: only compare records sharing label; threshold deliberately high.
    drop_ids = set()
    recs = clean.to_dict("records")
    for i in range(len(recs)):
        if recs[i]["message_id"] in drop_ids: continue
        a = normalize(recs[i]["subject_or_title"] + " " + recs[i]["body_text"])
        for j in range(i+1, len(recs)):
            if recs[j]["label"] != recs[i]["label"] or recs[j]["message_id"] in drop_ids: continue
            b = normalize(recs[j]["subject_or_title"] + " " + recs[j]["body_text"])
            if a and b and SequenceMatcher(None, a, b).ratio() >= 0.96:
                drop_ids.add(recs[j]["message_id"])
                removed.append({"message_id": recs[j]["message_id"], "duplicate_of": recs[i]["message_id"], "method": "near_duplicate_similarity_0.96"})
    clean = clean[~clean["message_id"].isin(drop_ids)].reset_index(drop=True)
    pd.DataFrame(removed, columns=["message_id","duplicate_of","method"]).to_csv(OUT/"deduplication_log.csv", index=False)
    return clean

def rule_predict(text):
    s = str(text).lower()
    cues = ["password", "verify your account", "credentials", "one-time code", "gift-card",
            "banking information", "shortened link", "unfamiliar message link", "bypass normal review",
            "account suspension", "cancelled today", "send your password", "provide your password"]
    score = sum(1 for cue in cues if cue in s)
    return "phishing" if score >= 1 else "benign"

def metrics(y_true, y_pred):
    labels=["benign","phishing"]
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return {"accuracy": round(accuracy_score(y_true,y_pred),4),
            "precision_phishing": round(precision_score(y_true,y_pred,pos_label="phishing",zero_division=0),4),
            "recall_phishing": round(recall_score(y_true,y_pred,pos_label="phishing",zero_division=0),4),
            "f1_phishing": round(f1_score(y_true,y_pred,pos_label="phishing",zero_division=0),4),
            "false_positive_rate": round(fp/(fp+tn),4) if fp+tn else 0.0,
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def main():
    raw=pd.read_csv(DATA).fillna("")
    cleaned=deduplicate(raw)
    # Group holdout: complete campaign groups are held out from training.
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
    train_idx,test_idx=next(splitter.split(cleaned, cleaned["label"], groups=cleaned["campaign_group"]))
    train,test=cleaned.iloc[train_idx].copy(),cleaned.iloc[test_idx].copy()
    assert set(train.campaign_group).isdisjoint(set(test.campaign_group)), "Campaign leakage detected"
    train.to_csv(OUT/"train_partition.csv",index=False)
    test.to_csv(OUT/"heldout_campaign_test.csv",index=False)
    X_train=(train["subject_or_title"].astype(str)+" "+train["body_text"].astype(str))
    X_test=(test["subject_or_title"].astype(str)+" "+test["body_text"].astype(str))
    y_train,y_test=train["label"],test["label"]
    results={}
    # Baseline rules
    rule_preds=[rule_predict(x) for x in X_test]
    results["Rules baseline"]=metrics(y_test,rule_preds)
    # TF-IDF + Logistic Regression linear classifier
    linear=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=1, max_features=12000)),
                     ("clf",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
    linear.fit(X_train,y_train)
    preds=linear.predict(X_test)
    results["TF-IDF + Logistic Regression"]=metrics(y_test,preds)
    joblib.dump(linear,MODELS/"tfidf_logistic.joblib")
    # Third model: TF-IDF + Random Forest
    forest=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=6000)),
                     ("clf",RandomForestClassifier(n_estimators=160,random_state=42,class_weight="balanced",min_samples_leaf=2))])
    forest.fit(X_train,y_train)
    results["TF-IDF + Random Forest"]=metrics(y_test,forest.predict(X_test))
    joblib.dump(forest,MODELS/"tfidf_random_forest.joblib")
    results["Rules baseline"]["model_type"]="handwritten cue rules"
    results["TF-IDF + Logistic Regression"]["model_type"]="linear TF-IDF classifier"
    results["TF-IDF + Random Forest"]["model_type"]="third model comparison"
    with open(OUT/"model_comparison.json","w",encoding="utf-8") as f: json.dump(results,f,indent=2)
    pd.DataFrame([{"model":k,**v} for k,v in results.items()]).to_csv(OUT/"model_comparison.csv",index=False)
    # Per-message heldout predictions for review.
    prob=linear.predict_proba(X_test)
    classes=list(linear.classes_)
    p_phish=prob[:,classes.index("phishing")]
    pred_labels=[]
    for p in p_phish:
        pred_labels.append("phishing" if p>=0.60 else ("benign" if p<=0.40 else "review"))
    review=pd.DataFrame({"message_id":test["message_id"],"campaign_group":test["campaign_group"],
                         "actual":y_test.values,"predicted_with_abstain":pred_labels,
                         "phishing_probability":p_phish.round(4),"body_text":test["body_text"].values})
    review.to_csv(OUT/"heldout_predictions.csv",index=False)
    # Controlled perturbation robustness test: punctuation and harmless urgency phrasing.
    samples=[
      ("AT-01","benign","Urgent: please reset your password using the official company portal you normally visit."),
      ("AT-02","phishing","A quick note from support: your mailbox access needs attention; share your login details to keep it active."),
      ("AT-03","phishing","SECURITY alert!!! verify-account now (shortened link is represented as [dot] and [slash], never opened)."),
      ("CP-5-1","benign","Action needed today: please submit your timesheet through the normal employee portal."),
      ("CP-5-2","phishing","Your work access may pause; send the one-time code to the sender right away."),
      ("CP-5-3","benign","Urgent payroll reminder: payslips are available in the established HR portal."),
      ("CP-5-4","phishing","A refund is waiting; enter banking information using the message link."),
      ("CP-5-5","benign","The helpdesk will perform maintenance at 7 PM. No response is required."),
      ("CP-5-6","phishing","Please validate your identity through the unfamiliar shortened link today."),
      ("CP-5-7","benign","Please review the meeting agenda before the 4 PM discussion."),
      ("CP-5-8","phishing","Do not contact anyone else; provide your password to prevent a lockout.")
    ]
    pert=[]
    for case,label,txt in samples:
      variants=[txt,txt.replace("!",".").replace("  "," "),re.sub(r'\s+',' ',txt.replace(":", " —"))]
      for j,v in enumerate(variants):
        p=float(linear.predict_proba([v])[0][classes.index("phishing")])
        pert.append({"case":case,"variant":j+1,"expected_label":label,"text":v,"phishing_probability":round(p,4),
                     "decision":"phishing" if p>=.60 else ("benign" if p<=.40 else "review")})
    pd.DataFrame(pert).to_csv(OUT/"robustness_tests.csv",index=False)
    # Data/partition evidence summary
    summary={"raw_records":int(len(raw)),"after_deduplication":int(len(cleaned)),
             "duplicates_removed":int(len(raw)-len(cleaned)),"train_records":int(len(train)),
             "test_records":int(len(test)),"train_campaigns":sorted(train.campaign_group.unique().tolist()),
             "test_campaigns":sorted(test.campaign_group.unique().tolist()),
             "campaign_overlap":sorted(set(train.campaign_group)&set(test.campaign_group)),
             "label_distribution":cleaned.label.value_counts().to_dict(),
             "dataset_note":"All examples are locally authored synthetic training messages; no real inbox data or active URLs."}
    with open(OUT/"data_split_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    print("Training complete.")
    print(json.dumps(summary,indent=2))
    print(json.dumps(results,indent=2))
if __name__=="__main__": main()
