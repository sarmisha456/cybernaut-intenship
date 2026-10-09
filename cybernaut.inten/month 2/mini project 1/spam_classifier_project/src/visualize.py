"""All plotting helpers. Figures are saved to the outputs/ folder."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import OUTPUT_DIR

PALETTE = {"ham": "#4C9F70", "spam": "#D64545"}


def _save(name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=130)
    plt.close()


def plot_class_distribution(df):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="label", data=df, order=["ham", "spam"],
                       palette=[PALETTE["ham"], PALETTE["spam"]])
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Class Distribution (Ham vs Spam)")
    _save("01_class_distribution.png")


def plot_length_distribution(df):
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="msg_length", hue="label", bins=50, kde=True, palette=PALETTE)
    plt.title("Message Length Distribution")
    plt.xlabel("Characters per message")
    _save("02_message_length.png")


def plot_confusion(cm, model_name, fname):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    _save(fname)


def plot_model_comparison(results_df):
    melted = results_df.melt(id_vars="Model",
                             value_vars=["Accuracy", "Precision", "Recall", "F1"],
                             var_name="Metric", value_name="Score")
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
    plt.ylim(0.8, 1.01)
    plt.title("Model Comparison")
    plt.legend(loc="lower right")
    _save("06_model_comparison.png")


def plot_roc(roc_data):
    plt.figure(figsize=(5.5, 5))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    _save("07_roc_curves.png")


def get_top_words(pipeline, top_n=20):
    """Return (spam_words, spam_weights, ham_words, ham_weights) from a linear model."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    features = np.array(vec.get_feature_names_out())
    coefs = clf.coef_.ravel()
    spam_idx = np.argsort(coefs)[-top_n:][::-1]
    ham_idx = np.argsort(coefs)[:top_n]
    return features[spam_idx], coefs[spam_idx], features[ham_idx], np.abs(coefs[ham_idx])


def plot_top_words(pipeline, top_n=20):
    sw, sc, hw, hc = get_top_words(pipeline, top_n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].barh(sw[::-1], sc[::-1], color=PALETTE["spam"])
    axes[0].set_title(f"Top {top_n} SPAM-indicating words")
    axes[1].barh(hw[::-1], hc[::-1], color=PALETTE["ham"])
    axes[1].set_title(f"Top {top_n} HAM-indicating words")
    for ax in axes:
        ax.set_xlabel("Weight")
    _save("08_top_keywords.png")
    print("\nTop spam keywords:", ", ".join(sw[:15]))
    print("Top ham keywords :", ", ".join(hw[:15]))
