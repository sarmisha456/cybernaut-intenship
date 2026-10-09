"""NLP text preprocessing.

Steps: lowercase -> replace URL/email/phone/number with tokens -> remove
punctuation -> remove stop-words -> stem (NLTK Porter, if available).
"""
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)
STEMMER = None
try:  # NLTK is optional: the code still works without it
    import nltk
    from nltk.stem import PorterStemmer

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words("english"))
    STEMMER = PorterStemmer()
except Exception:
    pass

URL_RE = re.compile(r"(http\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")
NUM_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    if STEMMER is not None:
        tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)
