"""Run with:  pytest -q   (from the project root; train the model first)"""
import pytest

from src.config import MODEL_PATH
from src.data_loader import load_data
from src.preprocess import clean_text


def test_clean_text_replaces_tokens():
    out = clean_text("WIN a FREE prize!!! Visit http://spam.com or call 09061701461")
    assert "urltoken" in out
    assert "phonetoken" in out
    assert "!" not in out
    assert out == out.lower()


def test_clean_text_removes_stopwords():
    assert "the" not in clean_text("the cat is on the mat").split()


def test_load_data_shape():
    df = load_data()
    assert list(df.columns) == ["label", "text", "target"]
    assert set(df["target"].unique()) == {0, 1}
    assert df["text"].notna().all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_model_predicts_obvious_cases():
    from src.predict import SpamDetector

    d = SpamDetector()
    assert d.predict("Free entry! Text WIN to 80086 to claim your prize now")["label"] == "SPAM"
    assert d.predict("Ok, see you at home later")["label"] == "HAM"
