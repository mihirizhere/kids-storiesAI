import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

MODEL_PATH = "classifier.joblib"

def train_and_save_classifier(texts, labels):
    """
    Offline: build your TF-IDF + Naive Bayes model and save it to disk.

    Args:
        texts (list[str]): Example user prompts.
        labels (list[str]): Corresponding theme keys.
    """
    # 1) Vectorize (uni- & bi-grams)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(texts)

    # 2) Train the Naive Bayes model
    clf = MultinomialNB()
    clf.fit(X, labels)

    # 3) Persist the pipeline
    joblib.dump((vectorizer, clf), MODEL_PATH)
    print(f"Saved classifier to {MODEL_PATH}")


def load_classifier():
    """
    Loads (vectorizer, model) tuple from disk.

    Raises:
        FileNotFoundError: If the model file isn't present.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Please run train_classifier.py first."
        )
    return joblib.load(MODEL_PATH)


def classify_topic_ml(user_prompt: str) -> str:
    """
    Predicts one of your theme keys:
      'fantasy_adventure', 'moral_quest',
      'number_journey', 'lullaby_rhyme'

    Returns:
        str: The predicted theme key.
    """
    # Lazy-load on call to avoid import-time errors
    vectorizer, clf = load_classifier()
    X = vectorizer.transform([user_prompt])
    return clf.predict(X)[0]
