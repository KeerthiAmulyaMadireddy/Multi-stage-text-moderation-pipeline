"""
train_model.py
--------------
Training a simple text moderation model using the Jigsaw dataset
and save it as `moderation_model.pkl` for production use.

Model:
    - TF-IDF (1–2 grams)
    - One-vs-Rest Logistic Regression (multi-label)

Labels:
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_DIR = "data"

TARGET_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_data():
    """Load train and validation splits from prepared CSV files."""
    train = pd.read_csv(f"{DATA_DIR}/train.csv").dropna(subset=["text"])
    val   = pd.read_csv(f"{DATA_DIR}/val.csv").dropna(subset=["text"])

    X_train = train["text"].tolist()
    y_train = train[TARGET_COLS].values

    X_val   = val["text"].tolist()
    y_val   = val[TARGET_COLS].values

    return X_train, y_train, X_val, y_val


def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()

    # Define pipeline: TF-IDF + Logistic Regression (One-vs-Rest)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=3,
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced"
            )
        )),
    ])

    print(" Training model...")
    model.fit(X_train, y_train)

    print(" Evaluating on validation set...")
    y_pred = (model.predict_proba(X_val) >= 0.5).astype(int)
    print(classification_report(y_val, y_pred, target_names=TARGET_COLS, zero_division=0))

    # Save model for production
    print("Saving model to moderation_model.pkl ...")
    joblib.dump(model, "moderation_model.pkl")
    print("Done. File: moderation_model.pkl")


if __name__ == "__main__":
    main()
