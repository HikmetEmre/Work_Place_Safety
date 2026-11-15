"""Train a Fatal / Nonfatal classifier on the OSHA accident reports.

Rebuilds a CountVectorizer + MultinomialNB pipeline from ``model_df.csv``
(text is already cleaned and stemmed). Output files default to *_retrained.pkl
so the shipped ``cv1_model.pkl`` / ``mnb11_model.pkl`` are never overwritten.

Usage:  python train_model.py [--data model_df.csv] [--out-dir .]

NOTE: the vectorizer settings below are sensible defaults, not the settings
from the original notebook (which is not in this repo).
"""
import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path):
    """Read the modelling data and drop empty rows."""
    df = pd.read_csv(path, index_col=0)
    return df.dropna(subset=["text", "level"])


def train(df):
    """Fit vectorizer and classifier; return both plus the held-out split."""
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["level"],
        test_size=TEST_SIZE,
        stratify=df["level"],
        random_state=RANDOM_STATE,
    )
    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    classifier = MultinomialNB()
    classifier.fit(x_train_vec, y_train)

    cv_scores = cross_val_score(classifier, x_train_vec, y_train, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    return vectorizer, classifier, x_test, y_test


def evaluate(vectorizer, classifier, x_test, y_test):
    """Print a classification report on the held-out test set."""
    predictions = classifier.predict(vectorizer.transform(x_test))
    print(classification_report(y_test, predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="model_df.csv")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} reports:\n{df['level'].value_counts()}\n")

    vectorizer, classifier, x_test, y_test = train(df)
    evaluate(vectorizer, classifier, x_test, y_test)

    out_dir = Path(args.out_dir)
    dump(vectorizer, out_dir / "cv1_retrained.pkl")
    dump(classifier, out_dir / "mnb11_retrained.pkl")
    print(f"Saved models to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
