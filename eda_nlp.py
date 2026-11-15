"""EDA and NLP analysis of OSHA workplace accident reports.

Pipeline (mirrors the logic of the original project):
  1. Load the cleaned, stemmed report text and its label (Fatal / Nonfatal).
  2. EDA: class balance and report-length distribution.
  3. NLP: word frequencies (overall and per class), words most indicative of
     fatal reports, word clouds, and TextBlob polarity / sentiment.
  4. Model: CountVectorizer + MultinomialNB, evaluated on a held-out split.

Usage:
    python eda_nlp.py [--data model_df.csv] [--out-dir figures] [--save-models]

Raw data: https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517
(OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv).

Scope note: the raw OSHA table (firm, state, hour, body part ...) is not in this
repository, so the EDA here covers the report text and label only.
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_WORDS = 20
COLORS = {"Fatal": "#c0392b", "Nonfatal": "#2e86c1"}


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_data(path):
    """Load the modelling data and drop rows without text or label."""
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=["text", "level"]).reset_index(drop=True)
    df["n_words"] = df["text"].str.split().str.len()
    return df


def save_figure(fig, out_dir, name):
    """Save a figure as PNG and close it."""
    path = Path(out_dir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------
# 2. EDA
# --------------------------------------------------------------------------
def plot_class_distribution(df, out_dir):
    """Bar chart of Fatal vs Nonfatal report counts."""
    counts = df["level"].value_counts()
    print(f"Class distribution:\n{counts}\n")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    ax.set_title("Accident reports by damage level")
    ax.set_ylabel("Reports")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    save_figure(fig, out_dir, "01_class_distribution")


def plot_report_length(df, out_dir):
    """Histogram of report length (in words) per class."""
    print(f"Report length (words):\n{df.groupby('level')['n_words'].describe()}\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    for level, group in df.groupby("level"):
        ax.hist(group["n_words"], bins=30, alpha=0.6, label=level, color=COLORS[level])
    ax.set_title("Report length distribution")
    ax.set_xlabel("Words per report")
    ax.set_ylabel("Reports")
    ax.legend()
    save_figure(fig, out_dir, "02_report_length")


# --------------------------------------------------------------------------
# 3. NLP
# --------------------------------------------------------------------------
def word_frequencies(texts):
    """Count word occurrences across an iterable of texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    return counter


def plot_top_words(df, out_dir, top_n=TOP_N_WORDS):
    """Horizontal bar charts of the most frequent words: overall and per class."""
    groups = [("All reports", df)] + [(lvl, g) for lvl, g in df.groupby("level")]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 6))
    for ax, (title, group) in zip(axes, groups):
        words, counts = zip(*word_frequencies(group["text"]).most_common(top_n))
        ax.barh(words[::-1], counts[::-1], color=COLORS.get(title, "#7f8c8d"))
        ax.set_title(f"Top {top_n} words: {title}")
    save_figure(fig, out_dir, "03_top_words")


def fit_vectorizer_and_model(x_train, y_train):
    """Fit the bag-of-words vectorizer and the Naive Bayes classifier."""
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()
    classifier.fit(vectorizer.fit_transform(x_train), y_train)
    return vectorizer, classifier


def plot_fatal_indicative_words(vectorizer, classifier, out_dir, top_n=TOP_N_WORDS):
    """Words whose presence most shifts a report toward Fatal (or Nonfatal).

    Uses the difference of Naive Bayes per-class log probabilities.
    """
    classes = list(classifier.classes_)
    score = (
        classifier.feature_log_prob_[classes.index("Fatal")]
        - classifier.feature_log_prob_[classes.index("Nonfatal")]
    )
    scores = pd.Series(score, index=vectorizer.get_feature_names_out()).sort_values()
    top = pd.concat([scores.head(top_n), scores.tail(top_n)])
    print(f"Most Fatal-indicative words: {list(scores.tail(10).index[::-1])}\n")

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(
        top.index,
        top.values,
        color=[COLORS["Fatal"] if v > 0 else COLORS["Nonfatal"] for v in top.values],
    )
    ax.set_title("Words indicative of Fatal (right) vs Nonfatal (left) reports")
    ax.set_xlabel("log P(word | Fatal) - log P(word | Nonfatal)")
    save_figure(fig, out_dir, "04_fatal_indicative_words")


def plot_word_clouds(df, out_dir):
    """Word cloud per class (skipped if the wordcloud package is missing)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  wordcloud not installed; skipping word clouds")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, level in zip(axes, ["Fatal", "Nonfatal"]):
        text = " ".join(df.loc[df["level"] == level, "text"])
        cloud = WordCloud(width=800, height=500, background_color="white").generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"{level} reports")
        ax.axis("off")
    save_figure(fig, out_dir, "05_word_clouds")


def add_sentiment(df):
    """Add TextBlob polarity (-1..+1) and a Positive/Negative sentiment label.

    Returns df unchanged if TextBlob is not installed.
    """
    try:
        from textblob import TextBlob
    except ImportError:
        print("  textblob not installed; skipping sentiment")
        return df
    df = df.copy()
    df["polarity"] = df["text"].map(lambda t: TextBlob(t).sentiment.polarity)
    df["sentiment"] = df["polarity"].map(lambda p: "Negative" if p < 0 else "Positive")
    return df


def plot_polarity(df, out_dir):
    """Polarity histogram per class."""
    if "polarity" not in df:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for level, group in df.groupby("level"):
        ax.hist(group["polarity"], bins=30, alpha=0.6, label=level, color=COLORS[level])
    ax.set_title("TextBlob polarity by damage level")
    ax.set_xlabel("Polarity")
    ax.legend()
    save_figure(fig, out_dir, "06_polarity")


# --------------------------------------------------------------------------
# 4. Model evaluation
# --------------------------------------------------------------------------
def evaluate_model(vectorizer, classifier, x_train, y_train, x_test, y_test, out_dir):
    """Print CV accuracy and a classification report; save the confusion matrix."""
    cv_scores = cross_val_score(
        MultinomialNB(), vectorizer.transform(x_train), y_train, cv=5
    )
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n")

    predictions = classifier.predict(vectorizer.transform(x_test))
    print(classification_report(y_test, predictions))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix (test set)")
    save_figure(fig, out_dir, "07_confusion_matrix")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EDA and NLP on OSHA accident reports")
    parser.add_argument("--data", default="model_df.csv", help="CSV with text, level")
    parser.add_argument("--out-dir", default="figures", help="where to save charts")
    parser.add_argument(
        "--save-models", action="store_true",
        help="also save cv1_retrained.pkl / mnb11_retrained.pkl in --out-dir",
    )
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    print(f"Loaded {len(df)} reports\n")

    print("== EDA ==")
    plot_class_distribution(df, args.out_dir)
    plot_report_length(df, args.out_dir)

    print("\n== NLP ==")
    plot_top_words(df, args.out_dir)
    plot_word_clouds(df, args.out_dir)
    plot_polarity(add_sentiment(df), args.out_dir)

    print("\n== Model ==")
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["level"], test_size=TEST_SIZE,
        stratify=df["level"], random_state=RANDOM_STATE,
    )
    vectorizer, classifier = fit_vectorizer_and_model(x_train, y_train)
    plot_fatal_indicative_words(vectorizer, classifier, args.out_dir)
    evaluate_model(vectorizer, classifier, x_train, y_train, x_test, y_test, args.out_dir)

    if args.save_models:
        dump(vectorizer, Path(args.out_dir) / "cv1_retrained.pkl")
        dump(classifier, Path(args.out_dir) / "mnb11_retrained.pkl")
        print("Models saved.")


if __name__ == "__main__":
    main()
