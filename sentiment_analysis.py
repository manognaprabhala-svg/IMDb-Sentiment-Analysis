# ============================================================
# IMDb Movie Review Sentiment Analysis
# Models:
# 1. Naive Bayes
# 2. Logistic Regression
# 3. Support Vector Machine (SVM)
# ============================================================

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\nLoading IMDb dataset...")

df = pd.read_csv("IMDB Dataset.csv")

print("\nDataset loaded successfully!")
print("Number of reviews:", len(df))
print("Columns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2: CHECK DATASET
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning reviews...")

df["cleaned_review"] = df["review"].apply(clean_text)

print("Text preprocessing completed!")


# ============================================================
# STEP 4: CONVERT SENTIMENT INTO NUMBERS
# ============================================================

# positive = 1
# negative = 0

df["sentiment_label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("\nSentiment labels:")
print(df[["sentiment", "sentiment_label"]].head())


# ============================================================
# STEP 5: SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X = df["cleaned_review"]
y = df["sentiment_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split completed!")

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# ============================================================
# STEP 6: TF-IDF
# ============================================================

print("\nApplying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed!")

print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)


# ============================================================
# STEP 7: NAIVE BAYES
# ============================================================

print("\n" + "=" * 60)
print("NAIVE BAYES")
print("=" * 60)

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_prediction = nb_model.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_prediction)

print("Naive Bayes Accuracy:",
      round(nb_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    nb_prediction,
    target_names=["Negative", "Positive"]
))


# ============================================================
# STEP 8: LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(X_train_tfidf, y_train)

lr_prediction = lr_model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print("Logistic Regression Accuracy:",
      round(lr_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    lr_prediction,
    target_names=["Negative", "Positive"]
))


# ============================================================
# STEP 9: SUPPORT VECTOR MACHINE
# ============================================================

print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

svm_model = LinearSVC()

svm_model.fit(X_train_tfidf, y_train)

svm_prediction = svm_model.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_prediction)

print("SVM Accuracy:",
      round(svm_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    svm_prediction,
    target_names=["Negative", "Positive"]
))


# ============================================================
# STEP 10: COMPARE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "SVM"
    ],
    "Accuracy": [
        nb_accuracy,
        lr_accuracy,
        svm_accuracy
    ]
})

results["Accuracy (%)"] = results["Accuracy"] * 100

print("\n")
print(results[["Model", "Accuracy (%)"]].to_string(index=False))


# ============================================================
# STEP 11: ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy (%)"]
)

plt.title("IMDb Sentiment Analysis - Model Accuracy")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

for i, value in enumerate(results["Accuracy (%)"]):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: CONFUSION MATRIX - SVM
# ============================================================

cm = confusion_matrix(y_test, svm_prediction)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 13: TEST YOUR OWN REVIEWS
# ============================================================

def predict_sentiment(review):

    cleaned = clean_text(review)

    review_tfidf = tfidf.transform([cleaned])

    prediction = svm_model.predict(review_tfidf)

    if prediction[0] == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"


print("\n" + "=" * 60)
print("TEST YOUR OWN MOVIE REVIEWS")
print("=" * 60)

test_reviews = [
    "This movie was absolutely amazing and I loved every minute of it.",
    "The movie was boring, poorly written and a complete waste of time.",
    "The acting was excellent and the story was fantastic."
]

for review in test_reviews:

    result = predict_sentiment(review)

    print("\nReview:", review)
    print("Prediction:", result)


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)