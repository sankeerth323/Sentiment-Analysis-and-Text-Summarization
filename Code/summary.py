import os
import glob
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

base_path = "BBC News Summary"
articles_path = os.path.join(base_path, "News Articles")
summaries_path = os.path.join(base_path, "Summaries")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

article_files = glob.glob(os.path.join(articles_path, "*/*.txt"))

articles_data = []
for file_path in article_files:
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
            category = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)
            articles_data.append({"category": category, "filename": filename, "text": text})
    except Exception as e:
        print(f"Error reading article {file_path}: {e}")

articles_df = pd.DataFrame(articles_data)
print(f"{len(articles_df)} articles loaded.")

summary_files = glob.glob(os.path.join(summaries_path, "*/*.txt"))

summaries_data = []
for file_path in summary_files:
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            summary_text = f.read()
            category = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)
            summaries_data.append({"category": category, "filename": filename, "summary_text": summary_text})
    except Exception as e:
        print(f"Error reading summary {file_path}: {e}")

summaries_df = pd.DataFrame(summaries_data)
print(f"{len(summaries_df)} summaries loaded.")

def calculate_sentence_features(sentences):
    features = []
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence.split())
        sentence_position = (i + 1) / len(sentences) if len(sentences) > 0 else 0
        features.append({
            "sentence": sentence,
            "sentence_length": sentence_length,
            "sentence_position": sentence_position,
            "sentence_index": i
        })
    return features

articles_df["sentences"] = articles_df["text"].apply(lambda x: sent_tokenize(x) if pd.notna(x) else [])
articles_df["sentence_features"] = articles_df["sentences"].apply(calculate_sentence_features)
sentences_df = articles_df.explode("sentence_features").reset_index(drop=True)
sentences_df = pd.concat(
    [sentences_df.drop(["sentence_features"], axis=1), sentences_df["sentence_features"].apply(pd.Series)], axis=1
)

merged_df = pd.merge(sentences_df, summaries_df, on=["category", "filename"], how="left")

def is_sentence_in_summary(sentence, summary_text):
    if pd.isna(summary_text) or pd.isna(sentence):
        return False
    return sentence.strip() in summary_text.strip()

merged_df["is_summary_sentence"] = merged_df.apply(
    lambda row: is_sentence_in_summary(row["sentence"], row["summary_text"]), axis=1
)

X = merged_df[["sentence_length", "sentence_position"]]
y = merged_df["is_summary_sentence"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save model
joblib.dump(model, "extractive_summary_model.joblib")
print("Model saved as extractive_summary_model.joblib")

def generate_extractive_summary(article_text, model, top_n=3):
    sentences = sent_tokenize(article_text)
    sentence_features = calculate_sentence_features(sentences)
    features_df = pd.DataFrame(sentence_features)

    X_predict = features_df[["sentence_length", "sentence_position"]]
    sentence_scores = model.predict_proba(X_predict)[:, 1]
    features_df["score"] = sentence_scores

    sorted_sentences = features_df.sort_values(by="score", ascending=False)
    top_sentences = sorted_sentences.head(top_n).sort_values(by="sentence_index")
    summary = " ".join(top_sentences["sentence"])

    return summary

# Example usage
first_article_text = articles_df["text"].iloc[0]
generated_summary = generate_extractive_summary(first_article_text, model, top_n=3)
print("\nOriginal Article:\n", first_article_text[:500], "...")
print("\nGenerated Summary:\n", generated_summary)
