import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train_df = pd.read_csv("training.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("validation.csv")

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["text"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["text"])
X_val_tfidf = tfidf_vectorizer.transform(val_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]
y_val = val_df["label"]

# Train SVM
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_tfidf, y_train)

print("SVM model trained successfully!")

# Save model and vectorizer
joblib.dump(svm_model, "svm_model.joblib")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")

# Evaluate
y_pred_svm = svm_model.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print(f"SVM Accuracy: {accuracy_svm:.4f}")
print("SVM Classification Report:")
print(report_svm)
