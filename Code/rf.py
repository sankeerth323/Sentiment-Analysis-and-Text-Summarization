import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train_df = pd.read_csv("training.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("validation.csv")

# Initialize TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["text"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["text"])
X_val_tfidf = tfidf_vectorizer.transform(val_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]
y_val = val_df["label"]

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

print("Random Forest model trained successfully!")

# Save model + vectorizer
joblib.dump(rf_model, "rf_model.joblib")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")

# Evaluate
y_pred = rf_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
