import streamlit as st
import joblib
from gtts import gTTS
import io
import pytesseract as pyt
from PIL import Image
def naive_bayes(input_text):
    nb_model = joblib.load('nb_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear"
    }

    sentence_tfidf = tfidf_vectorizer.transform([input_text])
    predicted_label = nb_model.predict(sentence_tfidf)[0]
    predicted_emotion_name = emotion_map.get(predicted_label, "unknown")

    return predicted_emotion_name

def RandomForest(input_text):
    rf_model = joblib.load("rf_model.joblib")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear"
    }

    text_tfidf = tfidf_vectorizer.transform([input_text])
    predicted_label = rf_model.predict(text_tfidf)[0]
    return emotion_map.get(predicted_label, "unknown")

def SVM(input_text):
    svm_model = joblib.load("svm_model.joblib")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear"
    }

    text_tfidf = tfidf_vectorizer.transform([input_text])
    predicted_label = svm_model.predict(text_tfidf)[0]
    return emotion_map.get(predicted_label, "unknown")

import pandas as pd
from nltk.tokenize import sent_tokenize

def extractive_summary(input_text, top_n=2):
    # Load saved model first
    model = joblib.load("extractive_summary_model.joblib")

    # This part will compute the sentence features
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

    # Tokenize
    sentences = sent_tokenize(input_text)
    sentence_features = calculate_sentence_features(sentences)
    features_df = pd.DataFrame(sentence_features)

    # Prediction of the scores
    X_predict = features_df[["sentence_length", "sentence_position"]]
    sentence_scores = model.predict_proba(X_predict)[:, 1]
    features_df["score"] = sentence_scores

    # Select top_n sentences which will come according to the prediction
    top_sentences = (
        features_df.sort_values(by="score", ascending=False)
        .head(top_n)
        .sort_values(by="sentence_index")
    )

    summary = " ".join(top_sentences["sentence"])
    return summary

def speak_summary(summary_text):
    tts = gTTS(summary_text)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    st.audio(audio_bytes.getvalue(), format="audio/mp3")

st.title(":rainbow[Text Emotion Detection and Summarization using ML]")
st.divider()
st.header(":blue[Naive Bayes model]")
input_text = st.text_input("Enter text here")
if input_text:
    st.write(f"The predicted emotion for the sentence '**:red[_{input_text}_**]' is **:green[_{naive_bayes(input_text)}_**]")
st.divider()
st.header(":blue[SVM model]")
input_text_SVM = st.text_input("Enter text")
if input_text_SVM:
    st.write(f"The predicted emotion for the sentence ':red[**_{input_text_SVM}_**]' is **:green[_{SVM(input_text_SVM)}_**]")
st.divider()
st.header(":blue[Random Forest Classifier]")
input_text_RF = st.text_input("Enter sample text")
if input_text_RF:
    st.write(f"The predicted emotion for the sentence ':red[**_{input_text_RF}_**]' is :green[**_{RandomForest(input_text_RF)}_**]")
st.divider()
st.header(":violet[Text Summarization]")
summary_text = st.text_area(
    "Enter the text to analyze"
)

st.write("Use your camera as input:")
enable = st.checkbox("Enable camera")
# picture = st.camera_input("Take a picture", disabled=not enable)

# if picture:
#     st.image(picture)
#     summary_text = pyt.image_to_string(Image.open(picture))

st.write("Or upload an image for summarizing")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    
    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    summary_text = pyt.image_to_string(Image.open(uploaded_file))

if summary_text:
    st.write("The summary for the text entered is:")
    s = extractive_summary(summary_text)
    st.write(f"**_{s}_**")
    speak_summary(s)