import streamlit as st
import pandas as pd
import joblib

models = {
    "Logistic Regression": joblib.load("logistic.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "SVM": joblib.load("svm.pkl"),
    "KNN": joblib.load("knn.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl")
}

scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Music Streaming Habit Prediction",
    layout="centered"
)

st.title("Music Streaming Habit Prediction")

model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

age = st.number_input("Age", min_value=0, max_value=100, value=25)

daily_listening_minutes = st.number_input(
    "Daily Listening Minutes",
    min_value=0,
    max_value=1000,
    value=120
)

songs_per_day = st.number_input(
    "Songs Per Day",
    min_value=0,
    max_value=100,
    value=20
)

playlists_count = st.number_input(
    "Playlists Count",
    min_value=0,
    max_value=100,
    value=10
)

skip_rate_pct = st.number_input(
    "Skip Rate %",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

country = st.selectbox("Country", ["INDIA", "USA", "UK", "Australia"])

platform = st.selectbox("Platform",["Spotify","Apple Music", "YouTube Music", "Amazon Music"])

top_genre = st.selectbox("Top Genre", ["Pop", "Rock", "Hip Hop", "Jazz", "Classical"])

top_artist = st.selectbox("Top Artist",["Arijit singh","Shreya ghoshal","Kishor kumar","Lata mangeshkar","Arman malik","Atif aslam","Neha kakkar"])

discover_weekly_user = st.selectbox(
    "Discover Weekly User",
    ["Yes", "No"]
)

top_mood = st.selectbox("Top Mood", ["Happy", "Sad", "Energetic", "Relaxed"])

uses_offline_mode = st.selectbox(
    "Uses Offline Mode",
    ["Yes", "No"]
)

if st.button("Predict"):

    user_data = pd.DataFrame({
        "age":[age],
        "daily_listening_minutes":[daily_listening_minutes],
        "songs_per_day":[songs_per_day],
        "playlists_count":[playlists_count],
        "skip_rate_pct":[skip_rate_pct],
        "country":[country],
        "platform":[platform],
        "top_genre":[top_genre],
        "top_artist":[top_artist],
        "discover_weekly_user":[discover_weekly_user],
        "top_mood":[top_mood],
        "uses_offline_mode":[uses_offline_mode]
    })

    user_data_encoded = pd.get_dummies(user_data)

    user_data_encoded = user_data_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    user_data_encoded = scaler.transform(user_data_encoded)

    model = models[model_name]
    prediction = model.predict(user_data_encoded)

    st.success(f"Prediction : {prediction[0]}")