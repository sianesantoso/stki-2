import streamlit as st
import re
import nltk
from nltk.tokenize import word_tokenize

# Unduh resource NLTK yang dibutuhkan
nltk.download('stopwords')
# Unduh resource 'punkt' untuk tokenisasi
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Inisialisasi NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Hilangkan karakter non-huruf
    tokens = word_tokenize(cleaned_text)  # Tokenisasi
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Muat vectorizer dan model
with open('project_akhir/vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('project_akhir/sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="IMDb Sentiment Analysis", page_icon=":movie_camera:", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #2e7d32;
        font-size: 40px;
        font-weight: bold;
    }
    .description {
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
    .warning {
        color: #d9534f;
        font-weight: bold;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        color: #333;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="title">SENTIMEN ANALISIS IMDB</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Masukkan ulasan film Anda untuk mengetahui apakah sentimennya positif atau negatif.</p>', unsafe_allow_html=True)

# Input dari pengguna
user_input = st.text_area("Tulis ulasan di sini :", placeholder="Misalnya: The movie was fantastic!", height=200)

if st.button("Prediksi", key="predict_button"):
    if user_input:
        # Preprocessing teks
        cleaned_input = preprocess_text(user_input)
        
        # Transform teks ke TF-IDF
        transformed_input = tfidf.transform([cleaned_input])
        
        # Prediksi sentimen
        prediction = model.predict(transformed_input)[0]
        sentiment = "Positif" if prediction == 1 else "Negatif"
        
        # Tampilkan hasil prediksi dengan gaya
        st.markdown(f'<p class="result">Sentimen ulasan Anda adalah: <span style="color: #4CAF50;">{sentiment}</span></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="warning">Harap masukkan teks untuk analisis.</p>', unsafe_allow_html=True)
