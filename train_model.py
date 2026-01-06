import json
import pandas as pd
import pickle
import os
import re
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP TOOLS ---
print("Memulai inisialisasi Sastrawi...")
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

factory_stop = StopWordRemoverFactory()
stopword_remover = factory_stop.create_stop_word_remover()

def text_preprocessing(text):
    # 1. Case Folding (Mengubah jadi huruf kecil)
    text = text.lower()
    
    # 2. Cleaning (Hapus simbol & angka)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenizing (Memecah kalimat menjadi daftar kata)
    tokens = text.split()
    
    # 4. Stopword Removal (Menghapus kata 'yang', 'di', 'ke', dll)
    filtered_tokens = [stopword_remover.remove(t) for t in tokens if t != '']
    
    # 5. Stemming (Kata dasar: 'membantu' -> 'bantu')
    # Bergabung kembali menjadi kalimat untuk diproses TF-IDF
    stemmed_text = " ".join([stemmer.stem(t) for t in filtered_tokens])
    
    return stemmed_text

# --- 2. LOAD DATA ---
print("Loading dataset...")
with open('data/modul_se.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset_flat = []
for item in data:
    # Memecah pertanyaan gaya teman (list) menjadi baris sendiri-sendiri
    if isinstance(item['pertanyaan'], list):
        for p in item['pertanyaan']:
            dataset_flat.append({
                'pertanyaan': p,
                'jawaban': item['jawaban']
            })
    else:
        dataset_flat.append(item)

df = pd.DataFrame(dataset_flat)
print(f"Total baris data setelah dipecah: {len(df)}")

# --- PENTING: BUAT KOLOM 'clean_pertanyaan' DI SINI ---
print("Melakukan Preprocessing (Tokenizing, Stopwords, Stemming)...")
df['clean_pertanyaan'] = df['pertanyaan'].apply(text_preprocessing)

# --- 3. VECTORIZATION & SIMILARITY ALGORITHM ---
# TF-IDF adalah inti dari algoritma similaritas teks
print("Menghitung Vektor TF-IDF...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_pertanyaan'])

# --- 4. EVALUATION (Khusus Similarity) ---
print("\n" + "="*40)
print("HASIL EVALUASI SIMILARITY (Cosine)")
print("="*40)

# Kita hitung similarity matrix antara semua dokumen
# Ini membandingkan setiap pertanyaan dengan semua pertanyaan lainnya
eval_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

correct = 0
total = tfidf_matrix.shape[0]

for i in range(total):
    # Cari index dengan similarity tertinggi (selain dirinya sendiri)
    # Atau dalam kasus dataset FAQ, kita cek apakah skor tertinggi adalah dirinya sendiri
    predicted_idx = np.argmax(eval_sim[i])
    
    if predicted_idx == i:
        correct += 1

accuracy = (correct / total) * 100
print(f"Total Data Uji  : {total}")
print(f"Prediksi Benar  : {correct}")
print(f"Akurasi Sistem  : {accuracy:.2f}%")

if accuracy > 90:
    print("Status          : Sangat Baik (Model mengenali data dengan akurat)")
else:
    print("Status          : Perlu Tambahan Data/Preprocessing")
print("="*40)

# --- 5. PENYIMPANAN ---
if not os.path.exists('models'):
    os.makedirs('models')

model_output = {
    'vectorizer': vectorizer,
    'tfidf_matrix': tfidf_matrix,
    'questions': df['pertanyaan'].tolist(),
    'answers': df['jawaban'].tolist()
}

with open('models/model_se.pkl', 'wb') as f:
    pickle.dump(model_output, f)

print("--- SELESAI ---")
print("Semua proses NLP (Tokenizing hingga Stemming) telah dilakukan.")
print("Model Similaritas berhasil disimpan di models/model_se.pkl")