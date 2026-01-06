import streamlit as st
import pandas as pd
import json
import os
import re
import pickle
import base64  # <--- PASTIKAN ADA INI
import streamlit.components.v1 as components # <--- PASTIKAN ADA INI
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. INISIALISASI SASTRAWI (Harus sama dengan train_model.py) ---
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
factory_stop = StopWordRemoverFactory()
stopword_remover = factory_stop.create_stop_word_remover()

def text_preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    # Proses stopword dan stemming per kata
    filtered_tokens = [stopword_remover.remove(t) for t in tokens if t != '']
    stemmed_text = " ".join([stemmer.stem(t) for t in filtered_tokens])
    return stemmed_text

# --- 2. Load data ---
def load_data():
    path = os.path.join('data', 'modul_se.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset_flat = []
        for item in data:
            if isinstance(item.get('pertanyaan'), list):
                for p in item['pertanyaan']:
                    dataset_flat.append({
                        'tag': item.get('tag'),
                        'pertanyaan': p,
                        'jawaban': item.get('jawaban')
                    })
            else:
                dataset_flat.append(item)
        return pd.DataFrame(dataset_flat)
    return pd.DataFrame(columns=['tag', 'pertanyaan', 'jawaban'])

# Cek apakah model ada, jika tidak, jalankan fungsi training sederhana di sini
if not os.path.exists('models/model_se.pkl'):
    st.warning("Menyiapkan sistem untuk pertama kali, mohon tunggu...")
    # Kamu bisa memanggil fungsi dari train_model.py atau 
    # membiarkan Streamlit membaca .pkl yang sudah kamu upload ke GitHub.

# --- 3. LOGIKA BOT (SIMILARITY) ---
def get_bot_response(user_input, df_dummy):
    try:
        # Load file pkl yang dihasilkan train_model.py
        path_model = 'models/model_se.pkl'
        if not os.path.exists(path_model):
            return "Sistem sedang sibuk, pastikan model_se.pkl sudah ada di folder models."

        with open(path_model, 'rb') as f:
            data_load = pickle.load(f)
        
        vectorizer = data_load['vectorizer']
        tfidf_matrix = data_load['tfidf_matrix']
        answers = data_load['answers']

        # Preprocessing input user
        clean_input = text_preprocessing(user_input)

        # Transform ke vektor
        user_vector = vectorizer.transform([clean_input])

        # Hitung Similarity (Algoritma Similaritas)
        
        similarities = cosine_similarity(user_vector, tfidf_matrix)
        
        # Cari nilai kemiripan tertinggi
        idx_max = similarities.argmax()
        score_max = similarities[0][idx_max]

        # Threshold: Jika kemiripan > 0.1 (10%), tampilkan jawaban
        if score_max > 0.1:
            return answers[idx_max]
        else:
            return "Maaf, saya belum memahami pertanyaan itu, Silakan ajukan pertanyaan lain seputar Sensus Ekonomi 2026, Bisa diulangi?"

    except Exception as e:
        return f"Terjadi kesalahan teknis: {str(e)}"
    

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Ambil data gambar dari folder static
img_data = get_base64_image("picture/latar.jpg")

# --- UI DEPLOYMENT ---
def main():
    st.set_page_config(page_title="SE 2026", layout="centered")

    # CSS MURNI: Mengatur layout agar melambung ke atas dan scrollable
    st.markdown(f"""
    <style>
        [data-testid="stHeader"] {{ display: none !important; }}
        footer {{ visibility: hidden !important; }}
        .stApp {{ 
            background: linear-gradient(135deg, #e0f2f7, #bbdefb) !important;
        }}
        
        /* Kontainer Utama */
        .block-container {{
            max-width: 520px !important;
            padding: 0px !important;
            padding-bottom: 0px !important; /* Pasang titik koma di sini */
            margin: 0 auto !important;
            background-image: url("data:image/jpg;base64,{img_data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            height: 100vh !important; /* Kunci tinggi layar agar tidak jebol ke bawah */
            overflow: hidden !important; /* Paksa halaman diam, biar chat saja yang scroll */
            position: relative;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}

        .chat-content {{
            display: flex;
            height: 70vh;
            flex-direction: column; /* Mengurutkan dari atas ke bawah */
            height: calc(100vh - 160px); /* KUNCI: Memberikan tinggi tetap agar scroll muncul */
            overflow-y: auto !important; /* Paksa scroll aktif */
            padding: 100px 15px 20px 15px;
            gap: 10px;
            width: 100%;
            box-sizing: border-box;
        }}

        /* Header */
        .chat-header {{
            background-color: #007bff;
            color: white;
            padding: 15px 20px;
            position: fixed;
            top: 0;
            /* Gunakan ini jika left 50% meleset */
            width: 520px !important; 
            left: auto !important;
            right: auto !important;
            z-index: 1000;
            border-bottom: 4px solid #ffa500;
            box-sizing: border-box; /* Memastikan padding 20px dihitung DI DALAM 520px */
        }}

        .bubble {{
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 85%;
            font-family: 'Poppins', sans-serif;
            line-height: 1.4;
            word-wrap: break-word;
            font-size: 14px;
            color: black !important;
        }}
        .bot {{ background-color: #e3f2fd; align-self: flex-start; border-radius: 15px 15px 15px 4px; }}
        .user {{ background-color: #dcedc8; align-self: flex-end; border-radius: 15px 15px 4px 15px; }}

        /* Memperbaiki kolom input agar tidak ada celah putih */
        [data-testid="stChatInput"] {{
            position: fixed !important;
            bottom: 0 !important;
            width: 520px !important; /* Samakan dengan lebar kolom chat */
            background-color: #007bff !important;
            padding: 10px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            box-sizing: border-box !important; /* KUNCI: Biar biru penuh sampai pinggir chat */
            z-index: 1000;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Render Header
    st.markdown("""
    <div class="chat-header">
        <div style="font-size: 22px; font-weight: 800;">SE 2026</div>
        <div style="font-size: 10px;">Sensus Ekonomi Membantu Aktivitas Rakyat</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Selamat datang! Saya <b>Layanan SE 2026</b>. Silakan ketik pertanyaan seputar Sensus Ekonomi 2026."}]

    # BAGIAN PENTING: Menggabungkan semua riwayat menjadi satu blok HTML
    # Ini agar Streamlit tidak menyisipkan div sampah di antara bubble
    chat_html = '<div class="chat-content">'
    for msg in st.session_state.messages:
        cls = "bot" if msg["role"] == "assistant" else "user"
        chat_html += f'<div class="bubble {cls}">{msg["content"]}</div>'
    chat_html += '</div>'

    st.markdown(chat_html, unsafe_allow_html=True)

    # Input User
    if prompt := st.chat_input("Ketik pesan di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_bot_response(prompt, df)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    components.html(
        f"""
        <script>
        (function() {{
            var container = window.parent.document.querySelector('.chat-content');
            if (container) {{
                // Memaksa scroll ke paling bawah setiap kali ada perubahan
                container.scrollTop = container.scrollHeight;
                
                // Observasi jika ada pesan baru masuk (Auto-scroll otomatis)
                var observer = new MutationObserver(function() {{
                    container.scrollTop = container.scrollHeight;
                }});
                observer.observe(container, {{ childList: true }});
            }}
        }})();
        </script>
        """,
        height=0,
    )

if __name__ == "__main__":
    main()