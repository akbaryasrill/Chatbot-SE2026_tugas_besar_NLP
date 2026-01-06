Chatbot Sensus Ekonomi 2026 (SE2026) 🤖
Aplikasi Chatbot berbasis web yang dirancang untuk menjawab pertanyaan seputar Sensus Ekonomi 2026. Sistem ini menggunakan pendekatan Similarity Matching untuk mencocokkan input pengguna dengan basis pengetahuan yang ada.

🛠️ Spesifikasi Teknologi (Berdasarkan Sistem)
Sistem ini dibangun menggunakan pustaka Python berikut:

Streamlit: Framework untuk antarmuka pengguna (UI) chatbot.

Scikit-Learn: Digunakan untuk ekstraksi fitur teks (TfidfVectorizer) dan perhitungan kemiripan (cosine_similarity).

Sastrawi: Pustaka khusus untuk pengolahan teks Bahasa Indonesia (Stemming & Stopword Removal).

Pandas: Untuk pengolahan data dari format JSON ke DataFrame.

Pickle: Untuk menyimpan model hasil training agar bisa digunakan langsung oleh aplikasi.

📁 Struktur File Utama
## 📁 Struktur Proyek
- `app.py`: Interface utama chatbot (Streamlit).
- `train_model.py`: Script pengolahan data, training, dan evaluasi akurasi.
- `data/modul_se.json`: Basis pengetahuan (knowledge base).
- `models/model_se.pkl`: Model biner hasil training (Vektor & TF-IDF matrix).
- `requirements.txt`: Daftar library (Streamlit, Pandas, Scikit-learn, Sastrawi).
- `picture/latar.jpg`: Folder aset gambar yang digunakan untuk background (latar.jpg).

⚙️ Cara Menjalankan Sistem
1. Instalasi Library
Instal semua library yang tercantum dalam requirements.txt:

Bash
   pip install -r requirements.txt

2. Pelatihan & Evaluasi Model
Jalankan script training untuk menghasilkan model dan melihat akurasi sistem:

Bash
   python train_model.py


Script ini akan memecah 50 kategori utama menjadi 279 variasi pertanyaan dan menguji kemampuan temu kembali sistem.

3. Menjalankan Chatbot
Gunakan perintah berikut untuk membuka chatbot di browser:

Bash
   streamlit run app.py

🧠 Alur Pengolahan Teks (NLP Pipeline)
Setiap input (baik saat training maupun saat chat) melewati tahap berikut:

Case Folding: Mengubah teks ke huruf kecil.

Cleaning: Menghapus karakter selain huruf (simbol/angka).

Tokenizing: Pemecahan kalimat menjadi kata.

Stopword Removal: Menghapus kata umum yang tidak bermakna (menggunakan Sastrawi).

Stemming: Mengubah kata ke bentuk dasar (contoh: "pendataan" menjadi "data").

📊 Hasil Evaluasi Terakhir
Berdasarkan pengujian pada data modul_se.json:

Total Data Uji  : 281
Prediksi Benar  : 274
Akurasi Sistem  : 97.51%
Status          : Sangat Baik (Model mengenali data dengan akurat)

Metode: Cosine Similarity Retrieval.

Catatan: Pastikan folder models/ tersedia sebelum menjalankan train_model.py. Jika kamu mengubah isi modul_se.json, kamu wajib menjalankan ulang train_model.py agar perubahan tersebut dikenali oleh chatbot.