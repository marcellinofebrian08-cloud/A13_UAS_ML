# Aplikasi Identifikasi Penulis Tulisan Tangan

Aplikasi web sederhana (Streamlit) untuk menebak penulis sebuah potongan kalimat tulisan tangan dari 50 mahasiswa dataset MLA. Model yang dipakai adalah **CNN Custom**, metode terbaik dari hasil eksperimen tugas (akurasi rata-rata 94,53%).

## Isi folder

| File | Fungsi |
|---|---|
| `train_model_cnn.ipynb` | Dijalankan di Google Colab untuk melatih dan menyimpan model |
| `app.py` | Kode aplikasi (antarmuka + prediksi) |
| `requirements.txt` | Daftar library yang perlu di-install |
| `.streamlit/config.toml` | Pengaturan warna tampilan aplikasi |

Setelah training, tambahkan isi `hasil_model.zip` ke folder ini:
`model_cnn.keras`, `label_kelas.json`, dan folder `contoh_uji/`.

## Langkah singkat

1. Jalankan `train_model_cnn.ipynb` di Colab (GPU T4) sampai selesai.
2. Download `hasil_model.zip` dari Google Drive, ekstrak isinya ke folder ini.
3. Di terminal folder ini:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Browser akan terbuka otomatis di `http://localhost:8501`.

Untuk menjalankan lagi di lain waktu cukup:
```
.venv\Scripts\activate
streamlit run app.py
```
