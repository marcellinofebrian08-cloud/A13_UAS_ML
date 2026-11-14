import os
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model_cnn.keras"
LABEL_PATH = "label_kelas.json"
CONTOH_DIR = "contoh_uji"
TINGGI, LEBAR = 64, 512
BATAS_YAKIN = 0.30

st.set_page_config(page_title="Identifikasi Penulis Tulisan Tangan", page_icon="✍️", layout="wide")

st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #1F3A93, #4A7BE0);
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 16px;
}
.banner .judul { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
.banner .sub { color: #DCE6F5; font-size: 1rem; margin: 4px 0 0 0; }
[data-testid="stMetric"] {
    background-color: #EEF3FB;
    border-left: 4px solid #1F3A93;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def muat_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def muat_label():
    with open(LABEL_PATH) as f:
        return json.load(f)


def preprocessing(gambar):
    abu = gambar.convert("L").resize((LEBAR, TINGGI), Image.BILINEAR)
    return np.array(abu, dtype=np.uint8)


def prediksi(model, citra):
    x = citra.astype("float32")[np.newaxis, ..., np.newaxis]
    return model.predict(x, verbose=0)[0]


with st.sidebar:
    st.header("Tentang aplikasi")
    st.write(
        "Aplikasi ini menebak **siapa penulis** sebuah potongan kalimat tulisan tangan "
        "dari 50 mahasiswa pada dataset MLA."
    )
    st.caption("Model: CNN Custom (4 blok konvolusi)")
    st.caption("Input: citra grayscale 512 × 64 piksel")
    st.divider()
    st.subheader("Dataset MLA")
    st.markdown(
        "| Bagian | Jumlah citra |\n"
        "|---|---|\n"
        "| Data latih | 1.825 |\n"
        "| Data validasi | 323 |\n"
        "| Data uji | 537 |\n"
        "| **Total** | **2.685** |"
    )
    st.caption("80% latih dan 20% uji, lalu 15% data latih dipakai sebagai validasi.")
    st.divider()
    st.subheader("Cara pakai")
    st.markdown(
        "1. Unggah gambar atau pilih contoh citra uji.\n"
        "2. Periksa citra asli dan hasil preprocessing.\n"
        "3. Klik **Identifikasi penulis**."
    )

st.markdown("""
<div class="banner">
    <p class="judul">✍️ Identifikasi Penulis Tulisan Tangan</p>
    <p class="sub">Unggah potongan kalimat tulisan tangan (hasil scan), lalu model akan menebak penulisnya.</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    a, b, c, d = st.columns(4)
    a.metric("Jumlah penulis", "50")
    b.metric("Total citra", "2.685")
    c.metric("Akurasi rata-rata", "94,53%")
    d.metric("Ukuran input", "512 × 64")

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error(
        f"File `{MODEL_PATH}` atau `{LABEL_PATH}` tidak ditemukan. "
        "Letakkan kedua file hasil training di folder yang sama dengan app.py."
    )
    st.stop()

model = muat_model()
kelas = muat_label()

with st.container(border=True):
    st.subheader("1. Pilih gambar")
    sumber = st.radio("Sumber gambar", ["Unggah gambar", "Pilih contoh citra uji"], horizontal=True)

    gambar, nama_file = None, None
    if sumber == "Unggah gambar":
        file = st.file_uploader("Pilih file gambar (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])
        if file is not None:
            gambar, nama_file = Image.open(file), file.name
    else:
        daftar = sorted(os.listdir(CONTOH_DIR)) if os.path.isdir(CONTOH_DIR) else []
        if daftar:
            nama_file = st.selectbox("Pilih contoh citra uji", daftar)
            gambar = Image.open(os.path.join(CONTOH_DIR, nama_file))
        else:
            st.warning(f"Folder `{CONTOH_DIR}` kosong atau tidak ditemukan.")

if gambar is None:
    st.info("Unggah gambar atau pilih contoh citra uji untuk memulai.")
    st.stop()

citra = preprocessing(gambar)

with st.container(border=True):
    st.subheader("2. Periksa citra")
    kiri, kanan = st.columns(2)
    with kiri:
        st.image(gambar, caption=f"Citra asli: {nama_file} ({gambar.size[0]} × {gambar.size[1]} piksel)", width="stretch")
    with kanan:
        st.image(citra, caption="Setelah preprocessing: grayscale 512 × 64 piksel", width="stretch")

    tombol = st.button("🔍 Identifikasi penulis", type="primary", width="stretch")

if tombol:
    t0 = time.perf_counter()
    prob = prediksi(model, citra)
    waktu = (time.perf_counter() - t0) * 1000

    urutan = np.argsort(prob)[::-1]
    terbaik = urutan[0]
    penulis, yakin = kelas[terbaik], prob[terbaik] * 100

    with st.container(border=True):
        st.subheader("3. Hasil identifikasi")

        a, b, c = st.columns(3)
        a.metric("Penulis yang diprediksi", penulis)
        b.metric("Tingkat keyakinan", f"{yakin:.2f}%")
        c.metric("Waktu prediksi", f"{waktu:.0f} ms")
        st.progress(float(prob[terbaik]))

        if prob[terbaik] < BATAS_YAKIN:
            st.warning(
                f"Keyakinan model di bawah {BATAS_YAKIN:.0%}. Gambar mungkin bukan milik salah satu "
                "dari 50 penulis di dataset, atau kualitas scan kurang jelas."
            )

        label_asli = nama_file.split("_")[0]
        if label_asli in kelas:
            if label_asli == penulis:
                st.success(f"Prediksi benar. Label sebenarnya: {label_asli}")
            else:
                st.error(f"Prediksi salah. Label sebenarnya: {label_asli}")

        st.markdown("**5 kandidat penulis teratas**")
        top5 = pd.DataFrame({
            "Penulis": [kelas[i] for i in urutan[:5]],
            "Peluang (%)": [round(prob[i] * 100, 2) for i in urutan[:5]],
        })
        st.dataframe(
            top5,
            hide_index=True,
            width="stretch",
            column_config={
                "Peluang (%)": st.column_config.ProgressColumn(
                    "Peluang (%)", format="%.2f%%", min_value=0, max_value=100
                ),
            },
        )
