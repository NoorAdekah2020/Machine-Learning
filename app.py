
import pandas as pd
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Cluster SRQ-20", layout="centered")
st.title("🧠 Aplikasi Prediksi Cluster Berdasarkan SRQ-20")

srq_questions = [
    "Apakah anda sering mengalami sakit kepala?",
    "Apakah Anda selalu kurang nafsu makan?",
    "Apakah tidur anda kurang nyenyak?",
    "Apakah anda merasa takut?",
    "Apakah tangan Anda gemetaran?",
    "Apakah anda merasa gugup, tegang, atau khawatir?",
    "Apakah pencernaan Anda kurang baik?",
    "Apakah Anda merasa kesulitan untuk berpikir secara jernih?",
    "Apakah Anda merasa kurang bahagia?",
    "Apakah Anda menangis lebih sering?",
    "Apakah Anda sukar menikmati apa yang Anda lakukan sehari-hari?",
    "Apakah Anda mengalami kesulitan dalam mengambil keputusan?",
    "Apakah pekerjaan sehari-hari terasa sebagai beban yang menyulitkan?",
    "Apakah Anda tidak dapat berguna dalam kehidupan sehari-hari?",
    "Apakah Anda kehilangan minat terhadap berbagai hal?",
    "Apakah Anda merasa sebagai orang yang tidak berharga?",
    "Apakah Anda pernah berpikir mengenai bunuh diri?",
    "Apakah Anda merasa lelah sepanjang waktu?",
    "Apakah Anda mempunyai keluhan tidak nyaman pada bagian perut?",
    "Apakah Anda mudah merasa lelah?"
]

# Input pengguna
user_input = []
st.subheader("📝 Silakan Jawab 20 Pertanyaan Berikut:")
for i, question in enumerate(srq_questions):
    jawaban = st.radio(f"{i+1}. {question}", ["Tidak", "Iya"], horizontal=True, key=str(i))
    nilai = 1 if jawaban == "Iya" else 0
    user_input.append(nilai)

# Tombol prediksi
if st.button("🔍 Prediksi"):
    try:
        input_array = np.array(user_input).reshape(1, -1)
        total_ya = int(sum(user_input))

        # Load model clustering
        model = joblib.load("/content/drive/MyDrive/Machine Learning/kmeans_model.pkl")
        cluster_pred = model.predict(input_array)[0]

        # Mapping hasil cluster berdasarkan analisis di notebook
        cluster_gejala = 1  # cluster dengan rata-rata TOTAL YA tinggi
        cluster_normal = 0  # cluster dengan rata-rata TOTAL YA rendah

        # Tampilkan skor total YA
        st.success(f"✅ TOTAL YA kamu: {total_ya} dari 20 pertanyaan")

        # Interpretasi berdasarkan cutoff WHO
        if total_ya < 6:
            st.info("📘 Berdasarkan cutoff WHO, kamu tergolong **tidak menunjukkan gejala psikologis yang signifikan**.")
        else:
            st.warning("⚠️ Berdasarkan cutoff WHO, terdapat **indikasi adanya gejala psikologis**.")

        # Interpretasi berdasarkan hasil cluster (untuk orang awam)
        if cluster_pred == cluster_normal:
            st.success("🔮 Berdasarkan pola jawaban kamu, sistem mengelompokkan kamu ke dalam kategori **gejala ringan atau normal**.")
        elif cluster_pred == cluster_gejala:
            st.warning("🔮 Berdasarkan pola jawaban kamu, sistem mengelompokkan kamu ke dalam kategori **kemungkinan memiliki gejala psikologis lebih tinggi**.")
        else:
            st.info("ℹ️ Hasil cluster tidak dapat dikenali.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses prediksi: {e}")
