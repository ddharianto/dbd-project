import streamlit as st

st.header("Model Prediksi DBD Berbasis - ELM")

st.subheader("Masukan Fitur")

st.divider()

st.text("Penggunana.....")

f_nadi = st.slider("Masukan Nilai Nadi Pasien", min_value=0, max_value=300)
f_suhu = st.slider("Masukan Nilai Suhu", min_value=20, max_value=45)
f_hemoglobin = st.slider("Masukan Nilai Hemoglobin", min_value=0, max_value=20)

# Load Model ELM

# Button Prediksi

if st.button("Prediksi"):
    # feature
    all_feature = [f_nadi, f_suhu, f_hemoglobin]
    hasil_prediksi = "Positif DBD"
    st.success(f"Hasil Prediksi Anda adalah: {hasil_prediksi}")