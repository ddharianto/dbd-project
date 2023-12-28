import streamlit as st
from streamlit_option_menu import option_menu

#navigas sidebar
with st.sidebar :
    selected = option_menu('Prediksi Data Pasien Penyakit DBD',
    ['Pertanyaan Gejala Pasien', 'Prediksi Pasien DBD'], 
    default_index=0)


st.header("Model Prediksi DBD Berbasis - ELM")

st.subheader("Masukan Data Pasien")

st.divider()
#Halaman Pertanyaan
if(selected == 'Pertanyaan Gejala Pasien'):
    st.text("Berikan Tanda Ceklis jika anda mengalami gejala Penyakit DBD")

    f_keluhan_utama = st.text_input("Keluhan Utama Pasien...")
    f_mual = st.checkbox("Apakah pasien mengalami gejala mual")
    f_panas = st.checkbox("Apakah pasien mengalami gejala panas")
    f_muntah = st.checkbox("Apakah pasien mengalami gejala muntah")
    f_perutsakit = st.checkbox("Apakah pasien mengalami gejala perut sakit")
    f_pusing = st.checkbox("Apakah pasien mengalami gejala pusing")
    f_batuk = st.checkbox("Apakah pasien mengalami gejala batuk")
    f_pilek = st.checkbox("Apakah pasien mengalami gejala pilek")
    tombol = st.button("Menu Selanjutnya")
    


if(selected == 'Prediksi Pasien DBD'):
    f_nadi = st.number_input("Masukan Berapa Nilai Nadi Pasien", min_value=0, max_value=300)
    f_suhu = st.slider("Masukan Berapa Nilai Suhu Pasien", min_value=20, max_value=45)
    f_hemoglobin = st.slider("Masukan Berapa Nilai Hemoglobin Pasien", min_value=0, max_value=20)
    f_lekosit = st.number_input("Masukkan Berapa Nilai Leukosit Pasien", min_value=0, max_value=20000)
    f_hematrokit = st.slider("Masukkan Berapa Nilai Hematrokit Pasien", min_value=0, max_value=60)
    f_trombosit = st.number_input("Masukkan Berapa Nilai Trombosit Badan Pasien", min_value=0, max_value=150000)

# Load Model ELM

# Button Prediksi

    if st.button("Prediksi"):
        # feature
        all_feature = [f_nadi, f_suhu, f_hemoglobin, f_lekosit, f_hematrokit, f_trombosit]
        hasil_prediksi = "Positif DBD"
        st.success(f"Hasil Prediksi Anda adalah: {hasil_prediksi}")
    else:
        st.error("Negatif DBD")

