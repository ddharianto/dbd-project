import streamlit as st

# Function to initialize session state
def initialize_session():
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

# Function to go to the next page
def next_page():
    st.session_state.page_number += 1

def previous_page():
    st.session_state.page_number -= 1

# Streamlit app
def main():
    # Initialize session state
    initialize_session()

    st.title("Prediksi Diagnosa Penyakit Demam Berdarah Menggunakan Machine Learning")

    if st.session_state.page_number == 0:
        st.header("Page 1 Pertanyaan kepada Pasien")
        st.write("Berikan Tanda Ceklis jika anda mengalami gejala Penyakit DBD")
        f_keluhan_utama = st.text_input("Keluhan Utama Pasien...")
        f_mual = st.checkbox("Apakah pasien mengalami gejala mual")
        f_panas = st.checkbox("Apakah pasien mengalami gejala panas")
        f_muntah = st.checkbox("Apakah pasien mengalami gejala muntah")
        f_perutsakit = st.checkbox("Apakah pasien mengalami gejala perut sakit")
        f_pusing = st.checkbox("Apakah pasien mengalami gejala pusing")
        f_batuk = st.checkbox("Apakah pasien mengalami gejala batuk")
        f_pilek = st.checkbox("Apakah pasien mengalami gejala pilek")
    elif st.session_state.page_number == 1:
        st.header("Page 2 Content")
        st.write("This is the content of page 2.")
        f_nadi = st.number_input("Masukan Berapa Nilai Nadi Pasien", min_value=0, max_value=300)
        f_suhu = st.slider("Masukan Berapa Nilai Suhu Pasien", min_value=20, max_value=45)
        f_hemoglobin = st.slider("Masukan Berapa Nilai Hemoglobin Pasien", min_value=0, max_value=20)
        f_lekosit = st.number_input("Masukkan Berapa Nilai Leukosit Pasien", min_value=0, max_value=20000)
        f_hematrokit = st.slider("Masukkan Berapa Nilai Hematrokit Pasien", min_value=0, max_value=60)
        f_trombosit = st.number_input("Masukkan Berapa Nilai Trombosit Badan Pasien", min_value=0, max_value=150000)
        if st.button("Prediksi"):
        # feature
            all_feature = [f_nadi, f_suhu, f_hemoglobin, f_lekosit, f_hematrokit, f_trombosit]
            hasil_prediksi = "Positif DBD"
            st.success(f"Hasil Prediksi Anda adalah: {hasil_prediksi}")
        else:
            st.write(f"Hasil Prediksi Anda adalah Negatif DBD")

    # Button to go to the next page
    if st.button("Next Page"):
        next_page()

    if st.button("Previuos Page"):
        previous_page()

if __name__ == "__main__":
    main()
