import streamlit as st
import pandas as pd
import datetime

df = pd.read_csv("./dataset/df_final.csv")

def main(): 
    st.title("Prediksi Diagnosa Penyakit DBD Menggunakan Machine Learning")
    st.divider()
    st.write("#### Website ini adalah sebuah aplikasi yang mengimplementasikan model klasifikasi penyakit demam berdarah (DBD) menggunakan Machien Learning.")
    st.write("#### Dataset yang digunakan merupakan data rekam medik penyakit DBD yang diperoleh dari RSUD Cilegon, Banten.")
    st.write("#### Algoritma machine learning yang dipakai adalah Extreme Learning Machine.")
    st.write("#### Semoga aplikasi ini dapat membantu Anda dalam mendeteksi penyakit DBD secara dini.")
    st.link_button("Prediksi!", "http://dbd-project.streamlit.app/Prediksi_DBD", type="primary")
    st.divider()
    st.write(""":copyright: {}  
             Tim DDR3 sebagai tim peneliti dan pengembang aplikasi.""".format(datetime.datetime.now().year))

if __name__=='__main__': 
    main()