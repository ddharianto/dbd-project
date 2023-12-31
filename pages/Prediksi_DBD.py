import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk_stopwords = set(stopwords.words('indonesian'))
words_excluded = {'hari'}
stop = nltk_stopwords - words_excluded

# Load the saved model and encoder
with open('trained_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

input_weights = model_data['input_weights']
hidden_bias = model_data['hidden_bias']
output_weights = model_data['output_weights']
threshold = model_data['threshold']

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
tf_idf = encoder['tf_idf']

def main(): 
    st.title("Prediksi Diagnosa Penyakit DBD Menggunakan Machine Learning")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Aplikasi Prediksi DBD </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    f_keluhan_utama = st.text_input("Keluhan Utama", help="Gunakan angka apabila memasukkan waktu. Contoh '1 hari, 5 hari, 1 minggu, dst'")
    st.text("Keluhan Tambahan")
    f_mual = st.checkbox("Mual")
    f_muntah = st.checkbox("Muntah")
    f_perutsakit = st.checkbox("Perut Sakit")
    f_pusing = st.checkbox("Pusing")
    f_batuk = st.checkbox("Batuk")
    f_pilek = st.checkbox("Pilek")
    f_tidak_nafsu_makan = st.checkbox("Tidak Nafsu Makan")
    f_mencret = st.checkbox("Mencret")
    f_imunisasi = st.selectbox("Imunisasi", ["Lengkap","Tidak Lengkap","Tidak Diimunisasi"])
    f_kesadaran = st.selectbox("Kesadaran", ['Compos Mentis', 'Delirium', 'Apatis'])
    st.text("Pemeriksaan Laboratorium")
    f_pl_gds = st.checkbox("GDS")
    f_pl_widal = st.checkbox("WIDAL")
    f_nadi = st.slider("Nadi", min_value=20, max_value=200, value=70)
    f_suhu = st.slider("Suhu Tubuh", min_value=20.0, max_value=45.0, value=37.5)
    f_hemoglobin = st.slider("Hemoglobin", min_value=0.0, max_value=20.0, value=15.0)
    f_leukosit = st.slider("Leukosit", min_value=1000, max_value=20000, value=9000)
    f_hematrokit = st.slider("Hematokrit", min_value=10.0, max_value=80.0, value=40.0)
    f_trombosit = st.slider("Trombosit", min_value=10000, max_value=450000, value=150000)
    f_diagnosa = st.selectbox("Diagnosa DBD", ['Positif', 'Negatif'])
    
    if st.button("Predict"): 
        features = [f_keluhan_utama, f_mual, f_muntah, f_perutsakit, f_pusing, f_batuk, f_pilek, 
                    f_tidak_nafsu_makan, f_mencret, f_pl_gds, f_pl_widal, f_imunisasi, f_kesadaran, 
                    f_nadi, f_suhu, f_hemoglobin, f_leukosit, f_hematrokit, f_trombosit, f_diagnosa]
        
        df=pd.DataFrame([list(features)], columns=['Keluhan Utama','KT Mual','KT Muntah','KT Perut Sakit','KT Pusing',
                                                   'KT Batuk','KT Pilek','KT Tidak Nafsu Makan','KT Mencret','PL GDS','PL WIDAL',
                                                   'Imunisasi','Kesadaran','Nadi','Suhu','Hemoglobin','Leukosit','Hematokrit','Trombosit', 'Diagnosa Masuk'])
        df = df.replace({True: 0, False: 1})
        df = df.replace({'Compos Mentis': 1, 'Delirium': 2, 'Apatis': 0})
        df = df.replace({'Lengkap': 0, 'Tidak Lengkap': 1, 'Tidak Diimunisasi': 2})
        df = df.replace({'Positif': 1, 'Negatif': 0})
        df['Keluhan Utama'] = df['Keluhan Utama'].astype(str).str.lower() # mengubah menjadi lowercase
        df['Keluhan Utama'] = df['Keluhan Utama'].replace('[^a-zA-Z0-9\s]+', ' ', regex=True) # menghilangkan simbol-simbol dan angka
        df['Keluhan Utama'] = df['Keluhan Utama'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
        KU = tf_idf.transform(df['Keluhan Utama'])
        KU = KU.toarray()
        df['Keluhan Utama'] = KU.tolist()

        list_lengths = len(df['Keluhan Utama'].loc[0])

        # Extract TF-IDF features into new columns
        max_features = list_lengths  # Replace with the maximum length of your TF-IDF vectors
        for i in range(max_features):
            df[f'KU{i+1}'] = 0.0  # Create new columns

        # Populate the new columns with TF-IDFvalues
        for index, row in df.iterrows():
            tfidf_values = row['Keluhan Utama']  # Extract TF-IDF vector
            for i, value in enumerate(tfidf_values[:max_features]):  # Iterate through vector
                df.at[index, f'KU{i+1}'] = value  # Assign value to respective column

        # # Drop the original 'Keluhan Utama' column as it's no longer needed
        df.drop(columns=['Keluhan Utama'], inplace=True)

        # print(df)
        # print(df.dtypes)
        # print(df.info())

        # Make predictions using the loaded model
        hidden_layer_output = np.dot(df.values, input_weights) + hidden_bias
        hidden_layer_output = np.maximum(hidden_layer_output, 0)
        predictions = np.dot(hidden_layer_output, output_weights)
        predicted_classes = (predictions >= threshold).astype(int)

        print("Predicted classes:", predicted_classes[0])

        output = int(predicted_classes[0])
        if output == 1:
            st.error('Anda memiliki gejala Positif DBD')
        else:
            st.success('Anda memiliki gejala Negatif DBD')


if __name__=='__main__': 
    main()