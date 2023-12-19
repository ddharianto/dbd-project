import pandas as pd
import nltk
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
stop = stopwords.words('indonesian')
label_encoder = preprocessing.LabelEncoder()
bow = CountVectorizer()
tf_idf = TfidfVectorizer()

df = pd.read_excel('./dataset/Dataset DBD.xlsx', sheet_name=3)
df.astype('str')

df['Kesadaran'].unique()

df['Kesadaran'] = df['Kesadaran'].replace([' Compos Mentis'], 'Compos Mentis') # Menyamakan value yang tidak sesuai
df['Kesadaran'].unique()

df['Kesadaran'] = label_encoder.fit_transform(df['Kesadaran']) #Mengubah value menjadi angka
df['Kesadaran'].unique()

df['Imunisasi'].unique()

df['Imunisasi'].isnull().sum()

df['Imunisasi'].fillna(df['Imunisasi'].mode()[0], inplace=True) # mengisi value NaN dengan nilai mean
df['Imunisasi'].unique()

df['Imunisasi'] = label_encoder.fit_transform(df['Imunisasi']) #Mengubah value menjadi angka
df['Imunisasi'].unique()

df[df.columns[1:12]]

df[df.columns[1:12]] = df[df.columns[1:12]].apply(label_encoder.fit_transform) #Mengubah value menjadi angka
df[df.columns[1:12]]

df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']]

df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']] = df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']].apply(lambda x: x.astype(str).str.lower()) # mengubah menjadi lowercase
df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']] = df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']].replace('[^a-zA-Z\s]+', ' ', regex=True) # menghilangkan simbol-simbol
df['Keluhan Utama'] = df['Keluhan Utama'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
df['Riwayat Penyakit Sekarang'] = df['Riwayat Penyakit Sekarang'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']]

df_textual = df
df_textual

df_numeric = pd.read_excel("./dataset/Dataset DBD.xlsx", sheet_name=2)
df_numeric.dtypes

df_numeric['Nadi'].unique()

df_numeric['Nadi'] = df_numeric['Nadi'].replace(['  ', '0'], np.nan) # menghilangkan value 0 dan ' ' menjadi NaN
df_numeric['Nadi'].unique()

df_numeric['Nadi'] = pd.to_numeric(df_numeric['Nadi'], errors='coerce') #mengubah semua fitur 'Nadi' menjadi numeric

df_numeric.dtypes

df_numeric.isnull().sum()

df_numeric.fillna(df_numeric.median(numeric_only=True).round(1), inplace=True) # mengisi nilai NaN dengan nilai median
df_numeric.isnull().sum()

df_numeric

df_fixed = pd.concat([df_textual, df_numeric], axis=1, join="inner") # menggabungkan kedua dataframes
df_fixed

df_fixed.isnull().sum()

# df_fixed.to_csv('df_final.csv', index=0) # menyimpan dataset ke csv

# Bag of Words (BoW)

df_final = df_fixed
df_final

df_final.dtypes

X1 = bow.fit_transform(df_final['Keluhan Utama'])
X1 = X1.toarray()
X1

Y1 = bow.fit_transform(df_final['Riwayat Penyakit Sekarang'])
Y1 = Y1.toarray()
Y1

# Term Frequency–Inverse Document Frequency (TF-IDF)

X2 = tf_idf.fit_transform(df_final['Keluhan Utama'])
X2 = X2.toarray()
X2

Y2 = tf_idf.fit_transform(df_final['Riwayat Penyakit Sekarang'])
Y2 = Y2.toarray()
Y2

df_final['Keluhan Utama'] = X2.tolist()
df_final['Keluhan Utama'].values
# df_final['Riwayat Penyakit Sekarang']