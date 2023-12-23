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

print(df['Kesadaran'].unique())

df['Kesadaran'] = df['Kesadaran'].replace([' Compos Mentis'], 'Compos Mentis') # Menyamakan value yang tidak sesuai
print(df['Kesadaran'].unique())

df['Kesadaran'] = label_encoder.fit_transform(df['Kesadaran']) #Mengubah value menjadi angka
print(df['Kesadaran'].unique())

print(df['Imunisasi'].unique())

print(df['Imunisasi'].isnull().sum())

df['Imunisasi'].fillna(df['Imunisasi'].mode()[0], inplace=True) # mengisi value NaN dengan nilai mean
print(df['Imunisasi'].unique())

df['Imunisasi'] = label_encoder.fit_transform(df['Imunisasi']) #Mengubah value menjadi angka
print(df['Imunisasi'].unique())

print(df[df.columns[1:12]])

df[df.columns[1:12]] = df[df.columns[1:12]].apply(label_encoder.fit_transform) #Mengubah value menjadi angka
print(df[df.columns[1:12]])

print(df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']])

df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']] = df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']].apply(lambda x: x.astype(str).str.lower()) # mengubah menjadi lowercase
df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']] = df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']].replace('[^a-zA-Z\s]+', ' ', regex=True) # menghilangkan simbol-simbol
df['Keluhan Utama'] = df['Keluhan Utama'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
df['Riwayat Penyakit Sekarang'] = df['Riwayat Penyakit Sekarang'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
print(df[['Keluhan Utama', 'Riwayat Penyakit Sekarang']])

print(df.dtypes)

df_textual = df

df_numeric = pd.read_excel("./dataset/Dataset DBD.xlsx", sheet_name=2)
print(df_numeric.dtypes)

print(df_numeric['Nadi'].unique())

df_numeric['Nadi'] = df_numeric['Nadi'].replace(['  ', '0'], np.nan) # menghilangkan value 0 dan ' ' menjadi NaN
print(df_numeric['Nadi'].unique())

df_numeric['Nadi'] = pd.to_numeric(df_numeric['Nadi'], errors='coerce') #mengubah semua fitur 'Nadi' menjadi numeric

print(df_numeric.dtypes)

print(df_numeric.isnull().sum())

df_numeric.fillna(df_numeric.median(numeric_only=True).round(1), inplace=True) # mengisi nilai NaN dengan nilai median
print(df_numeric.isnull().sum())

print(df_numeric)

df_fixed = pd.concat([df_textual, df_numeric], axis=1, join="inner") # menggabungkan kedua dataframes
print(df_fixed)

print(df_fixed.isnull().sum())

# df_fixed.to_csv('df_final.csv', index=0) # menyimpan dataset ke csv

df_final = df_fixed

# Bag of Words (BoW)
print(df_final.dtypes)

X1 = bow.fit_transform(df_final['Keluhan Utama'])
X1 = X1.toarray()

Y1 = bow.fit_transform(df_final['Riwayat Penyakit Sekarang'])
Y1 = Y1.toarray()

# Term Frequency–Inverse Document Frequency (TF-IDF)
X2 = tf_idf.fit_transform(df_final['Keluhan Utama'])
X2 = X2.toarray()

Y2 = tf_idf.fit_transform(df_final['Riwayat Penyakit Sekarang'])
Y2 = Y2.toarray()

df_final['Keluhan Utama'] = X2.tolist()
df_final['Riwayat Penyakit Sekarang'] = Y2.tolist()

df_final['Keluhan Utama'] = df_final['Keluhan Utama'].astype(str)
df_final['Riwayat Penyakit Sekarang'] = df_final['Riwayat Penyakit Sekarang'].astype(str)

# def convert_to_array(x):
#     if isinstance(x, list) or isinstance(x, np.ndarray):
#         return np.array(x)
#     return x

# df_final['Keluhan Utama'] = df_final['Keluhan Utama'].apply(convert_to_array)
# df_final['Riwayat Penyakit Sekarang'] = df_final['Riwayat Penyakit Sekarang'].apply(convert_to_array)

print(df_final['Keluhan Utama'].loc[[0]].values)
print(df_final.dtypes)
print(df_final)
print(df_final['Riwayat Penyakit Sekarang'].apply(type).unique())
print(df_final['Keluhan Utama'].apply(type).unique())


df_final.to_csv('./dataset/df_final.csv', index=0) # menyimpan dataset ke csv
