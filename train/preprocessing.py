import pandas as pd
import nltk
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk_stopwords = set(stopwords.words('indonesian'))
words_excluded = {'hari', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan'}
stop = nltk_stopwords - words_excluded
label_encoder = preprocessing.LabelEncoder()
bow = CountVectorizer()
tf_idf = TfidfVectorizer()

print()

df_textual = pd.read_excel('./dataset/Dataset DBD.xlsx', sheet_name=3)
# df_textual.astype('str')

print(df_textual)

df_textual = df_textual.drop(['Diagnosis Akhir', 'Tingkat DBD'], axis=1)
print(df_textual)
print(df_textual.dtypes)

print(df_textual['Kesadaran'].unique())

df_textual['Kesadaran'] = df_textual['Kesadaran'].replace([' Compos Mentis'], 'Compos Mentis') # Menyamakan value yang tidak sesuai
print(df_textual['Kesadaran'].unique())

df_textual['Kesadaran'] = label_encoder.fit_transform(df_textual['Kesadaran']) # Mengubah value menjadi angka
print(df_textual['Kesadaran'].unique())

print(df_textual['Imunisasi'].unique())
print(df_textual['Imunisasi'].isnull().sum())

df_textual['Imunisasi'].fillna(df_textual['Imunisasi'].mode()[0], inplace=True) # Mengisi value NaN dengan nilai modus
print(df_textual['Imunisasi'].unique())

df_textual['Imunisasi'] = label_encoder.fit_transform(df_textual['Imunisasi']) # Mengubah value menjadi angka
print(df_textual['Imunisasi'].unique())

print(df_textual[df_textual.columns[1:11]])

df_textual[df_textual.columns[1:11]] = df_textual[df_textual.columns[1:11]].apply(label_encoder.fit_transform) # Mengubah value menjadi angka
print(df_textual[df_textual.columns[1:11]])
print(df_textual[df_textual.columns[1:11]].nunique())

print(df_textual['Keluhan Utama'])

df_textual['Keluhan Utama'] = df_textual['Keluhan Utama'].astype(str).str.lower() # mengubah menjadi lowercase
df_textual['Keluhan Utama'] = df_textual['Keluhan Utama'].replace('[^a-zA-Z0-9\s]+', ' ', regex=True) # menghilangkan simbol-simbol dan angka
df_textual['Keluhan Utama'] = df_textual['Keluhan Utama'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
# df_textual['Riwayat Penyakit Sekarang'] = df_textual['Riwayat Penyakit Sekarang'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) # menghilangkan stop words
print(df_textual['Keluhan Utama'])

# Term Frequency–Inverse Document Frequency (TF-IDF)
keluhan_utama = tf_idf.fit_transform(df_textual['Keluhan Utama'])
keluhan_utama = keluhan_utama.toarray()
df_textual['Keluhan Utama'] = keluhan_utama.tolist()

print(df_textual['Keluhan Utama'])

list_lengths = len(df_textual['Keluhan Utama'].loc[0])

# Extract TF-IDF features into new columns
max_features = list_lengths  # Replace with the maximum length of your TF-IDF vectors
for i in range(max_features):
    df_textual[f'KU{i+1}'] = 0.0  # Create new columns

# Populate the new columns with TF-IDFvalues
for index, row in df_textual.iterrows():
    tfidf_values = row['Keluhan Utama']  # Extract TF-IDF vector
    for i, value in enumerate(tfidf_values[:max_features]):  # Iterate through vector
        df_textual.at[index, f'KU{i+1}'] = value  # Assign value to respective column

# Drop the original 'Keluhan Utama' column as it's no longer needed
df_textual.drop(columns=['Keluhan Utama'], inplace=True)

print(df_textual['Diagnosa Masuk'].unique())

df_textual['Diagnosa Masuk'].fillna(df_textual['Diagnosa Masuk'].mode()[0], inplace=True) # Mengisi value NaN dengan nilai modus
print(df_textual['Diagnosa Masuk'].unique())

df_textual['Diagnosa Masuk'] = label_encoder.fit_transform(df_textual['Diagnosa Masuk']) # Mengubah value menjadi angka
print(df_textual['Diagnosa Masuk'].unique())

print(df_textual)
print(df_textual.dtypes)


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

df_final = pd.concat([df_textual, df_numeric], axis=1, join="inner") # menggabungkan kedua dataframes

print(df_final)
print(df_final.dtypes)

# # Bag of Words (BoW)

# X1 = bow.fit_transform(df_final['Keluhan Utama'])
# X1 = X1.toarray()

# Y1 = bow.fit_transform(df_final['Riwayat Penyakit Sekarang'])
# Y1 = Y1.toarray()

# # Term Frequency–Inverse Document Frequency (TF-IDF)
# X2 = tf_idf.fit_transform(df_final['Keluhan Utama'])
# X2 = X2.toarray()

# Y2 = tf_idf.fit_transform(df_final['Riwayat Penyakit Sekarang'])
# Y2 = Y2.toarray()

# df_final['Keluhan Utama'] = X2.tolist()
# df_final['Riwayat Penyakit Sekarang'] = Y2.tolist()

# # df_final['Keluhan Utama'] = df_final['Keluhan Utama'].astype(str)
# # df_final['Riwayat Penyakit Sekarang'] = df_final['Riwayat Penyakit Sekarang'].astype(str)

# # def convert_to_array(x):
# #     if isinstance(x, list) or isinstance(x, np.ndarray):
# #         return np.array(x)
# #     return x

# # df_final['Keluhan Utama'] = df_final['Keluhan Utama'].apply(convert_to_array)
# # df_final['Riwayat Penyakit Sekarang'] = df_final['Riwayat Penyakit Sekarang'].apply(convert_to_array)

# print(df_final['Keluhan Utama'].loc[[0]].values)
# print(df_final['Keluhan Utama'].values)
# print(X2)
# print(df_final.dtypes)
# print(df_final[['Keluhan Utama', 'Riwayat Penyakit Sekarang']])


df_final.to_csv('./dataset/df_final.csv', index=0) # menyimpan dataset ke csv
