# from elmmodel import elm
import numpy as np
import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df_dbd = pd.read_csv('./dataset/new_df_final.csv')

# print(df_dbd)

# df_dbd["Class"] = np.nan
# for x in range(100):
#   if x > 79 :
#     df_dbd.loc[x, 'Class'] = int(0)
#   else:
#     df_dbd.loc[x, 'Class'] = int(1)

# # print(df_dbd['Class'].isnull().sum())
# df_dbd['Class'] = df_dbd['Class'].astype(int)
# df_dbd = df_dbd.drop(['Diagnosa Masuk', 'Diagnosis Keluar'], axis=1)
# df_dbd.to_csv('./dataset/new_df_final.csv', index=False)

# print(df_dbd)

df_dbd['Keluhan Utama'] = df_dbd['Keluhan Utama'].apply(eval)
df_dbd['Riwayat Penyakit Sekarang'] = df_dbd['Riwayat Penyakit Sekarang'].apply(eval)

df_dbd['Keluhan Utama'] = np.array(df_dbd['Keluhan Utama'])
df_dbd['Riwayat Penyakit Sekarang'] = np.array(df_dbd['Riwayat Penyakit Sekarang'])

df_dbd['Keluhan Utama'] = df_dbd['Keluhan Utama'].apply(lambda x: [float(i) for i in x])
df_dbd['Riwayat Penyakit Sekarang'] = df_dbd['Riwayat Penyakit Sekarang'].apply(lambda x: [float(i) for i in x])

list_lengths = df_dbd['Keluhan Utama'].apply(len)

print(len(df_dbd['Keluhan Utama'].loc[0]))

# Extract TF-IDF features into new columns
max_features = 100  # Replace with the maximum length of your TF-IDF vectors
for i in range(max_features):
    df_dbd[f'KU{i+1}'] = 0.0  # Create new columns

# Populate the new columns with TF-IDFvalues
for index, row in df_dbd.iterrows():
    tfidf_values = row['Keluhan Utama']  # Extract TF-IDF vector
    for i, value in enumerate(tfidf_values[:max_features]):  # Iterate through vector
        df_dbd.at[index, f'KU{i+1}'] = value  # Assign value to respective column

# Drop the original 'Keluhan Utama' column as it's no longer needed
df_dbd.drop(columns=['Keluhan Utama'], inplace=True)

# print(df_dbd['Keluhan Utama'].apply(type).unique())
# print(df_dbd['Keluhan Utama'].values)
# # print(df_dbd['Riwayat Penyakit Sekarang'].loc[[0]].dtypes)
# print(df_dbd[['Keluhan Utama', 'Riwayat Penyakit Sekarang']])

# first_row_list = df_dbd.loc[4, 'Riwayat Penyakit Sekarang']

# # Check the data types of elements in the list
# for element in first_row_list:
#     print(type(element))

y = df_dbd[["Class"]].values
X = df_dbd.drop(['Class', 'Riwayat Penyakit Sekarang'], axis=1).values
# X = df_dbd.drop(['Class', 'Keluhan Utama', 'Riwayat Penyakit Sekarang'], axis=1).values

# Convert lists in X to numerical arrays
X_numeric = np.array([np.array(xi) if isinstance(xi, list) else xi for xi in X])

# Check for non-numeric elements in X_numeric
for xi in X_numeric:
    if not all(isinstance(el, (int, float)) for el in xi):
        print("Non-numeric element detected:", xi)

# Binarize labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# print("X shape:", X.shape)
# print("X sample:", X[0])  # Print a sample row of X

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Model Parameters
input_size = X.shape[1]  # Number of features
hidden_size = 100  # Number of neurons in the hidden layer
output_size = 2  # For binary classification (0 and 1)

# Generate random weights and biases for input layer to hidden layer
input_weights = np.random.normal(size=(input_size, hidden_size))
hidden_bias = np.random.normal(size=(hidden_size,))

print("input_weights shape:", input_weights.shape)
print("input_weights sample:", input_weights[0])  # Print a sample row of input_weights


# Calculate hidden layer output using ReLU activation function
hidden_layer_output = np.dot(X_train, input_weights) + hidden_bias
hidden_layer_output = np.maximum(hidden_layer_output, 0)

# Calculate output weights
output_weights = np.dot(np.linalg.pinv(hidden_layer_output), y_train)

# Predict using the trained ELM
test_hidden_layer_output = np.dot(X_test, input_weights) + hidden_bias
test_hidden_layer_output = np.maximum(test_hidden_layer_output, 0)
predictions = np.dot(test_hidden_layer_output, output_weights)

# Convert predictions to classes using a threshold (e.g., 0.5)
threshold = 0.5
predicted_classes = (predictions >= threshold).astype(int)
true_classes = y_test

# Calculate evaluation metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='binary')
recall = recall_score(true_classes, predicted_classes, average='binary')
f1 = f1_score(true_classes, predicted_classes, average='binary')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)