import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df_dbd = pd.read_csv('./dataset/df_final.csv')

# print(df_dbd)

y = df_dbd[['Diagnosis DBD']].values
X = df_dbd.drop(['Diagnosis DBD'], axis=1).values

# Binarize labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# print("X shape:", X.shape)
# print("X sample:", X[0])  # Print a sample row of X

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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