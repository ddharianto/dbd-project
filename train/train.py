import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import pickle

# Load dataset
df_dbd = pd.read_csv('./dataset/df_final.csv')

# print(df_dbd.info())

y = df_dbd[['Diagnosis DBD']].values
X = df_dbd.drop(['Diagnosis DBD'], axis=1).values

# Binarize labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# print("X shape:", X.shape)
# print("X sample:", X[0])  # Print a sample row of X

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Min-Max scaling for input features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Parameters
input_size = X.shape[1]  # Number of features
hidden_size = 2000  # Number of neurons in the hidden layer
output_size = y_train.shape[1]  # For binary classification (0 and 1)

print(hidden_size)

# Generate random weights and biases for input layer to hidden layer
input_weights = np.random.normal(size=(input_size, hidden_size))
hidden_bias = np.random.normal(size=(hidden_size,))

# # Calculate hidden layer output using ReLU activation function
# hidden_layer_output = np.dot(X_train_scaled, input_weights) + hidden_bias
# hidden_layer_output = np.maximum(hidden_layer_output, 0)

# Calculate hidden layer output using Sigmoid activation function
hidden_layer_output = np.dot(X_train_scaled, input_weights) + hidden_bias
hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_output))  # Sigmoid activation

# Calculate output weights
output_weights = np.dot(np.linalg.pinv(hidden_layer_output), y_train)

# Predict using the trained ELM
test_hidden_layer_output = np.dot(X_test_scaled, input_weights) + hidden_bias
test_hidden_layer_output = np.maximum(test_hidden_layer_output, 0)
predictions = np.dot(test_hidden_layer_output, output_weights)

# Convert predictions to classes using a threshold (e.g., 0.5)
threshold = 0.8
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
print("F-1 Score:", f1)

# # Compute ROC curve and AUC
# fpr, tpr, thresholds_roc = roc_curve(true_classes, predictions)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# Create a DataFrame with predictions, true labels, and input features for display
predictions_df = pd.DataFrame({
    'Predicted Probability': predictions.flatten(),
    'Predicted Class': predicted_classes.flatten(),
    'True Class': true_classes.flatten()
})
result_df = pd.concat([predictions_df], axis=1)

# Display the DataFrame with prediction results
# print(result_df)

model_data = {
    'input_weights': input_weights,
    'hidden_bias': hidden_bias,
    'output_weights': output_weights,
    'threshold': threshold,
    'scaler': scaler 
}

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)
