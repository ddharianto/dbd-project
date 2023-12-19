# from elmmodel import elm
import numpy as np
import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# Load dataset
df_dbd = pd.read_csv('./dataset/df_numeric.csv')
# print(df_dbd.head())
# df_dbd['Class'] = df_dbd['Class'].astype(int)
X = df_dbd.drop('Class', axis=1).values
y = df_dbd[["Class"]].values

# Binarize labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Extreme Learning Machine class
class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_weights = np.random.rand(self.input_size, self.hidden_size)
        self.bias = np.random.rand(self.hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        # Random hidden layer weights
        H = self.sigmoid(np.dot(X, self.input_weights) + self.bias)
        H_pseudo_inv = np.linalg.pinv(H)
        # Output layer weights
        self.output_weights = np.dot(H_pseudo_inv, y)

    def predict(self, X):
        H = self.sigmoid(np.dot(X, self.input_weights) + self.bias)
        predicted = np.dot(H, self.output_weights)
        return predicted

# Initialize ELM
input_size = X_train.shape[1]# Berapa banyak fitur yang digunakan
hidden_size = 50  # Change the number of hidden nodes as needed
output_size = y_train.shape[1]

elm = ELM(input_size, hidden_size, output_size)

# Train ELM
elm.train(X_train, y_train)

# Make predictions on test set
predictions = elm.predict(X_test)

# Convert predictions to classes
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print("Accuracy:", accuracy)