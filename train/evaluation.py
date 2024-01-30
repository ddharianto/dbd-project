import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

# Define a list of hidden layer sizes to loop through
hidden_sizes = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

# Store accuracy values for each hidden size
accuracy_relu = []
accuracy_sigmoid = []

for hidden_size in hidden_sizes:
    # Generate random weights and biases for input layer to hidden layer
    input_weights = np.random.normal(size=(X.shape[1], hidden_size))
    hidden_bias = np.random.normal(size=(hidden_size,))

    # Calculate hidden layer output using ReLU activation function
    hidden_layer_output_relu = np.dot(X_train_scaled, input_weights) + hidden_bias
    hidden_layer_output_relu = np.maximum(hidden_layer_output_relu, 0)

    # Calculate output weights
    output_weights_relu = np.dot(np.linalg.pinv(hidden_layer_output_relu), y_train)

    # Predict using the trained ELM with ReLU activation
    test_hidden_layer_output_relu = np.dot(X_test_scaled, input_weights) + hidden_bias
    test_hidden_layer_output_relu = np.maximum(test_hidden_layer_output_relu, 0)
    predictions_relu = np.dot(test_hidden_layer_output_relu, output_weights_relu)

    # Convert predictions to classes using a threshold (e.g., 0.5)
    threshold = 0.5
    predicted_classes_relu = (predictions_relu >= threshold).astype(int)

    # Calculate accuracy and store it
    accuracy_relu.append(accuracy_score(y_test, predicted_classes_relu))

    # Calculate hidden layer output using Sigmoid activation function
    hidden_layer_output_sigmoid = np.dot(X_train_scaled, input_weights) + hidden_bias
    hidden_layer_output_sigmoid = 1 / (1 + np.exp(-hidden_layer_output_sigmoid))  # Sigmoid activation

    # Calculate output weights
    output_weights_sigmoid = np.dot(np.linalg.pinv(hidden_layer_output_sigmoid), y_train)

    # Predict using the trained ELM with Sigmoid activation
    test_hidden_layer_output_sigmoid = np.dot(X_test_scaled, input_weights) + hidden_bias
    test_hidden_layer_output_sigmoid = 1 / (1 + np.exp(-test_hidden_layer_output_sigmoid))
    predictions_sigmoid = np.dot(test_hidden_layer_output_sigmoid, output_weights_sigmoid)

    # Convert predictions to classes using a threshold (e.g., 0.5)
    predicted_classes_sigmoid = (predictions_sigmoid >= threshold).astype(int)

    # Calculate accuracy and store it
    accuracy_sigmoid.append(accuracy_score(y_test, predicted_classes_sigmoid))

df_accuracy = pd.DataFrame({
    'Hidden Size': hidden_sizes,
    'ReLU': accuracy_relu,
    'Sigmoid': accuracy_sigmoid
})
result_df = pd.concat([df_accuracy], axis=1)
print("Hidden layer: ", len(hidden_sizes))
print(result_df)

# Plotting the results
plt.plot(hidden_sizes, accuracy_relu, marker='o', label='ReLU Activation')
plt.plot(hidden_sizes, accuracy_sigmoid, marker='o', label='Sigmoid Activation')
plt.xlabel('Hidden Layer Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Sample data
accuracy = 0.9
precision = 0.94
recall = 0.94
f1score = 0.94

# Bar chart
fig, ax = plt.subplots()

bar_width = 0.35
bar_positions = [1, 2, 3, 4]

bars = ax.bar(bar_positions, [accuracy, precision, recall, f1score], bar_width, color=['blue', 'orange', 'green', 'red'])

ax.set_title('ELM model with 2000 Hidden Layer and Sigmoid Activation Function')
ax.set_xticks(bar_positions)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F-1 Score'])

# Display the values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()


