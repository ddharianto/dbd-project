import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_textual = pd.read_excel('./dataset/Dataset DBD.xlsx', sheet_name=3)
df_numeric = pd.read_excel("./dataset/Dataset DBD.xlsx", sheet_name=2)

# Create a DataFrame from the dataset
df = pd.concat([df_textual, df_numeric], axis=1, join="inner")
df['target'] = df['Diagnosis DBD']  # Adding target column

# Basic Information about the data
print(df)  # Display first few rows
print(df.head())  # Display first few rows
print(df.info())  # Overview of columns and data types
print(df.describe())  # Summary statistics

# Check for missing values
print(df.isnull().sum())  

# Visualize distributions and relationships
sns.pairplot(df, hue='target', diag_kind='hist')  # Pairplot for feature distributions
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Explore categorical data (here, target variable)
sns.countplot(x='target', data=df)
plt.title('Count of Each Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
