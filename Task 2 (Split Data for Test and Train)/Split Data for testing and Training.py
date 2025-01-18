# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# Replace 'churn.csv' with the actual path to your dataset
data = pd.read_csv('churn.csv')

# Display dataset info
print("Dataset Information:")
print(data.info())

# Display the first few rows of the dataset
print("\nDataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# Fill missing values (if any)
data.fillna(method='ffill', inplace=True)

# Handle categorical columns
# Encoding binary categorical columns using LabelEncoder
label_encoder = LabelEncoder()
data['International plan'] = label_encoder.fit_transform(data['International plan'])
data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# If other categorical columns (e.g., 'State') exist, use One-Hot Encoding
if 'State' in data.columns:
    data = pd.get_dummies(data, columns=['State'], drop_first=True)

# Check for class distribution before balancing
print("\nClass Distribution Before Balancing:")
print(data['Churn'].value_counts(normalize=True))

# Visualize the class distribution
sns.countplot(x='Churn', data=data)
plt.title("Class Distribution Before Balancing")
plt.show()

# Split the data into features (X) and target (y)
X = data.drop('Churn', axis=1)  # Drop the target column
y = data['Churn']  # Target column

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the new class distribution after SMOTE
print("\nClass Distribution After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Visualize the new class distribution
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After SMOTE")
plt.show()

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance visualization (optional)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()
