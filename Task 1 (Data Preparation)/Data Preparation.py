import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
file_path = "churn.csv"  # Use file path
data = pd.read_csv(file_path)

# Step 2: Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Step 3: Encode categorical variables
# Label encode binary columns
label_encoder = LabelEncoder()
data['International plan'] = label_encoder.fit_transform(data['International plan'])
data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])

# One-hot encode the 'State' column
data = pd.get_dummies(data, columns=['State'], drop_first=True)

# Step 4: Convert 'Churn' to numeric
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# Step 5: Display processed dataset and structure
print("\nDataset Info After Preprocessing:\n")
print(data.info())
print("\nFirst 5 Rows of Processed Data:\n")
print(data.head())
