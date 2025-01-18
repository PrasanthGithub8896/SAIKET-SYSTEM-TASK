import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv('churn.csv')

# Preprocess the data
data['International plan'] = LabelEncoder().fit_transform(data['International plan'])
data['Voice mail plan'] = LabelEncoder().fit_transform(data['Voice mail plan'])
data['State'] = LabelEncoder().fit_transform(data['State'])

# Feature Selection based on domain knowledge and dataset
selected_features = ['Account length', 'Total day minutes', 'Total day charge', 
                     'Total eve minutes', 'Total eve charge', 'Total intl minutes', 
                     'Total intl charge', 'Customer service calls']

# Extract the selected features and target variable
X_selected = data[selected_features]
y = data['Churn']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Model: Random Forest Classifier to calculate feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.Series(model.feature_importances_, index=X_selected.columns).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance for Churn Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Print feature importance values
print("Feature Importances:")
print(feature_importance)

# Perform Correlation Analysis
correlation_matrix = X_selected.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.title('Correlation Matrix')
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

# Additional step: Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_train, y_train)

# Print RFE feature ranking
print("RFE Feature Ranking:")
for i in range(X_rfe.shape[1]):
    print(f"Feature: {X_train.columns[rfe.support_][i]}, Rank: {rfe.ranking_[i]}")

# Final selected features after RFE
selected_rfe_features = X_train.columns[rfe.support_]
print("Selected Features after RFE:")
print(selected_rfe_features)
