import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load Dataset
data = pd.read_csv('churn.csv')

# Preprocess Data
data['International plan'] = LabelEncoder().fit_transform(data['International plan'])
data['Voice mail plan'] = LabelEncoder().fit_transform(data['Voice mail plan'])
data['State'] = LabelEncoder().fit_transform(data['State'])

# Selected Features and Target Variable
selected_features = ['Account length', 'Total day minutes', 'Total day charge', 
                     'Total eve minutes', 'Total eve charge', 'Total intl minutes', 
                     'Total intl charge', 'Customer service calls']
X = data[selected_features]
y = data['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and Evaluate Models
for model_name, model in models.items():
    print(f"\n--- {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
