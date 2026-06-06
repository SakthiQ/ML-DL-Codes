import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Dataset loading
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Data preprocessing
def preprocess_data(df):
    # Encoding categorical variables
    le_contract = LabelEncoder()
    df['Contract'] = le_contract.fit_transform(df['Contract'])
    
    le_payment = LabelEncoder()
    df['PaymentMethod'] = le_payment.fit_transform(df['PaymentMethod'])
    
    # Drop CustomerID as it's not a feature
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    return X, y, le_contract, le_payment

# 3. Train-test split
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model training
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluation metrics
    y_pred = model.predict(X_test_scaled)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model, scaler

# 6. Model saving
def save_model(model, scaler, encoders, file_path):
    artifacts = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders
    }
    with open(file_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Model and artifacts saved to {file_path}")

if __name__ == "__main__":
    df = load_data('data.csv')
    X, y, le_contract, le_payment = preprocess_data(df)
    model, scaler = train_model(X, y)
    
    encoders = {
        'Contract': le_contract,
        'PaymentMethod': le_payment
    }
    
    save_model(model, scaler, encoders, 'model.pkl')
