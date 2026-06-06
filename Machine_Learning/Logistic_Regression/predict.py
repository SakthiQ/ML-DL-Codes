import pandas as pd
import numpy as np
import pickle

# 7. Model loading
def load_model_artifacts(file_path):
    with open(file_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['scaler'], artifacts['encoders']

# 8. Prediction example
def predict_churn(model, scaler, encoders, input_data):
    df = pd.DataFrame([input_data])
    
    # Preprocess input using same encoders and scaler
    df['Contract'] = encoders['Contract'].transform(df['Contract'])
    df['PaymentMethod'] = encoders['PaymentMethod'].transform(df['PaymentMethod'])
    
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    
    return prediction[0], probability[0][1]

if __name__ == "__main__":
    model, scaler, encoders = load_model_artifacts('model.pkl')
    
    # Sample prediction
    sample_input = {
        'Age': 34,
        'Tenure': 12,
        'MonthlyCharges': 65.5,
        'TotalCharges': 780.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check'
    }
    
    prediction, prob = predict_churn(model, scaler, encoders, sample_input)
    print(f"Prediction for Sample Input: {'Churn' if prediction == 1 else 'No Churn'}")
    print(f"Churn Probability: {prob:.4f}")
