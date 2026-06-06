import pandas as pd
import numpy as np
import os

def generate_churn_data(file_path):
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 70, n_samples),
        'Tenure': np.random.randint(0, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Dataset generated at {file_path}")

if __name__ == "__main__":
    target_dir = "Machine_Learning/Logistic_Regression"
    os.makedirs(target_dir, exist_ok=True)
    generate_churn_data(os.path.join(target_dir, "data.csv"))
    
    target_dir_rf = "Machine_Learning/Random_Forest"
    os.makedirs(target_dir_rf, exist_ok=True)
    generate_churn_data(os.path.join(target_dir_rf, "data.csv"))
