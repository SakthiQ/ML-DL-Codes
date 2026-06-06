# Customer Churn Prediction (Random Forest)

This project demonstrates a binary classification approach to predict whether a customer will churn using Random Forest.

## Implementation Details

- **Dataset**: `data.csv` (Synthetic customer data)
- **Preprocessing**: Label Encoding for categorical features, Standard Scaling for numerical features.
- **Model**: Scikit-learn's `RandomForestClassifier`.
- **Metrics**: Accuracy, Precision, Recall, and F1-Score.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Run inference:
   ```bash
   python predict.py
   ```
