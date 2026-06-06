# Customer Churn Prediction (Logistic Regression)

This project demonstrates a binary classification approach to predict whether a customer will churn using Logistic Regression.

## Implementation Details

- **Dataset**: `data.csv` (Synthetic customer data)
- **Preprocessing**: Label Encoding for categorical features, Standard Scaling for numerical features.
- **Model**: Scikit-learn's `LogisticRegression`.
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
