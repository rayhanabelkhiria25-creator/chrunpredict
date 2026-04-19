
#  Telco Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer is likely to churn, built with Streamlit.

## Overview
This project covers the full ML pipeline — from exploratory data analysis to model training and an interactive prediction interface.

##  Features
- Exploratory Data Analysis dashboard
- Model performance comparison
- Predict churn for a single customer in real time

##  Model Performance
Three models were trained using 5-fold Stratified K-Fold Cross-Validation with SMOTE applied inside each fold to prevent data leakage.

| Model | ROC-AUC | Avg Precision | F1 | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.8441 | 0.6521 | 0.6225 | 0.4669 |
| Random Forest | 0.8401 | 0.6386 | 0.6264 | 0.4795 |
| XGBoost | 0.8378 | 0.6463 | 0.6285 | 0.4759 |

## How to Run
1. Clone the repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the notebook to generate model files
4. Launch the app:
```bash
   streamlit run app.py
```

##  Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- SMOTE (imbalanced-learn)
- Streamlit
##Author
rayhana .
