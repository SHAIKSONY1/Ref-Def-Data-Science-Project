# Customer Churn Prediction using Ensemble Machine Learning

This project analyzes customer churn using the IBM Telco Customer Churn dataset. A series of preprocessing steps, model building, evaluation, and interpretability techniques are applied to develop and assess predictive models using Python and Scikit-learn.

## Dataset
- Source: IBM Telco Customer Churn Dataset  
- Link: [IBM Dataset on GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data)

The dataset includes 7,043 customer records and 21 features describing demographics, services, billing, and churn behavior.

## Objective
To build and compare several machine learning models for predicting customer churn and identifying the key drivers behind it using SHAP interpretability.

## Key Features

### Preprocessing:
- Missing value handling for TotalCharges
- Feature engineering: tenure_category, charges_ratio, service_count
- Encoding: binary, ordinal, and one-hot
- Dropped irrelevant customerID column

### Models Used:
- Logistic Regression (baseline and optimized)
- Random Forest (baseline and optimized)
- XGBoost (baseline and optimized)

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, MCC, Cohen's Kappa
- McNemar’s test for statistical comparison

### Interpretability:
- SHAP values for model explanation
- Feature importance visualization (XGBoost)

### Deployment:
- Streamlit-ready code is provided with:
  - best_model.pkl
  - scaler.pkl
  - feature_names.csv
  - model_metadata.csv

## How to Run

1. Clone the repository:
   git clone https://github.com/SHAIKSONY1/Ref-Def-Data-Science-Project.git
   cd churn-prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Run the notebook:
   jupyter notebook Sony_Rework_code_final.ipynb

4. (Optional) Run Streamlit app after saving models:
   streamlit run app.py

## Ethics & Data Usage
- Dataset is public and licensed for educational use.
- Ethical considerations include anonymized data, GDPR compliance, and University of Hertfordshire research ethics.
- This work does not involve personally identifiable or sensitive information.

## File Structure
.
├── Sony_Rework_code_final.ipynb
├── requirements.txt
├── README.md
├── app.py (optional Streamlit interface)
├── saved_models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.csv
│   └── model_metadata.csv

## Acknowledgements
- IBM for dataset
- Scikit-learn, XGBoost, SHAP, and related libraries
- University of Hertfordshire MSc Data Science Programme
