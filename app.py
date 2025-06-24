import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(script_dir, "best_model.pkl"))
scaler = joblib.load(os.path.join(script_dir, "scaler.pkl"))
results_path = os.path.join(script_dir, "model_metadata.csv")
image_path = os.path.join(script_dir, "academic_churn_analysis.png") 

try:
    expected_features = scaler.feature_names_in_
except AttributeError:
    try:
        feature_names_df = pd.read_csv(os.path.join(script_dir, "feature_names.csv"))
        expected_features = feature_names_df.iloc[:, 0].tolist()
    except Exception:
        expected_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No internet service', 'TechSupport_Yes',
            'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No internet service', 'StreamingMovies_Yes',
            'Contract', 'PaperlessBilling',
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check', 'MonthlyCharges', 'TotalCharges',
            'charges_ratio', 'service_count',
            'tenure_category_Medium', 'tenure_category_Long', 'tenure_category_Very_Long'
        ]

# Streamlit UI setup
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction System")
st.caption("MSc Data Science Project - Predictive Analytics")

# Sidebar Information
with st.sidebar:
    st.header("About This Application")
    st.write("This application predicts customer churn probability using machine learning algorithms trained on telecom customer data.")
    st.markdown("---")
    st.info("Enter customer information in the form to generate predictions.")
    st.markdown("---")
    st.write("**Model:** Logistic Regression")
    st.write("**Accuracy:** 80.5%")
    st.write("**Created by:** Sony_Shaik-23035673")

# Main input form
st.header("Customer Information Input")

with st.form("churn_prediction_form"):
    # Basic customer information
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner = st.selectbox("Has Partner", ['No', 'Yes'])
        dependents = st.selectbox("Has Dependents", ['No', 'Yes'])
        
    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.01)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=0.01)
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    
    # Service information
    st.subheader("Service Details")
    col3, col4 = st.columns(2)
    
    with col3:
        phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
        multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
        
    with col4:
        device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
        payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    # Billing information
    st.subheader("Billing Information")
    paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
    
    # Submit button
    submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

# Process prediction when form is submitted
if submitted:
    st.header("Prediction Results")
    
    try:
        # Create base input dictionary
        input_dict = {
            'gender': 1 if gender == 'Male' else 0,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract],
            'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        # Feature engineering
        input_dict['charges_ratio'] = total_charges / (tenure + 1) if tenure > 0 else total_charges
        
        # Service count calculation
        service_cols = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
        input_dict['service_count'] = sum([1 for service in service_cols if service == 'Yes'])
        
        # Tenure categorization
        if tenure <= 12:
            tenure_cat = 'Short'
        elif tenure <= 24:
            tenure_cat = 'Medium'
        elif tenure <= 48:
            tenure_cat = 'Long'
        else:
            tenure_cat = 'Very_Long'
        
        # Create DataFrame with all possible features initialized to 0
        input_df = pd.DataFrame([input_dict])
        
        # Add all one-hot encoded features (initialize to 0)
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Set the correct one-hot encoded values
        # Multiple Lines
        if multiple_lines == 'No phone service':
            input_df['MultipleLines_No phone service'] = 1
        elif multiple_lines == 'Yes':
            input_df['MultipleLines_Yes'] = 1
        
        # Internet Service
        if internet_service == 'Fiber optic':
            input_df['InternetService_Fiber optic'] = 1
        elif internet_service == 'No':
            input_df['InternetService_No'] = 1
        
        # Online Security
        if online_security == 'No internet service':
            input_df['OnlineSecurity_No internet service'] = 1
        elif online_security == 'Yes':
            input_df['OnlineSecurity_Yes'] = 1
        
        # Online Backup
        if online_backup == 'No internet service':
            input_df['OnlineBackup_No internet service'] = 1
        elif online_backup == 'Yes':
            input_df['OnlineBackup_Yes'] = 1
        
        # Device Protection
        if device_protection == 'No internet service':
            input_df['DeviceProtection_No internet service'] = 1
        elif device_protection == 'Yes':
            input_df['DeviceProtection_Yes'] = 1
        
        # Tech Support
        if tech_support == 'No internet service':
            input_df['TechSupport_No internet service'] = 1
        elif tech_support == 'Yes':
            input_df['TechSupport_Yes'] = 1
        
        # Streaming TV
        if streaming_tv == 'No internet service':
            input_df['StreamingTV_No internet service'] = 1
        elif streaming_tv == 'Yes':
            input_df['StreamingTV_Yes'] = 1
        
        # Streaming Movies
        if streaming_movies == 'No internet service':
            input_df['StreamingMovies_No internet service'] = 1
        elif streaming_movies == 'Yes':
            input_df['StreamingMovies_Yes'] = 1
        
        # Payment Method
        if payment_method == 'Credit card (automatic)':
            input_df['PaymentMethod_Credit card (automatic)'] = 1
        elif payment_method == 'Electronic check':
            input_df['PaymentMethod_Electronic check'] = 1
        elif payment_method == 'Mailed check':
            input_df['PaymentMethod_Mailed check'] = 1
        
        # Tenure categories
        if tenure_cat == 'Medium':
            input_df['tenure_category_Medium'] = 1
        elif tenure_cat == 'Long':
            input_df['tenure_category_Long'] = 1
        elif tenure_cat == 'Very_Long':
            input_df['tenure_category_Very_Long'] = 1
        
        # Ensure all expected features exist and are in the correct order
        input_df = input_df[expected_features]
        
        # Scale input data
        scaled_input = scaler.transform(input_df)
        
        # Generate prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Churn Prediction",
                value="Will Churn" if prediction == 1 else "Will Not Churn"
            )
        
        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{probability*100:.2f}%"
            )
        
        # Risk assessment
        if probability >= 0.7:
            st.error(f"HIGH RISK: Customer has a {probability*100:.2f}% probability of churning")
        elif probability >= 0.4:
            st.warning(f"MEDIUM RISK: Customer has a {probability*100:.2f}% probability of churning")
        else:
            st.success(f"LOW RISK: Customer has a {probability*100:.2f}% probability of churning")
        
        # Feature importance explanation
        with st.expander("Model Explanation (SHAP Analysis)"):
            st.write("This section shows which features contributed most to the prediction:")
            
            try:
                # Create SHAP explainer
                explainer = shap.Explainer(model, masker=scaled_input)
                shap_values = explainer(scaled_input)
                
                # Create SHAP waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig, bbox_inches="tight")
                plt.close()
                
            except Exception as e:
                st.warning("SHAP explanation could not be generated. This does not affect the prediction accuracy.")
                st.write(f"Technical details: {str(e)}")
    
    except Exception as e:
        st.error("An error occurred during prediction. Please check your inputs and try again.")
        st.write(f"Error details: {str(e)}")

# Model performance section
with st.expander("Model Performance Metrics"):
    st.subheader("Model Evaluation Results")
    
    try:
        # Load and display results
        results_df = pd.read_csv(results_path, index_col=0)
        st.dataframe(
            results_df.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True
        )
        
        # Display evaluation visualization (only if image exists)
        if os.path.exists(image_path):
            st.subheader("Model Evaluation Visualization")
            st.image(image_path, caption="Comprehensive Model Evaluation", use_container_width=True)
        
    except Exception as e:
        st.warning("Model performance metrics could not be loaded.")
        st.write(f"Technical details: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    **Technical Information:**
    - Model: Logistic Regression with feature engineering
    - Features: 47 engineered features including one-hot encoded categorical variables
    - Training Data: Telco Customer Churn Dataset
    - Preprocessing: StandardScaler normalization
    
    **Disclaimer:** This prediction is based on historical data patterns and should be used as a decision support tool.
    """
)