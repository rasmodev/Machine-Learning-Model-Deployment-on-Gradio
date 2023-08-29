
import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model and its key components
with open('rf_key_components.pkl', 'rb') as file:
    key_components = pickle.load(file)

categorical_imputer = key_components['categorical_imputer']
numerical_imputer = key_components['numerical_imputer']
encoder = key_components['encoder']
scaler = key_components['scaler']
best_model = key_components['best_model']

# Preprocessing function
def churn_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
                     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                     StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                     MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges):
    
    # Create a DataFrame from user inputs
    user_input_df = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'MonthlyCharges_TotalCharges_Ratio': [MonthlyCharges_TotalCharges_Ratio],
        'AverageMonthlyCharges': [AverageMonthlyCharges]
    })
    
    # Preprocessing for categorical data
    pred_cat_data = user_input_df.select_dtypes(include='object')
    encoded_pred_data = encoder.transform(categorical_imputer.transform(pred_cat_data))

    # Convert the encoded data to a DataFrame
    encoded_pred_data_df = pd.DataFrame.sparse.from_spmatrix(encoded_pred_data,
                                                                  columns=encoder.get_feature_names_out(pred_cat_data.columns),
                                                                  index=pred_cat_data.index)

    # Preprocessing for numerical data
    pred_num_data = user_input_df.select_dtypes(include=['int', 'float'])
    scaled_pred_data = scaler.transform(numerical_imputer.transform(pred_num_data))

    # Convert the scaled numerical data to a DataFrame
    scaled_pred_data_df = pd.DataFrame(scaled_pred_data, columns=pred_num_data.columns, index=pred_num_data.index)

    # Concatenate the encoded categorical data and scaled numerical data
    final_df = pd.concat([encoded_pred_data_df, scaled_pred_data_df], axis=1)

    # Make predictions using the loaded model
    predictions = best_model.predict(final_df)

    # Map the predictions to 'Yes' or 'No'
    input_prediction = 'Churn' if predictions[0] == 1 else 'Not Churn'

    
    return input_prediction


# Define input components
input_components = [
    gr.Radio(label='Customer Gender', choices=['Female', 'Male']),
    gr.Radio(label='Is the customer a senior citizen?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have a partner?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have dependents?', choices=['No', 'Yes']),
    gr.Number(label='Number of months the customer has been with the company.', minimum=0, maximum=72),
    gr.Radio(label='Does the customer have a phone service?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have multiple lines?', choices=['No', 'Yes']),
    gr.Radio(label='Type of internet service', choices=['DSL', 'Fiber optic', 'No']),
    gr.Radio(label='Does the customer have online security?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have online backup?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have device protection?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have tech support?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have streaming TV?', choices=['No', 'Yes']),
    gr.Radio(label='Does the customer have streaming movies?', choices=['No', 'Yes']),
    gr.Radio(label='Contract type', choices=['Month-to-month', 'One year', 'Two year']),
    gr.Radio(label='Does the customer use paperless billing?', choices=['No', 'Yes']),
    gr.Radio(label='Payment method', choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
    gr.Number(label='Monthly charges for the customer.', minimum=18, maximum=119),
    gr.Number(label='Total charges for the customer.', minimum=19, maximum=8670),
    gr.Slider(label='Ratio of monthly charges to total charges.', minimum=0.00, maximum=1.0),
    gr.Number(label='Average monthly charges for the customer.', minimum=0, maximum=120)
]

# Create and launch the Gradio interface
iface = gr.Interface(
    fn=churn_prediction,
    inputs=input_components,
    outputs="text",
    title="Customer Churn Prediction App", 
    description="This app predicts whether a customer is likely to churn (leave) a telecommunications company. It uses machine learning to analyze customer data, such as gender, age, tenure, and service usage. The app can be used by stakeholders and customers to make informed decisions about customer retention.",
    live=False,  
    share=True,
    
)

iface.launch()
