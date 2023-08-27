import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load your trained model (already tuned)
lr_tuned = LogisticRegression(C=10, penalty='l2')

# Load your preprocessed scaler
mms = StandardScaler()
mms.mean_ = np.array([62.51020995, 64.93851365, 2291.81820616])
mms.scale_ = np.array([30.13503065, 30.22733403, 2272.36442647])

# Define the input components for Gradio
inputs = [
    gr.inputs.Slider(minimum=0, maximum=100, label="Tenure"),
    gr.inputs.Slider(minimum=0, maximum=100, label="Monthly Charges"),
    gr.inputs.Slider(minimum=0, maximum=5000, label="Total Charges"),
]

# Define the output components for Gradio
outputs = gr.outputs.Label()

# Define the model function
def churn_prediction(tenure, monthly_charges, total_charges):
    # Preprocess the input features
    input_features = np.array([[tenure, monthly_charges, total_charges]])
    input_features_scaled = mms.transform(input_features)
    
    # Make predictions using the model
    prediction = lr_tuned.predict(input_features_scaled)
    
    return "Churn Predicted" if prediction[0] == 1 else "No Churn Predicted"

# Create a Gradio interface
iface = gr.Interface(fn=churn_prediction, inputs=inputs, outputs=outputs)

# Launch the interface
iface.launch()