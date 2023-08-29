# Churn Prediction Web App using Gradio

## Overview

This repository hosts a web application developed with Gradio that predicts customer churn for a telecommunications company. By inputting customer information like tenure, monthly charges, and total charges, the app provides predictions about the likelihood of a customer churning.

## Usage Instructions

1. **Clone the Repository:**

    ```
    git clone https://github.com/your-username/churn-prediction-gradio.git
    ```

2. **Install Dependencies:**

    ```
    pip install gradio numpy scikit-learn
    ```

3. **Run the Gradio Interface:**

    ```
    gradio src/app.py
    ```

4. **Access the App:**

    Open a web browser and navigate to the URL displayed in the terminal (usually http://127.0.0.1:7860). The Churn Prediction Web App interface will be visible.

5. **Interact with the App:**

    Adjust the sliders for tenure, monthly charges, and total charges to observe real-time churn predictions.

## App Interface

### Before Prediction

![App Screenshot](https://github.com/rasmodev/Machine-Learning-Model-Deployment-on-Gradio/blob/main/Screenshots/Before_Pred.png)

### After Prediction
![App Screenshot](https://github.com/rasmodev/Machine-Learning-Model-Deployment-on-Gradio/blob/main/Screenshots/After_Pred.png)

## Model Details

The predictive model utilized in this app is a fine-tuned Logistic Regression model. It assesses whether a customer is likely to churn, considering input characteristics like tenure, monthly charges, and total charges. The model was trained using a processed dataset with the Scikit-learn library.

## About the Project

This project serves as a demonstration of deploying a machine learning model via Gradio. It can be used as a blueprint for developing similar predictive web applications.

## Author Information

- Your Name
- GitHub: https://github.com/rasmodev
- LinkedIn: https://www.linkedin.com/in/rasmo-/

## Licensing

This project operates under the MIT License. To understand the terms and conditions, refer to the [LICENSE](LICENSE) file.
Make sure to replace "your-username" with your actual GitHub username and provide accurate URLs for your GitHub profile and LinkedIn profile. Additionally, include the app_screenshot.png file (a screenshot of your app) in the repository, and if needed, upload the app.py file containing the Gradio code, along with any pertinent model weights or preprocessing steps.
