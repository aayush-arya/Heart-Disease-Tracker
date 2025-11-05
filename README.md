# Heart Disease Tracker: Full-Stack Predictive Dashboard

## 🚀 Project Overview
This project is an end-to-end, full-stack application that predicts the presence of heart disease (cardiac risk) in a patient based on 13 clinical input parameters. 

It demonstrates the entire Machine Learning lifecycle: from data preparation and model optimization to **model serialization** and deployment as a **RESTful API** with a dynamic web frontend.

---

## ✨ Key Technical Features

### **Model & Backend**
* **Model:** Optimized **Logistic Regression Classifier**.
* **Data:** **Cleveland Heart Disease Dataset (13 Features)**, chosen for its strong correlation metrics compared to the original Framingham data.
* **Pipeline:** Implements a robust **Scikit-learn Pipeline** (`Imputer` $\rightarrow$ `StandardScaler` $\rightarrow$ `LogisticRegression`) to prevent **data leakage** and ensure consistent data transformation between training and prediction.
* **Optimization:** Uses **`class_weight='balanced'`** to handle the dataset's target imbalance, prioritizing accurate detection of the minority (positive) class.
* **Serialization:** The entire final pipeline is serialized using **Joblib** (`final_heart_predictor.joblib`) for efficient loading by the API.
* **API:** Built using **Flask** and hosted on **Gunicorn**, providing a reliable `/predict` endpoint.

### **Frontend & Deployment**
* **Frontend:** A dynamic, high-contrast, mobile-responsive web interface built with pure **HTML/CSS (Tailwind CSS)** and JavaScript.
* **Visualization:** Features a real-time, **simulated ECG waveform** implemented using HTML `<canvas>` and JavaScript for dynamic visual appeal.
* **Deployment:** Fully deployed on **Render** (via Git/GitHub) using a custom build process to ensure the model is trained and ready upon server startup.

---

## 💻 Running the Project Locally

To run the application on your local machine (recommended for development), follow these steps:

### **1. Setup and Installation**

1.  Clone this repository.
2.  Navigate to the project directory.
3.  Install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### **2. Training and Serialization**

You must run the training script once to create the model file (`final_heart_predictor.joblib`).

```bash
python heart_disease_tracker.py
