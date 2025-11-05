# To run this script, install required libraries: pip install pandas numpy scikit-learn joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os 

# --- Configuration ---
DATA_FILE = 'cleveland.csv'
TARGET_COLUMN = 'target'
MODEL_FILENAME = 'final_heart_predictor.joblib'

# =========================================================
# SECTION 1: Data Loading and Initial Cleaning
# =========================================================
print(f"--- 1. Loading Data from {DATA_FILE} ---")
try:
    # Note: The Cleveland dataset uses '?' for missing values, so we handle that.
    data = pd.read_csv(DATA_FILE, na_values='?')
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: '{DATA_FILE}' not found. Please ensure the file is in the same directory.")
    exit()

# Drop rows with any remaining missing values 
data.dropna(inplace=True)
print(f"Final data shape after NA removal: {data.shape}")

# Define Features (X) and Target (Y)
X = data.drop(columns=[TARGET_COLUMN])
Y = data[TARGET_COLUMN]

FEATURE_NAMES = X.columns.tolist()
print(f"\nFeatures used ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# =========================================================
# SECTION 2: Data Splitting and Pipeline Setup
# =========================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Define Preprocessing Steps
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
classifier = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')

# Create the Full Pipeline (13 Features)
final_pipeline = Pipeline(steps=[
    ('imputer', imputer), 
    ('scaler', scaler),    
    ('classifier', classifier)
])

# Train the Final Pipeline
final_pipeline.fit(X_train, Y_train)
print("\nPipeline training complete (13-feature model).")

# =========================================================
# SECTION 3: Model Evaluation and Saving
# =========================================================
print("\n--- 3. Model Evaluation ---")
Y_pred = final_pipeline.predict(X_test)
print(f"Accuracy Score: {accuracy_score(Y_test, Y_pred):.4f}")
print("\nClassification Report (Improved Metrics Expected):\n")
print(classification_report(Y_test, Y_pred, zero_division=0))

# Save the Pipeline
try:
    joblib.dump(final_pipeline, MODEL_FILENAME)
    print(f"\nSUCCESS: New model pipeline saved to '{MODEL_FILENAME}'")
except Exception as e:
    print(f"\nERROR: Failed to save the model: {e}")