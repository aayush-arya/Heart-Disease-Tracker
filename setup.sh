#!/bin/bash
# setup.sh: Script to run before starting the web server.

# Check if the trained model already exists. If not, train and create it.
# This prevents the Flask API from crashing on startup if the model is missing.
if [ ! -f final_heart_predictor.joblib ]; then
    echo "Model not found. Running training script..."
    # The training script (healthtracker.py) creates final_heart_predictor.joblib
    python healthtracker.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Model training failed. Check heart_disease_tracker.py for errors."
        exit 1
    fi
    echo "Model training completed successfully."
else
    echo "Model file final_heart_predictor.joblib found. Skipping training."
fi