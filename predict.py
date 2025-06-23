import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os

# Load model and encoders
model = joblib.load("xgb_model.pkl")        # Trained model
encoders = joblib.load("encoders.pkl")      # Dict of label encoders used

# Input: Example transaction (replace with your input or load dynamically)
new_transaction = {
    'Transaction.Amount': 65,
    'Customer.Age': 22,
    'Account.Age.Days': 61,
    'Transaction.Hour': 5,
    'source': 'SEO',
    'browser': 'Chrome',
    'sex': 'F',
    'Payment.Method': 'PayPal',
    'Product.Category': 'home & garden',
    'Quantity': 3,
    'Device.Used': 'desktop',
    'Address.Match': 1
}

# Create DataFrame from input
input_df = pd.DataFrame([new_transaction])

# Encode categorical columns
for col in encoders:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

# Get probabilities and prediction
proba = model.predict_proba(input_df)[0]
threshold = 0.7
prediction = int(proba[1] > threshold)

proba = model.predict_proba(input_df)[0]
threshold = 0.7
prediction = int(proba[1] > threshold)

# Rounded values
confidence = round(proba[prediction] * 100, 2)
risk_score = round(proba[1] * 100, 2)
label = "Fraudulent Transaction" if prediction == 1 else "Safe Transaction"

# Output
print(f"Prediction: {label}")
print(f"Confidence: {confidence}%")
print(f"Risk Score: {risk_score}%")

# LIME Explanation
# Load training data and drop target + irrelevant columns
X_train = pd.read_csv("ecomdataset.csv").drop(columns=["Transaction.Date", "Is.Fraudulent"])
for col in encoders:
    if col in X_train.columns:
        X_train[col] = encoders[col].transform(X_train[col])

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Not Fraud", "Fraud"],
    mode="classification"
)

# Generate LIME explanation
exp = explainer.explain_instance(input_df.iloc[0].values, model.predict_proba, num_features=8)

# Save explanation as PNG
fig = exp.as_pyplot_figure()
plt.title("LIME Explanation")  # Custom title

# Add explanation for colors
plt.figtext(0.5, -0.05,
            "🟩 Green: Factors pushing towards fraud\n🟥 Red: Factors pushing away from fraud",
            wrap=True, horizontalalignment='center', fontsize=8)

# Ensure 'static' folder exists and save image
os.makedirs("static", exist_ok=True)
fig.tight_layout()
fig.savefig("static/lime_explanation.png", bbox_inches="tight")
print("LIME explanation saved to static/lime_explanation.png")