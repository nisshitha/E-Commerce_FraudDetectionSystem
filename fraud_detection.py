# training_fraud_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load and clean dataset
df = pd.read_csv("ecomdataset.csv")
df.drop(columns=["Transaction.Date"], inplace=True)

X = df.drop(columns=["Is.Fraudulent"])
y = df["Is.Fraudulent"]

# Encode categorical
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Handle class imbalance
scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Train model
model = xgb.XGBClassifier(scale_pos_weight=scale, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict with tuned threshold
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > 0.7).astype(int)  # lower threshold increases recall

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save model
joblib.dump(model, "xgb_model_threshold.pkl")
joblib.dump(encoders, "encoders.pkl")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot and save ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Fraud Detection')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('xgb_roc_curve.png')  # Save as PNG
plt.close()

