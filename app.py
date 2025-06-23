from flask import Flask, render_template, request, redirect, url_for, jsonify
import joblib
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import mysql.connector

app = Flask(__name__, static_folder='static')

# Load model and encoders
model = joblib.load("xgb_model_threshold.pkl")
encoders = joblib.load("encoders.pkl")

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Nisshi@2006",
    database="f_db"
)
cursor = db.cursor()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/form.html')
def redirect_to_form():
    return redirect(url_for('form'))

@app.route('/predict', methods=['POST'])
def predict():
    transaction = {
        'Transaction.Amount': float(request.form['t_amt']),
        'Customer.Age': int(request.form['customer_age']),
        'Account.Age.Days': int(request.form['account_age']),
        'Transaction.Hour': int(request.form['t_hour']),
        'source': request.form['source'],
        'browser': request.form['browser'],
        'sex': request.form['sex'].strip().capitalize(),
        'Payment.Method': request.form['payment_method'],
        'Product.Category': request.form['product_category'].strip().lower(),
        'Quantity': int(request.form['quantity']),
        'Device.Used': request.form['device_used'].strip().lower(),
        'Address.Match': int(request.form['address_match'])
    }

    input_df = pd.DataFrame([transaction])

    for col in encoders:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])

    proba = model.predict_proba(input_df)[0]
    threshold = 0.7
    prediction = int(proba[1] > threshold)
    confidence = round(float(proba[prediction] * 100), 2)
    risk_score = round(float(proba[1] * 100), 2)
    label = "Fraudulent Transaction" if prediction == 1 else "Safe Transaction"

    train_df = pd.read_csv("ecomdataset.csv").drop(columns=["Transaction.Date", "Is.Fraudulent"])
    for col in encoders:
        if col in train_df.columns:
            train_df[col] = encoders[col].transform(train_df[col])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_df.values,
        feature_names=train_df.columns.tolist(),
        class_names=["Not Fraud", "Fraud"],
        mode="classification"
    )

    exp = explainer.explain_instance(input_df.iloc[0].values, model.predict_proba, num_features=8)

    fig = exp.as_pyplot_figure()
    plt.title("LIME Explanation")
    plt.figtext(0.5, -0.05,
                "🟩 Green: Factors pushing towards fraud\n🔴 Red: Factors pushing away from fraud",
                wrap=True, horizontalalignment='center', fontsize=8)
    os.makedirs("static", exist_ok=True)
    fig.tight_layout()
    fig.savefig("static/lime_explanation.png", bbox_inches="tight")
    plt.close()

    query = """
    INSERT INTO transac (
        transaction_amount, customer_age, account_age_days, transaction_hour,
        source, browser, sex, payment_method, product_category, quantity,
        device_used, address_match, prediction_label, confidence, risk_score
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        float(transaction['Transaction.Amount']),
        int(transaction['Customer.Age']),
        int(transaction['Account.Age.Days']),
        int(transaction['Transaction.Hour']),
        str(transaction['source']),
        str(transaction['browser']),
        str(transaction['sex']),
        str(transaction['Payment.Method']),
        str(transaction['Product.Category']),
        int(transaction['Quantity']),
        str(transaction['Device.Used']),
        int(transaction['Address.Match']),
        str(label),
        float(confidence),
        float(risk_score)
    )

    cursor.execute(query, values)
    db.commit()

    return render_template("result.html",
                           is_fraud=prediction,
                           confidence_level=f"{confidence}%",
                           risk_score=risk_score,
                           shap_plot="lime_explanation.png")

@app.route('/dashboard-data')
def dashboard_data():
    cursor = db.cursor(dictionary=True)

    # Cards
    cursor.execute("""
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN prediction_label = 'Fraudulent Transaction' THEN 1 ELSE 0 END) as total_frauds,
            SUM(CASE WHEN prediction_label = 'Safe Transaction' THEN 1 ELSE 0 END) as total_safe
        FROM transac
    """)
    card_data = cursor.fetchone()

    # Bar: Frauds by Product Category
    cursor.execute("""
        SELECT 
            product_category,
            COUNT(*) as fraud_count
        FROM transac
        WHERE prediction_label = 'Fraudulent Transaction'
        GROUP BY product_category
        ORDER BY fraud_count DESC
        LIMIT 10
    """)
    bar_data = cursor.fetchall()

    # Pie: Frauds by Payment Method
    cursor.execute("""
        SELECT 
            payment_method,
            COUNT(*) as fraud_count
        FROM transac
        WHERE prediction_label = 'Fraudulent Transaction'
        GROUP BY payment_method
    """)
    pie_data = cursor.fetchall()

    # Donut: Transactions by Device Used
    cursor.execute("""
    SELECT 
        device_used,
        prediction_label,
        COUNT(*) as count
    FROM transac
    GROUP BY device_used, prediction_label
     """)
    donut_data = cursor.fetchall()

    # Line: Transactions by Browser (used as proxy for customer location)
    cursor.execute("""
        SELECT 
            browser,
            prediction_label,
            COUNT(*) as count
        FROM transac
        GROUP BY browser, prediction_label
    """)
    line_data = cursor.fetchall()

    return jsonify({
        'card_data': card_data,
        'bar_data': bar_data,
        'pie_data': pie_data,
        'donut_data': donut_data,
        'line_data': line_data
    })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
