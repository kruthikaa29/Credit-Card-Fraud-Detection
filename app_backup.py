from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

app = Flask(__name__)

# Load dataset
df = pd.read_csv("creditcard.csv")

# Separate input and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Credit Card Fraud Detection</title>

    <style>
        body {
            font-family: Arial;
            background: #f2f5f9;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 850px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }

        h1 {
            text-align: center;
            color: #1f2937;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        input {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 7px;
            width: 95%;
        }

        button {
            width: 100%;
            margin-top: 20px;
            padding: 14px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 17px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            background: #f8fafc;
        }

        .fraud {
            color: #dc2626;
            font-size: 25px;
            font-weight: bold;
        }

        .normal {
            color: #16a34a;
            font-size: 25px;
            font-weight: bold;
        }

        .high {
            color: #dc2626;
            font-weight: bold;
        }

        .medium {
            color: #ca8a04;
            font-weight: bold;
        }

        .low {
            color: #16a34a;
            font-weight: bold;
        }

        li {
            margin: 7px;
        }
    </style>
</head>

<body>

<div class="container">

<h1>💳 AI-Powered Credit Card Fraud Detection</h1>

<p class="subtitle">
Real-Time Detection + Risk Scoring + Explainable AI
</p>

<form method="POST">

<div class="grid">

<input type="number" step="any" name="Time" placeholder="Time" required>

<input type="number" step="any" name="Amount" placeholder="Amount" required>

{% for i in range(1,29) %}
<input type="number" step="any" name="V{{i}}" placeholder="V{{i}}" required>
{% endfor %}

</div>

<button type="submit">🔍 Analyze Transaction</button>

</form>

{% if result %}

<div class="result">

<h2>Prediction Result</h2>

<p>
Prediction:
{% if prediction == "FRAUD" %}
<span class="fraud">🚨 FRAUD</span>
{% else %}
<span class="normal">✅ NORMAL</span>
{% endif %}
</p>

<p>
Risk Score:
<strong>{{ risk_score }}/100</strong>
</p>

<p>
Risk Level:
<strong class="{{ risk_class }}">{{ risk_level }}</strong>
</p>

<h3>🔍 Why did the AI make this prediction?</h3>

<ul>
{% for feature, value in explanations %}
<li>
<strong>{{ feature }}</strong> contributed
<strong>{{ value }}</strong>
towards the fraud prediction.
</li>
{% endfor %}
</ul>

</div>

{% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = False
    prediction = ""
    risk_score = 0
    risk_level = ""
    risk_class = ""
    explanations = []

    if request.method == "POST":

        values = []

        # Keep same order as training dataset
        for column in X.columns:
            values.append(float(request.form[column]))

        transaction = pd.DataFrame([values], columns=X.columns)

        # Prediction
        prediction_value = model.predict(transaction)[0]

        # Fraud probability
        probability = model.predict_proba(transaction)[0][1]

        risk_score = round(probability * 100, 2)

        if risk_score >= 70:
            risk_level = "HIGH 🔴"
            risk_class = "high"
        elif risk_score >= 30:
            risk_level = "MEDIUM 🟡"
            risk_class = "medium"
        else:
            risk_level = "LOW 🟢"
            risk_class = "low"

        if prediction_value == 1:
            prediction = "FRAUD"
        else:
            prediction = "NORMAL"

        # SHAP explanation
        shap_result = explainer.shap_values(transaction)

        shap_array = np.array(shap_result)

        if shap_array.ndim == 3:
            shap_fraud = shap_array[0, :, 1]
        else:
            shap_fraud = shap_array[0]

        # Get strongest feature contributions
        feature_values = list(zip(X.columns, shap_fraud))

        feature_values.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanations = [
            (feature, round(value, 4))
            for feature, value in feature_values[:5]
        ]

        result = True

    return render_template_string(
        HTML,
        result=result,
        prediction=prediction,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_class=risk_class,
        explanations=explanations
    )


if __name__ == "__main__":
    app.run(debug=True)