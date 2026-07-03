from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("models/credit_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [

        float(request.form["laufkont"]),
        float(request.form["laufzeit"]),
        float(request.form["moral"]),
        float(request.form["verw"]),
        float(request.form["hoehe"]),
        float(request.form["sparkont"]),
        float(request.form["beszeit"]),
        float(request.form["rate"]),
        float(request.form["famges"]),
        float(request.form["buerge"]),
        float(request.form["wohnzeit"]),
        float(request.form["verm"]),
        float(request.form["alter"]),
        float(request.form["weitkred"]),
        float(request.form["wohn"]),
        float(request.form["bishkred"]),
        float(request.form["beruf"]),
        float(request.form["pers"]),
        float(request.form["telef"]),
        float(request.form["gastarb"])

    ]

    data = np.array(features).reshape(1, -1)

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
     result = "✅ Good Credit Risk"
     message = "The applicant is likely to repay the loan."

    else:
     result = "❌ Bad Credit Risk"
     message = "The applicant has a higher chance of default."

    confidence = round(probability * 100, 2)

    return render_template(
    "index.html",
    prediction=result,
    confidence=confidence,
    message=message
)

    confidence = round(probability * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)