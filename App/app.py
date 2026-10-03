from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import sys  # Move this up here

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return send_from_directory("Static", "ML.html")
    
# Load the trained model
with open("LightGBM.pkl", "rb") as file:
    model = joblib.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract and process input
        gender = 1 if data["Gender"] == "Male" else 0
        married = 1 if data["Married"] == "Yes" else 0
        education = 0 if data["Education"] == "Graduate" else 1
        self_employed = 1 if data["Self_Employed"] == "Yes" else 0
        credit_history = int(data["Credit_History"])
        applicant_income = float(data["ApplicantIncome"])
        coapplicant_income = float(data["CoapplicantIncome"])
        loan_amount = float(data["LoanAmount"])
        loan_term = float(data["Loan_Amount_Term"])

        # One-hot encode Dependents
        dependents = data["Dependents"]
        dependents_encoded = [0, 0, 0, 0]
        if dependents == "0":
            dependents_encoded[0] = 1
        elif dependents == "1":
            dependents_encoded[1] = 1
        elif dependents == "2":
            dependents_encoded[2] = 1
        else:  # "3+"
            dependents_encoded[3] = 1

        # Correct feature order
        features = np.array([
            loan_amount, applicant_income, coapplicant_income, loan_term, credit_history,
            gender, married, *dependents_encoded, education, self_employed
        ]).reshape(1, -1)

        print("Features sent to model:", features, file=sys.stdout, flush=True)

        prediction = model.predict(features)[0]
        print("Model prediction:", prediction, file=sys.stdout, flush=True)

        result = "Approved" if prediction == 1 else "Rejected"
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", e, file=sys.stderr, flush=True)
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
