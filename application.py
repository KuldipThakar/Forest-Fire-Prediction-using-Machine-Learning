from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

application = Flask(__name__)
app = application

# Load Ridge regression model and Standard Scaler
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scalar = pickle.load(open('Models/scalar.pkl', 'rb'))

# Home route
@app.route("/")
def index():
    return render_template('home.html')

# Prediction route
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Get user input from the form
        Temperature = float(request.form.get('RH'))  # Corrected name from 'Temprature'
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Scale input data
        new_data_scaled = standard_scalar.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )
        result = ridge_model.predict(new_data_scaled)[0]

        return render_template('home.html', result=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
