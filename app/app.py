from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained pipeline model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "house_price_model.pkl")
model = joblib.load(MODEL_PATH)

# Define home route
@app.route("/")
def home():
    return "🏠 House Price Prediction API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json(force=True)

        # Convert JSON to DataFrame (one row)
        input_data = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return result
        return jsonify({
            "prediction": float(prediction),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
