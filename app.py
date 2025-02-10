from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_recommendation_model.pkl')  # Ensure the model file is uploaded to Colab

@app.route('/')
def index():
    return render_template("Crop_Recommendation/templates/index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Collect input data from the form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Prepare the input data in a format that the model can understand
    input_data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temp,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict the label (crop recommendation)
    predicted_label = model.predict(input_df)

    # Return the predicted crop label to the user
    return render_template("result.html", predicted_label=predicted_label[0])

if __name__ == "__main__":
    app.run(debug=True)
