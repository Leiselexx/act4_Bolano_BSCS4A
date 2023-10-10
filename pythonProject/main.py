from flask import Flask, request, jsonify, render_template
import numpy as np
app = Flask(__name__)
import joblib


# Load the trained model
model = joblib.load('linearModel-2.pkl')


@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the "R&D Spend" value from the form input
        rd_spend = float(request.form['rd_spend'])

        # Create a NumPy array with the "R&D Spend" value
        input_data = np.array([rd_spend]).reshape(1, -1)

        # Make predictions using the loaded model
        predictions = model.predict(input_data)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)

