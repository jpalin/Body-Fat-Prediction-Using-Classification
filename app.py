import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

# -------------------------------
# App Initialization
# -------------------------------

app = Flask(__name__)

# Load the pre-trained ML model from disk
model = pickle.load(open('model.pkl', 'rb'))


# -------------------------------
# Routes
# -------------------------------

@app.route('/', methods=["GET"])
def home():
    """
    Home route that renders the main page with the input form.
    """
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    """
    Route to handle form submission from the front-end and return prediction result.
    Expects input values from form, performs prediction, and renders the result on the HTML page.
    """
    try:
        # Extract and preprocess input features from form
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        # Make prediction and get associated probability
        prediction = model.predict(final_features)
        prediction_proba = model.predict_proba(final_features)

        output_label = prediction[0].capitalize()
        output_confidence = round(np.max(prediction_proba) * 100, 2)

        # Render prediction to frontend
        return render_template(
            'index.html',
            prediction_text=f'Body fat percentage in a healthy range?: {output_label} with a {output_confidence}% probability'
        )

    except Exception as e:
        # Handle unexpected errors and return a generic error message
        return render_template(
            'index.html',
            prediction_text=f"An error occurred during prediction: {str(e)}"
        )


@app.route('/results', methods=['POST'])
def results():
    """
    API endpoint to handle JSON input for external requests (e.g., from JavaScript or another app).
    Returns prediction result in JSON format.
    """
    try:
        # Parse JSON input and convert to array
        data = request.get_json(force=True)
        features = [np.array(list(data.values()))]

        # Perform prediction
        prediction = model.predict(features)
        output = prediction[0]

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# App Runner
# -------------------------------

if __name__ == "__main__":
    # Starts the Flask development server
    app.run(port=5000, debug=True)
