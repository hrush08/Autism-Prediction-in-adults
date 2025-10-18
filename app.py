from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get all form values
        features = [
            int(request.form['A1_Score']),
            int(request.form['A2_Score']),
            int(request.form['A3_Score']),
            int(request.form['A4_Score']),
            int(request.form['A5_Score']),
            int(request.form['A6_Score']),
            int(request.form['A7_Score']),
            int(request.form['A8_Score']),
            int(request.form['A9_Score']),
            int(request.form['A10_Score']),
            int(request.form['age']),
            int(request.form['gender']),
            int(request.form['ethnicity']),
            int(request.form['jaundice']),
            int(request.form['autism']),
            int(request.form['country_of_res']),
            int(request.form['used_app_before']),
            int(request.form['relation'])
        ]
        # Reshape for prediction
        input_data = np.array(features).reshape(1, -1)
        result = model.predict(input_data)[0]
        prediction = 'Autism Present' if result == 1 else 'Autism Not Present'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
