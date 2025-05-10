import os
import joblib
import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle

app = Flask(__name__)

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'mental_health_model.pkl')

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
print("Model path:", model_path)

# Verify the file exists before trying to load it
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    # for rendering results on index.html
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    output = prediction[0]

    return render_template('results.html', prediction_text = 'Should you seek treatment? $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
