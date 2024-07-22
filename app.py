# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model (replace 'model.pkl' with your model file)
with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your trained model (replace 'model.pkl' with your model file)
with open('./scaler (1).pkl', 'rb') as f:
    scaler = pickle.load(f)    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    features = np.array([[cgpa, iq]])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        result = "Student got placement"
    else:
        result = "Student did not get placement"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
