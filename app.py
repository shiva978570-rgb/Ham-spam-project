from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    # Transform input
    data = vectorizer.transform([message])
    
    # Prediction
    result = model.predict(data)[0]

    if result == 1:
        output = "SPAM 🚫"
    else:
        output = "HAM ✅"

    return render_template("index.html", prediction=output)

if __name__ == "__main__":
    app.run(debug=True)