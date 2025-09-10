from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)
    result = "REAL" if prediction[0] == 1 else "FAKE"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
