# Main entry point for the project

from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string

#Flask app - starting point of our api

app = Flask(__name__)

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y) # preprocessed text

def predict_spam(message):

    # preprocess the message
    transformed_sms = transform_text(message)

    # Vectorize the processed message
    vector_input = tfidf.transform([transformed_sms]).toarray()

    # Predict using ML model
    result = model.predict(vector_input)[0]

    return result

@app.route('/')  #homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  #predict route
def predict():


    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html', result=result)

if __name__ == '__main__':
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    app.run(host='0.0.0.0')

# localhost ip address = 0.0.0.0
