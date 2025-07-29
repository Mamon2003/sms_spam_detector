from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
ps = PorterStemmer()

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
         y.append(ps.stem(i))

         return " ".join(y)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_sms = request.form['message']
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
