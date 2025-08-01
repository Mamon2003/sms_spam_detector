from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print("Model/vectorizer load error:", e)

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
        try:
            input_sms = request.form.get('message', '')
            print("Form Data:", input_sms)

            if input_sms.strip() == "":
                result = "No input provided."
            else:
                transformed_sms = transform_text(input_sms)
                print("Transformed Text:", transformed_sms)
                vector_input = tfidf.transform([transformed_sms])
                prediction = model.predict(vector_input)[0]
                result = 'Spam' if prediction == 1 else 'Not Spam'
                print("Prediction:", result)
        except Exception as e:
            result = f"Error: {e}"
            print("Prediction Error:", e)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')