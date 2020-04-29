import os
import numpy as np
import sqlite3
from flask import Flask, request, jsonify, render_template
from wtforms import Form, TextAreaField, validators
import torch
from transformers import BertForSequenceClassification, BertTokenizer


app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
saved_model_path = "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/checkpoints/checkpoint-11000/"
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
db_name = os.path.join(os.path.dirname(__file__), 'movie_reviews.sqlite')
model.eval()

class ReviewForm(Form):
    sentence = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])

def sqlite_insert(path, document, prediction):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, prediction))
    conn.commit()
    conn.close()

def encode_sentence(text, max_len=512):
    review = str(text)
    review = " ".join(review.split())
    encoded_inputs = tokenizer.encode_plus(review, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)
    return encoded_inputs

def text_pred(sentence, model, tokenizer):

    DEVICE = "cuda"
    model = model.to(DEVICE)

    encoded_inputs = encode_sentence(sentence, max_len=512)
    input_tensor = encoded_inputs['input_ids']
    attention_tensor = encoded_inputs['attention_mask']

    inputs = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0)
    attens = torch.tensor(attention_tensor, dtype=torch.long).unsqueeze(0)

    inputs = inputs.to(DEVICE)
    attens = attens.to(DEVICE)

    outputs = model(inputs, attention_mask=attens)
    outputs = torch.sigmoid(outputs[0]).cpu().detach().numpy()

    #results 
    neg_pred = outputs[0][0]
    pos_pred = 1- neg_pred
    predicted_class = np.argmax(outputs)

    return pos_pred, neg_pred, predicted_class

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['sentence']
        pos_pred, neg_pred, predicted_class = text_pred(review, model=model, tokenizer=tokenizer)
    return render_template('results.html',
                            content=review,
                            pos_prob=round(pos_pred*100, 2),
                            neg_prob= round(neg_pred*100, 2),
                            prediction = predicted_class)
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    if feedback == 'Incorrect':
        prediction = int(not(prediction))

    sqlite_insert(db_name, review, prediction)
    return render_template('thanks.html')


# @app.route("/predict")
# def predict():
#     txt = request.args.get("sentence")
#     neg_pred = text_pred(txt, model=model, tokenizer=tokenizer)
#     pos_pred = 1 - neg_pred
#     return jsonify({'sentence': txt, 'positive': str(pos_pred), 'negative': str(neg_pred)})

if __name__ == "__main__":
    app.run(port=8008, debug=True)

    #http://localhost:59829/predict?sentence=amazing!

    # review = str("i love this video!")
    # review = " ".join(review.split())
    # review

    # encoded_inputs = tokenizer.encode_plus(review, add_special_tokens=True, max_length=512, pad_to_max_length=True)
    # encoded_inputs

    # input_tensor = encoded_inputs['input_ids']
    # attention_tensor = encoded_inputs['attention_mask']

    # inputs = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0)
    # masks = torch.tensor(attention_tensor, dtype=torch.long).unsqueeze(0)

    # inputs = inputs.to(DEVICE)
    # masks = masks.to(DEVICE)

    # outputs = model(inputs, attention_mask=masks)
    # outputs = torch.sigmoid(outputs[0]).cpu().detach().numpy()
    # import numpy as np
    # outputs[0]
    # inv_label = {'negative': 0, 'positive': 1}
    # inv_label[np.argmax(outputs)]

    # np.argmax(outputs, 1).value

    # outputs[0][0]

    curr_dir = os.getcwd()
    db = os.path.join(curr_dir, "movie_reviews.sqlite")
    sqlite_insert(db, "testing", 0)
