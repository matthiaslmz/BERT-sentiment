from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch


app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
saved_model_path = "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/checkpoints/checkpoint-11000/"
model = BertForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)
model.eval()

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
    return outputs[0][0]

@app.route("/predict")

def predict():
    txt = request.args.get("sentence")
    neg_pred = text_pred(txt, model=model, tokenizer=tokenizer)
    pos_pred = 1 - neg_pred
    return jsonify({'sentence': txt, 'positive': str(pos_pred), 'negative': str(neg_pred)})

if __name__ == "__main__":
    app.run()

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
    # outputs = torch.sigmoid(outputs[0]).cpu().detach().numpy(
    # outputs[0]
    # outputs[0][0]


