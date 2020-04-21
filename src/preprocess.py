import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer

def preprocess_imdb(dataframe):
    dataframe.review = dataframe.review.str.lower()
    dataframe.review = dataframe.review.replace(regex=[r'<[^<]+?>', r'\[[^]]*\]'], value="")
    dataframe.review = dataframe.review.replace(regex=[r'[^a-zA-Z]', r'\s+'], value=" ")
    dataframe.review = dataframe.review.str.strip().str.lower()
    dataframe.sentiment = dataframe.sentiment.map({'positive': 1, 'negative': 0})

    return dataframe


class IMDBDataset:
    def __init__(self, review, label):
        self.review = review
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained("../models/bert-base-uncased", do_lower_case=True)

    def __len__(self):
        return len(self.review)

    def _getitem__(self, item):
        review = str(self.review[item])

        encoded_inputs = tokenizer.encode_plus(review, add_special_tokens=True, max_length=512, pad_to_max_length=True, return_tensors='pt')
        input_tensor = encoded_inputs['input_ids']
        attention_tensor = encoded_inputs['attention_mask']
        label_tensor = torch.LongTensor(self.label[item])

    return None