import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification

def preprocess_imdb(dataframe):
    dataframe.review = dataframe.review.str.lower()
    dataframe.review = dataframe.review.replace(regex=[r'<[^<]+?>', r'\[[^]]*\]'], value="")
    dataframe.review = dataframe.review.replace(regex=[r'[^a-zA-Z]', r'\s+'], value=" ")
    dataframe.review = dataframe.review.str.strip().str.lower()
    dataframe.sentiment = dataframe.sentiment.map({'positive': 1, 'negative': 0})

    return dataframe

def split_train_valid(dataframe, test_size, random_state=420):

    train, valid = model_selection.train_test_split(dataframe,
    test_size=test_size,
    random_state=random_state,
    stratify=dataframe.sentiment.values)

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)

    return train, valid


class IMDBBERT:
    def __init__(self, review, label, max_len=512):
        self.review = review
        self.label = label
        self.max_len  = max_len
        self.tokenizer = BertTokenizer.from_pretrained("C:/Users/MatthiasL/Desktop/DATA/Desktop/DATA/ghdata/models/bert-base-uncased", do_lower_case=True)

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])

        encoded_inputs = self.tokenizer.encode_plus(review, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True)
        
        input_tensor = encoded_inputs['input_ids']
        attention_tensor = encoded_inputs['attention_mask']
        label_tensor = torch.LongTensor(self.label[item])

        return {
            'input_ids': torch.tensor(input_tensor, dtype=torch.long),
            'attention_masks': torch.tensor(attention_tensor, dtype=torch.long),
            'label_tensor': torch.tensor(self.label[item], dtype=torch.long)
        }

class IMDBXLNet:
    def __init__(self, review, label, max_len):
        self.review = review
        self.label = label
        self.max_len = max_len
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])

        encoded_inputs = self.tokenizer.encode_plus(review, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True)
        
        input_tensor = encoded_inputs['input_ids']
        attention_tensor = encoded_inputs['attention_mask']
        label_tensor = torch.LongTensor(self.label[item])

        return {
            'input_ids': torch.tensor(input_tensor, dtype=torch.long),
            'attention_masks': torch.tensor(attention_tensor, dtype=torch.long),
            'label_tensor': torch.tensor(self.label[item], dtype=torch.long)
        }
        

# if __name__ == "__main__":
    # tokenizer = BertTokenizer.from_pretrained("/misc/labshare/datasets1/speech/models/bert-base-uncased", do_lower_case=True)
    # df = pd.read_csv("/misc/DLshare/home/rpzqk242/imdb_dataset.csv")
    # df = preprocess_imdb(df)

    # encoded_inputs = tokenizer.encode_plus(df.review.values[0], add_special_tokens=True, max_length=512, 
    # pad_to_max_length=True)

    # inputs_tensor = encoded_inputs['input_ids']
    # torch.tensor(inputs_tensor, dtype=torch.long).shape
    # labels_tensor =  torch.LongTensor(df.sentiment.values[0])
    # labels_tensor.shape
    # attention_tensor = encoded_inputs['attention_mask']
    # torch.tensor(inputs_tensor, dtype=torch.long).shape
    # torch.tensor(attention_tensor, dtype=torch.long)
    # inputs_tensor

    # qwe = {
    #     'input_ids':inputs_tensor,
    #     'labels':labels_tensor,
    #     'attention_masks': attention_tensor}
    # qwe
    
    # TensorDataset(inputs_tensor, labels_tensor, attention_tensor)[0]

    # df_train, df_valid = split_train_valid(df, test_size=0.1, random_state=420)

    # train_dataset = IMDBDataset(review = df_train.review.values, label=df_train.sentiment.values)
    # valid_dataset = IMDBDataset(review = df_valid.review.values, label=df_valid.sentiment.values)

    # train_loader = DataLoader(train_dataset,batch_size=8)
    # len(train_loader)

    # for step, data in enumerate(train_loader):
    #     print(data)


# m = nn.Sigmoid()
# loss = nn.BCEWithLogitsLoss()
# inp = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# output = loss(m(inp), target)
# output

# loss1 = nn.CrossEntropyLoss()
# inp1 = torch.randn(3, 2, requires_grad=True)
# target1 = torch.empty(3,dtype=torch.long).random_(2)
# output1 = loss1(inp1,target1)
# output1

# tokenizer = BertTokenizer.from_pretrained('/misc/labshare/datasets1/speech/models/bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('/misc/labshare/datasets1/speech/models/bert-base-uncased')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# input_ids
# labels = torch.LongTensor([1,2,3]).unsqueeze(0)  # Batch size 1

# labels.size(0)
# outputs = model(input_ids, labels=labels)
# outputs
# loss, logits = outputs[:2]
# logits = logits.detach().cpu().numpy()
# logits
# pred_flat = np.argmax(logits, axis=1).flatten()
# pred_flat
# torch.argmax(logits, dim=-1)