import pandas as pd
import numpy as np
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, XLNetTokenizer
from src.BERT import BERTSentiment
from src.XLNet import XLNetSentiment
from src.preprocess import preprocess_imdb, split_train_valid, IMDBBERT, IMDBXLNet

logging.basicConfig(level=logging.INFO)

def run():

    df = pd.read_csv("C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/imdb_dataset.csv")
    df = preprocess_imdb(df)
    df_train, df_valid = split_train_valid(dataframe=df, test_size=0.1, random_state=420)

    # max_len=3177 because that is the length of the longest encoded sequence for the IMDB Dataset using XLNetTokenizer (based on SentencePiece)
    train_dataset = IMDBXLNet(review=df_train.review.values, 
                              label=df_train.sentiment.values,
                              max_len=3177)

    valid_dataset = IMDBXLNet(review=df_valid.review.values, 
                              label=df_valid.sentiment.values,
                              max_len=3177)

    model = XLNetSentiment(train_dset=train_dataset, eval_dset=valid_dataset)
    model.train_model(ckpt_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/XLNetcheckpoints/checkpoint-%04d",
                    output_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/XLNetout/checkpoint-%04d")

if __name__ == "__main__":
    run()