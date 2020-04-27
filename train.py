import pandas as pd
import numpy as np
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from src.model import BERTSentiment
from src.preprocess import preprocess_imdb, split_train_valid, IMDBDataset

logging.basicConfig(level=logging.INFO)

def main():

    df = pd.read_csv("C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/imdb_dataset.csv")
    df = preprocess_imdb(df)
    df_train, df_valid = split_train_valid(dataframe=df, test_size=0.1, random_state=420)

    train_dataset = IMDBDataset(review=df_train.review.values, label=df_train.sentiment.values)
    valid_dataset = IMDBDataset(review=df_valid.review.values, label=df_valid.sentiment.values)

    model = BERTSentiment(train_dset=train_dataset, eval_dset=valid_dataset)
    model.train_model(ckpt_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/checkpoints/checkpoint-%04d",
                    output_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERT-sentiment/out/checkpoint-%04d")

if __name__ == "__main__":
    main()
