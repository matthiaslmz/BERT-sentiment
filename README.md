# BERT/XLNet Sentiment Classification with Flask

1. To use fine-tune BERT for sentiment classification with HuggingFace's [BERTForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) on the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/) and then to serve the fine-tuned BERT model using Flask. 

2. To experiment, understand and compare XLNet with BERT. 

# Dataset
The motivation to using this dataset because it's already quite clean and has balanced classes: 50% positive and 50% negative reviews. 

# Web application
![review_page](https://raw.githubusercontent.com/matthiaslmz/BERT-sentiment/master/results/review.png)

![results_page](https://raw.githubusercontent.com/matthiaslmz/BERT-sentiment/master/results/results.png)

![thankyou_page](https://raw.githubusercontent.com/matthiaslmz/BERT-sentiment/master/results/feedback.png)


# XLNet Fine-tuning
Using a single P100 GPU, these are the parameters that are used in order to fine-tune XLNet.

In `train.py`:
        train_dataset = IMDBXLNet(review=df_train.review.values, 
                              label=df_train.sentiment.values,
                              max_len=768)

        valid_dataset = IMDBXLNet(review=df_valid.review.values, 
                              label=df_valid.sentiment.values,
                              max_len=768)

        model = XLNetSentiment(train_dset=train_dataset, 
                               eval_dset=valid_dataset, 
                               train_batch_size = 6,
                               eval_batch_size = 6)