# BERT-sentiment

The objective of this repo is to use fine-tune BERT for sentiment classification with HuggingFace's [BERTForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) on the [IMDB 50K reviews dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and then to serve the fine-tuned BERT model using Flask. 

# Dataset
The motivation to using this dataset because it's already quite clean and has balanced classes: 50% positive and 50% negative reviews. 

# Web application
![review_page](https://raw.githubusercontent.com/matthiaslmz/BERT-sentiment/master/results/review.png)

![results_page](https://raw.githubusercontent.com/matthiaslmz/BERT-sentiment/master/results/results.png)

![thankyou_page](https://raw.githubusercontent.com/matthiaslmz/BERT-sentiment/master/results/feedback.png)
