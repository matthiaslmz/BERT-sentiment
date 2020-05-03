import os
import src.preprocess
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification,
                        AdamW, get_linear_schedule_with_warmup)

class BERTSentiment:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self,
                 bert_pretrained_model=None,
                 bert_pretrained_tokenizer=None,
                 train_batch_size=8,
                 eval_batch_size=8,
                 num_labels=2,
                 learning_rate=3e-5,
                 train_dset=None,
                 eval_dset=None):

        # define hyperparameters
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_labels = num_labels

        # loading pre-trained models
        self.config = BertConfig.from_pretrained(bert_pretrained_model)
        self.config.num_labels = num_labels
        self.model = BertForSequenceClassification.from_pretrained(
            bert_pretrained_model, config=self.config).to(self.DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)

        # creating / loading datasets
        self.train_dset = train_dset
        self.eval_dset = eval_dset
        self.train_loader = DataLoader(self.train_dset,
                                       batch_size=self.train_batch_size,
                                       shuffle=True)
        self.eval_loader = DataLoader(self.eval_dset,
                                      batch_size=self.eval_batch_size)
                                    
    def train_model(self,
                    num_epochs=5,
                    learning_rate=5e-5,
                    ckpt_every=1000,
                    warmup_ratio=0.1,
                    ckpt_path=None,
                    output_path=None):
        
        self.model.train()

        num_total_steps = len(self.train_loader) * num_epochs
        num_warmup_steps = int(num_total_steps * warmup_ratio)

        # instantiate optimizer
        optimizer = AdamW(self.model.parameters(),
                          lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                         num_warmup_steps=num_warmup_steps,
                                         num_training_steps=num_total_steps)

        # define empty lists and counters to keep track of metrics
        iter_count = 0
        ckpt_loss = 0
        all_train_loss = []
        all_eval_loss = []
        all_train_accuracy = []
        all_eval_accuracy = []
        ckpt_num_correct = 0
        ckpt_num_total = 0

        # loop over dataset for num_epochs
        for epoch in tqdm(range(num_epochs), desc="Epoch   "):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
            # loop over every batch within the training dataset
            for iter, batch in enumerate(tqdm(self.train_loader, desc="Training")):

                # reset optimizer gradient
                optimizer.zero_grad()

                # putting inputs to device
                inputs = batch["input_ids"]
                attns = batch["attention_masks"]
                labels = batch["label_tensor"]

                inputs = inputs.to(self.DEVICE)
                attns = attns.to(self.DEVICE)
                labels = labels.to(self.DEVICE)

                outputs = self.model(inputs, labels=labels, attention_mask=attns)
                loss, scores = outputs[:2]

                predictions = torch.argmax(scores, dim=-1)

                # backprop
                loss.backward()
                optimizer.step()
                scheduler.step()

                # add to totals for checkpoint
                ckpt_loss += loss.item()
                ckpt_num_correct += ((predictions == labels).long()).sum().item()
                ckpt_num_total += labels.size(0)

                # evaluate and save model every checkpoint
                if iter_count % ckpt_every == 0:

                    # loss
                    if iter_count == 0:
                        all_train_loss.append(ckpt_loss)
                    else:
                        all_train_loss.append(ckpt_loss / ckpt_every)
                    ckpt_loss = 0

                    # accuracy
                    all_train_accuracy.append(ckpt_num_correct / ckpt_num_total)
                    ckpt_num_correct = 0 
                    ckpt_num_total = 0

                    eval_loss, eval_accuracy = self.evaluate_model()

                    all_eval_loss.append(eval_loss)
                    all_eval_accuracy.append(eval_accuracy)

                    print(f"ckpt : {iter_count} ----- "
                          f"train_loss : {all_train_loss[-1]} ----- "
                          f"train_accuracy : {all_train_accuracy[-1]} -----"
                          f"eval_loss : {all_eval_loss[-1]} ----- "
                          f"eval_accuracy : {all_eval_accuracy[-1]}"
                          )

                    # save model checkpoint
                    ckpt_path_formatted = ckpt_path % (iter_count)
                    if not os.path.exists(ckpt_path_formatted):
                        os.makedirs(ckpt_path_formatted)

                        self.model.save_pretrained(ckpt_path_formatted)
                        self.tokenizer.save_pretrained(ckpt_path_formatted)
                    else:
                        torch.save(self.model, ckpt_path_formatted + ".pt")

                    output_path_formatted = output_path % (iter_count)
                    if not os.path.exists(output_path_formatted):
                        os.makedirs(output_path_formatted)
                    # save loss
                    np.savetxt(os.path.join(output_path_formatted, f"train_loss_{iter_count}.txt"),
                               all_train_loss)
                    np.savetxt(os.path.join(output_path_formatted, f"eval_loss_{iter_count}.txt"),
                               all_eval_loss)

                    # save accuracy
                    np.savetxt(os.path.join(output_path_formatted, f"train_accuracy_{iter_count}.txt"),
                               all_train_accuracy)
                    np.savetxt(os.path.join(output_path_formatted, f"eval_accuracy_{iter_count}.txt"),
                               all_eval_accuracy)

                iter_count += 1

    def evaluate_model(self, save_results=True):

        self.model.eval()

        total_eval_loss = 0
        total_num_correct = 0
        total_num_examples = 0
        all_labels = []
        all_predictions = []
        all_attns = []

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluate"):

                inputs = batch["input_ids"]
                attns = batch["attention_masks"]
                labels = batch["label_tensor"]

                # putting inputs to device
                inputs = inputs.to(self.DEVICE)
                labels = labels.to(self.DEVICE)
                attns = attns.to(self.DEVICE)

                # feed to model
                outputs = self.model(inputs, labels=labels,
                                    attention_mask=attns)
                loss, scores = outputs[:2]

                predictions = torch.argmax(scores, dim=-1)

                # add loss to total loss
                total_eval_loss += loss.item()
                total_num_correct += ((predictions == labels).long()).sum().item()
                total_num_examples += labels.size(0)

                # used to create confusion matrix
                all_labels += labels.cpu().flatten().tolist()
                all_predictions += predictions.cpu().flatten().tolist()
                all_attns += attns.cpu().flatten().tolist()

        self.model.train()

        # compute loss, accuracy and confusion matrix
        eval_loss = total_eval_loss / len(self.eval_loader)
        eval_accuracy = accuracy_score(all_labels, all_predictions)

        return eval_loss, eval_accuracy