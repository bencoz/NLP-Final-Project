import pandas as pd
import numpy as np
import gdown
import os.path
import torch
import transformers
from torch import nn
from transformers import AutoConfig

DATA_URL = "https://drive.google.com/uc?id=1RiQbgQRgdqxXZWwbkjLh_SnrN5pCysJ-"
FILE_PATH = "./data.csv"


class MultiTaskModel(nn.Module):
    def __init__(self, freeze_bert=False):
        super(MultiTaskModel, self).__init__()
        self.bert = transformers.AutoModelForSequenceClassification.from_config(
            AutoConfig.from_pretrained('bert-base-cased'))

        # Instantiate an one-layer feed-forward classifier
        self.fc1 = nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(50, 1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(50, 2)
        )
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        # Feed input to classifiers
        age = torch.sigmoid(self.fc1(last_hidden_state_cls))
        gender = self.fc2(last_hidden_state_cls)

        return [age, gender]


if __name__ == '__main__':
    if not os.path.exists(FILE_PATH):
        gdown.download(DATA_URL, "data.csv", quiet=False)

    data = pd.read_csv('./data.csv')
    print(data.columns)
    # Remove unnecessary columns
    data.drop('topic', axis=1, inplace=True)
    data.drop('sign', axis=1, inplace=True)
    data.drop('date', axis=1, inplace=True)

    nRow, nCol = data.shape
    print(f'There are {nRow} rows and {nCol} columns')
    print(f"max age is: {data['age'].max()}")

    data['age'] = data['age'] / data['age'].max()  # Age (non-negative) normalized to 0 - 1
    data['gender'].replace(['male', 'female'], {0, 1}, inplace=True)

    # train_size = int(0.8 * len(data))
    # test_size = len(data) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    # print(f'train len is: {len(train_dataset)} & test len is: {len(test_dataset)}')
