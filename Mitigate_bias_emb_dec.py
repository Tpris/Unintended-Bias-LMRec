import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer,BertModel
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import random
from torch.autograd import Variable

from utils import mit_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

CITY = 'Atlanta'
DECODER = False
EMBEDDING = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = pd.read_csv('data/Yelp_cities/'+CITY+'_reviews.csv')

df = mit_utils.get_sup(df,2)
df.price = df.price.astype(float).fillna(0.0)
df.loc['review_date'] = pd.to_datetime(df['review_date'])
df = df.loc[(df['review_date'] >= '2008-01-01') & (df['review_date'] <= '2020-01-01')]
df = mit_utils.get_sup(df,100)

NB_TOKEN = 512

labels = df['business_id']
print(len(labels))
LABELS = list(labels.unique())
NB_CLASSES = len(LABELS)
print(NB_CLASSES)


# ## Questions
train_q = pd.read_csv('data/bias_analysis/yelp/input_sentences/name_templates_split/names_templates_train.csv')
val_q = pd.read_csv('data/bias_analysis/yelp/input_sentences/name_templates_split/names_templates_validation.csv')


# ## Add indice sequences
train_q = mit_utils.add_index_question(train_q)
val_q = mit_utils.add_index_question(val_q)


### Get labels

### Group
name_lab_train = mit_utils.get_name_labels(train_q)
name_lab_val = mit_utils.get_name_labels(val_q)


### Price
business_price = df.groupby('business_id')['price'].unique()
business_price = business_price.to_frame()
business_price.price = business_price.price.astype(float)


### Dataloader

BATCH_SIZE = 100
if EMBEDDING:
    BATCH_SIZE = 5

trainset = mit_utils.MyMitDataset(train_q, name_lab_train)
valset = mit_utils.MyMitDataset(val_q, name_lab_val)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


### Model

model = mit_utils.BERT_classifier(NB_CLASSES)
model.load_state_dict(torch.load('models/'+CITY+'/model.pt', map_location=torch.device(device)))
model_mit = mit_utils.Bert_mitigator(model, business_price, DECODER, EMBEDDING).to(device)

### Training

### DEBUG

# batch_data, batch_label = next(iter(train_loader))
# for k, v in batch_data.items():
#     batch_data[k] = v.to(device)
#     print(v.shape)
# # batch_label = batch_label.to(device)
# for k, v in batch_label.items():
#     batch_label[k] = v.to(device)

# print(batch_label)

# batch_logits = model_mit(batch_data)
# print(batch_logits.shape)
# print(batch_logits)
# print(batch_label)
# print(batch_data['input_ids'].shape)
# # print(batch_label[0])

def fit(model, train_loader, val_loader, epochs, optimizer, criterion):
    loss_train_per_epoch = []
    acc_train_per_epoch = []
    loss_val_per_epoch = []
    acc_val_per_epoch = []
    min_val_loss = 1000000000
    for epoch in range(epochs):
        train_loss, train_acc = mit_utils.train_mit(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = mit_utils.val_mit(model, val_loader, criterion, device)

        if min_val_loss>val_loss:
            min_val_loss = val_loss
            torch.save(model.LMRec.state_dict(), 'models/'+CITY+'/best-model_mit_emb_dec-parameters.pt')
            print('model saved')

        print(f'[{epoch + 1}, {len(train_loader) + 1:5d}] loss: {train_loss:.3f}, accuracy: {train_acc:.3f} loss_val: {val_loss:.3f}, accuracy_val: {val_acc:.3f}')

        loss_train_per_epoch += [train_loss]
        acc_train_per_epoch += [train_acc]
        loss_val_per_epoch += [val_loss]
        acc_val_per_epoch += [val_acc]

    return loss_train_per_epoch, loss_val_per_epoch, acc_train_per_epoch, acc_val_per_epoch

optimizer = optim.Adam(model.parameters(), lr=1e-1)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
epochs=10
loss_train_per_epoch, loss_val_per_epoch, acc_train_per_epoch, acc_val_per_epoch = fit(model_mit, train_loader, val_loader, epochs, optimizer, criterion)



# model_mit.classifier[-1] = nn.Identity()

# batch_data, batch_label = next(iter(train_loader))
# for k, v in batch_data.items():
#     batch_data[k] = v.to(device)
#     print(v.shape)
# # batch_label = batch_label.to(device)
# for k, v in batch_label.items():
#     batch_label[k] = v.to(device)

# print(batch_label)

# batch_logits = model_mit(batch_data)
# print(batch_logits.shape)
# print(batch_logits)
# print(batch_label)
# print(batch_data['input_ids'].shape)
# # print(batch_label[0])






