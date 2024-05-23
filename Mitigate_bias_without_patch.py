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
DECODER = True
EMBEDDING = False
NB_TOKEN = 512
BATCH_SIZE_R = 80
BATCH_SIZE_Q = 10


##############################
######## Data Cleaning #######
##############################
df = pd.read_csv('data/Yelp_cities/'+CITY+'_reviews.csv')

df = mit_utils.get_sup(df,2)
df.price = df.price.astype(float).fillna(0.0)
df.loc['review_date'] = pd.to_datetime(df['review_date'])
df = df.loc[(df['review_date'] >= '2008-01-01') & (df['review_date'] <= '2020-01-01')]
df = mit_utils.get_sup(df,100)


labels = df['business_id']
print(len(labels))
LABELS = list(labels.unique())
NB_CLASSES = len(LABELS)
print(NB_CLASSES)

# train, test = mit_utils.split(df)
# train, val = mit_utils.split(train)

# trainset = mit_utils.MyDataset(train['text'].astype(str).tolist(), train['business_id'], LABELS, NB_CLASSES)
# print('train done')
# valset = mit_utils.MyDataset(val['text'].astype(str).tolist(), val['business_id'], LABELS, NB_CLASSES)

# train_loader_r = DataLoader(trainset, batch_size=BATCH_SIZE_R, shuffle=True)
# val_loader_r = DataLoader(valset, batch_size=BATCH_SIZE_R, shuffle=False)

# with open('train_dataloader.pkl', 'wb') as f:
#     pickle.dump(train_loader_r, f)
# with open('val_dataloader.pkl', 'wb') as f:
#     pickle.dump(val_loader_r, f)

with open('train_dataloader.pkl', 'rb') as f:
    train_loader_r = pickle.load(f)
with open('val_dataloader.pkl', 'rb') as f:
    val_loader_r = pickle.load(f)

### Weight loss ###
nb_resto = mit_utils.get_groupby_business(df)['review_id'].tolist()
MAX_NB_RESTO = max(nb_resto)
ratio_resto = [MAX_NB_RESTO/p for p in nb_resto]
WEIGHT = torch.FloatTensor(ratio_resto).to(device)

##############################
#### Questions with names ####
##############################
train_q = pd.read_csv('data/bias_analysis/yelp/input_sentences/name_split/names_train.csv')
val_q = pd.read_csv('data/bias_analysis/yelp/input_sentences/name_split/names_validation.csv')

train_q = mit_utils.add_index_question(train_q)
val_q = mit_utils.add_index_question(val_q)

### Retrieve groups
name_lab_train = mit_utils.get_name_labels(train_q)
name_lab_val = mit_utils.get_name_labels(val_q)

### Retrieve prices
business_price = df.groupby('business_id')['price'].unique()
business_price = business_price.to_frame()
business_price.price = business_price.price.astype(float)

trainset_q = mit_utils.MyMitDataset(train_q, train_q, name_lab_train)
valset_q = mit_utils.MyMitDataset(val_q, val_q, name_lab_val)

train_loader_q = DataLoader(trainset_q, batch_size=BATCH_SIZE_Q, shuffle=True)
val_loader_q = DataLoader(valset_q, batch_size=BATCH_SIZE_Q, shuffle=False)

##########################
######### Models #########
##########################


model = mit_utils.BERT_classifier(NB_CLASSES).to(device)
model_mit = mit_utils.Bert_mitigator(model, business_price, DECODER, EMBEDDING).to(device)


#DEBUG
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



optimizer1 = optim.Adam(model.parameters(), lr=5e-3)
optimizer2 = optim.Adam(model.parameters(), lr=5e-3)
criterion1 = nn.CrossEntropyLoss(weight=WEIGHT)
criterion2 = nn.MSELoss()
epochs=30
results = mit_utils.fit(model, model_mit, train_loader_r, val_loader_r, train_loader_q, val_loader_q, epochs, optimizer1, optimizer2, criterion1, criterion2, device)
loss_train_review_per_epoch, loss_val_review_per_epoch, acc_train_review_per_epoch, acc_val_review_per_epoch, loss_train_question_per_epoch, loss_val_question_per_epoch, acc_train_question_per_epoch, acc_val_question_per_epoch = results


###########################################################

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

# model_mit.load_state_dict(torch.load('models/'+CITY+'/model_mit.pt'))

# state_dict = model_mit.state_dict()

# # Get parameter names
# parameter_names = state_dict.keys()

# # Print parameter names
# for name in parameter_names:
#     print(name)





