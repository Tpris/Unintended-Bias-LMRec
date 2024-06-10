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

class MyDataset(Dataset):
    def __init__(self, df):
        self.wrapped_input = tokenizer(df['input_sentence'].astype(str).tolist(), max_length=NB_TOKEN, add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        self.df = df

    def __getitem__(self, idx):
        input_dict = {}
        for k in self.wrapped_input.keys():
            input_dict[k] = self.wrapped_input[k][idx]

        idx2 = mit_utils.get_sentence_with_other_labels(self.df, idx)
        input_dict2 = {}
        for k in self.wrapped_input.keys():
            input_dict2[k] = self.wrapped_input[k][idx2]

        return input_dict, input_dict2

    def __len__(self):
        return self.df.shape[0]

# batch_data, batch_label = next(iter(train_loader))

BATCH_SIZE = 100

trainset = mit_utils.MyMitDataset(train_q, name_lab_train)
valset = mit_utils.MyMitDataset(val_q, name_lab_val)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


### Model

class Bert_mitigator(nn.Module):
  def __init__(self, LMRec, business_price):
    super().__init__()
    self.LMRec = LMRec
    for param in self.LMRec.parameters():
        param.requires_grad = False

    nb_output = self.LMRec.classifier[-1].out_features

    price_idx = business_price['price'].to_numpy().astype(int)-1
    price_weight = np.zeros((nb_output, 4))
    price_weight[np.arange(nb_output), price_idx] = 1
    price_weight = price_weight.T

    self.mitigator = nn.Sequential(
        nn.Linear(nb_output, nb_output),
        nn.Softmax(1),
        # nn.ReLU(),
        nn.Linear(nb_output, 4),
    )

    self.mitigator[-1].weight = torch.nn.Parameter(torch.from_numpy(price_weight).type(torch.float32))
    self.mitigator[-1].bias = torch.nn.Parameter(torch.zeros(4).type(torch.float32))
    for p in self.mitigator[-1].parameters():
        p.requires_grad = False

  def forward(self, wrapped_input):
        hidden = self.LMRec(wrapped_input)
        # last_hidden_state, pooler_output = hidden[0], hidden[1]
        # # print('TTTTTTTT',last_hidden_state.shape)
        # # print('YYYYYYYYYY',pooler_output.shape)
        # print('hidden :',hidden.shape)
        logits = self.mitigator(hidden)
        # logits = self.softmax(logits)

        return logits.squeeze()

model = mit_utils.BERT_classifier(NB_CLASSES)
model.load_state_dict(torch.load('models/'+CITY+'/model.pt', map_location=torch.device(device)))
model_mit = Bert_mitigator(model, business_price).to(device)

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

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss, correct = 0, 0
    for data, data2 in tqdm(train_loader):
        for k, v in data.items():
            data[k] = v.to(device)

        for k, v in data2.items():
            data2[k] = v.to(device)

        optimizer.zero_grad()
        output = model(data)
        output2 = model(data2)
        loss = criterion(output, output2)
        loss = Variable(loss, requires_grad = True)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        lab = output2.argmax(dim=1)
        correct += pred.eq(lab).sum().item()

    train_loss /= len(train_loader)
    acc = correct/len(train_loader.dataset)

    return train_loss, acc


def val(model, val_loader, criterion, min_val_loss):
    model.eval()
    val_loss, correct = 0, 0.
    with torch.no_grad():
        for data, data2 in val_loader:
            for k, v in data.items():
                data[k] = v.to(device)

            for k, v in data2.items():
                data2[k] = v.to(device)

            output = model(data)
            output2 = model(data2)
            val_loss += criterion(output, output2).item()

            pred = output.argmax(dim=1)
            lab = output2.argmax(dim=1)
            correct += pred.eq(lab).sum().item()

        val_loss /= len(val_loader)
        acc = correct/len(val_loader.dataset)

    if min_val_loss>val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'best-model_mit-parameters.pt')
        print('model saved')

    return val_loss, acc, min_val_loss

def fit(model, train_loader, val_loader, epochs, optimizer, criterion):
    loss_train_per_epoch = []
    acc_train_per_epoch = []
    loss_val_per_epoch = []
    acc_val_per_epoch = []
    min_val_loss = 1000
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch+1)
        val_loss, val_acc, min_val_loss = val(model, val_loader, criterion, min_val_loss)

        print(f'[{epoch + 1}, {len(train_loader) + 1:5d}] loss: {train_loss:.3f}, accuracy: {train_acc:.3f} loss_val: {val_loss:.3f}, accuracy_val: {val_acc:.3f}')

        loss_train_per_epoch += [train_loss]
        acc_train_per_epoch += [train_acc]
        loss_val_per_epoch += [val_loss]
        acc_val_per_epoch += [val_acc]

    return loss_train_per_epoch, loss_val_per_epoch, acc_train_per_epoch, acc_val_per_epoch

optimizer = optim.Adam(model.parameters(), lr=1e-50)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
epochs=5
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






