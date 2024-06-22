import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import os.path

from utils import mit_utils
from utils.model_utils import MyDataset, BERT_classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

CITY = 'Atlanta'
NB_TOKEN = 512
BATCH_SIZE = 80


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


if os.path.isfile('train_dataloader.pkl') and os.path.isfile('val_dataloader.pkl'):
    print("Load dataloaders")
    with open('train_dataloader.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    with open('val_dataloader.pkl', 'rb') as f:
        val_loader = pickle.load(f)
else:
    print("Create dataloaders")
    train, test = mit_utils.split(df)
    train, val = mit_utils.split(train)

    trainset = MyDataset(train['text'].astype(str).tolist(), train['business_id'], LABELS, NB_CLASSES)
    print('train done')
    valset = MyDataset(val['text'].astype(str).tolist(), val['business_id'], LABELS, NB_CLASSES)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    with open('train_dataloader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
    with open('val_dataloader.pkl', 'wb') as f:
        pickle.dump(val_loader, f)


### Weight loss ###
nb_resto = mit_utils.get_groupby_business(df)['review_id'].tolist()
MAX_NB_RESTO = max(nb_resto)
ratio_resto = [MAX_NB_RESTO/p for p in nb_resto]
WEIGHT = torch.FloatTensor(ratio_resto).to(device)

#########################
######### Model #########
#########################

def fit(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    loss_train_review_per_epoch = []
    acc_train_review_per_epoch = []
    loss_val_review_per_epoch = []
    acc_val_review_per_epoch = []

    min_val_loss = 10000
    for epoch in range(epochs):
        train_review_loss, train_review_acc = mit_utils.train_review(model, train_loader, optimizer, criterion, device)
        val_review_loss, val_review_acc = mit_utils.val_review(model, val_loader, criterion, device)

        if min_val_loss> val_review_loss:
            min_val_loss = val_review_loss
            torch.save(model.state_dict(), 'best-model-parameters.pt')
            print('model saved')

        print(f'Base: [{epoch + 1}, {len(train_loader) + 1:5d}] loss: {train_review_loss:.3f}, accuracy: {train_review_acc:.3f} loss_val: {val_review_loss:.3f}, accuracy_val: {val_review_acc:.3f}')

        loss_train_review_per_epoch += [train_review_loss]
        acc_train_review_per_epoch += [train_review_acc]
        loss_val_review_per_epoch += [val_review_loss]
        acc_val_review_per_epoch += [val_review_acc]

    return loss_train_review_per_epoch, loss_val_review_per_epoch, acc_train_review_per_epoch, acc_val_review_per_epoch



model = BERT_classifier(NB_CLASSES).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss(weight=WEIGHT)
epochs=30
results = fit(model, train_loader, val_loader, epochs, optimizer, criterion, device)
loss_train_review_per_epoch, loss_val_review_per_epoch, acc_train_review_per_epoch, acc_val_review_per_epoch = results






