import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import os.path

from utils import mit_utils
from utils.model_utils import MyDataset, MyMitDataset, BERT_classifier, Bert_mitigator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

CITY = 'Atlanta'
NB_TOKEN = 512
BATCH_SIZE_R = 80
BATCH_SIZE_Q = 10


##############################
######## Data Cleaning #######
##############################
df = pd.read_csv('data/Yelp_cities/'+CITY+'_reviews.csv')

# Clean the dataframe
df = mit_utils.get_sup(df,2)
# apply good types
df.price = df.price.astype(float).fillna(0.0)
df.loc['review_date'] = pd.to_datetime(df['review_date'])
# filter data
df = df.loc[(df['review_date'] >= '2008-01-01') & (df['review_date'] <= '2020-01-01')]
df = mit_utils.get_sup(df,100)


labels = df['business_id']
print(len(labels))
LABELS = list(labels.unique())
NB_CLASSES = len(LABELS)
print(NB_CLASSES)

# Use saved dataloaders or create them
# The use of dataloaders help us to be faster and to use the same data everytime, it's better for comparing differents training
if os.path.isfile('train_dataloader.pkl') and os.path.isfile('val_dataloader.pkl'):
    print("Load dataloaders")
    with open('train_dataloader.pkl', 'rb') as f:
        train_loader_r = pickle.load(f)
    with open('val_dataloader.pkl', 'rb') as f:
        val_loader_r = pickle.load(f)
else:
    print("Create dataloaders")
    train, test = mit_utils.split(df)
    train, val = mit_utils.split(train)

    trainset = MyDataset(train['text'].astype(str).tolist(), train['business_id'], LABELS, NB_CLASSES)
    print('train done')
    valset = MyDataset(val['text'].astype(str).tolist(), val['business_id'], LABELS, NB_CLASSES)

    train_loader_r = DataLoader(trainset, batch_size=BATCH_SIZE_R, shuffle=True)
    val_loader_r = DataLoader(valset, batch_size=BATCH_SIZE_R, shuffle=False)

    with open('train_dataloader.pkl', 'wb') as f:
        pickle.dump(train_loader_r, f)
    with open('val_dataloader.pkl', 'wb') as f:
        pickle.dump(val_loader_r, f)


### Weight loss ###
# It's used to correct imbalanced data
nb_resto = mit_utils.get_groupby_business(df)['review_id'].tolist()
MAX_NB_RESTO = max(nb_resto)
ratio_resto = [MAX_NB_RESTO/p for p in nb_resto]
WEIGHT = torch.FloatTensor(ratio_resto).to(device)

##############################
#### Questions with names ####
##############################

# Load split data from files
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

trainset_q = MyMitDataset(train_q, name_lab_train)
valset_q = MyMitDataset(val_q, name_lab_val)

train_loader_q = DataLoader(trainset_q, batch_size=BATCH_SIZE_Q, shuffle=True)
val_loader_q = DataLoader(valset_q, batch_size=BATCH_SIZE_Q, shuffle=False)

##########################
######### Models #########
##########################

def fit(model1, model2, train_loader1, val_loader1, train_loader2, val_loader2, epochs, optimizer1, optimizer2, criterion1, criterion2, device):
    loss_train_review_per_epoch = []
    acc_train_review_per_epoch = []
    loss_val_review_per_epoch = []
    acc_val_review_per_epoch = []

    loss_train_mit_per_epoch = []
    acc_train_mit_per_epoch = []
    loss_val_mit_per_epoch = []
    acc_val_mit_per_epoch = []

    min_val_loss = 10000
    for epoch in range(epochs):
        train_review_loss, train_review_acc = mit_utils.train_review(model1, train_loader1, optimizer1, criterion1, device)
        train_mit_loss, train_mit_acc = mit_utils.train_mit(model2, train_loader2, optimizer2, criterion2, epoch+1, device)

        val_review_loss, val_review_acc = mit_utils.val_review(model1, val_loader1, criterion1, device)
        val_mit_loss, val_mit_acc = mit_utils.val_mit(model2, val_loader2, criterion2, device)

        loss_tot = val_review_loss+val_mit_loss
        if min_val_loss> loss_tot:
            min_val_loss = loss_tot
            torch.save(model1.state_dict(), 'best-model-parameters-2train.pt')
            print('model saved')

        print(f'Base: [{epoch + 1}, {len(train_loader1) + 1:5d}] loss: {train_review_loss:.3f}, accuracy: {train_review_acc:.3f} loss_val: {val_review_loss:.3f}, accuracy_val: {val_review_acc:.3f}')
        print(f'Mit: [{epoch + 1}, {len(train_loader2) + 1:5d}] loss: {train_mit_loss:.3f}, accuracy: {train_mit_acc:.3f} loss_val: {val_mit_loss:.3f}, accuracy_val: {val_mit_acc:.3f}')
        print(f'Total val loss: {loss_tot:.3f}')

        loss_train_review_per_epoch += [train_review_loss]
        acc_train_review_per_epoch += [train_review_acc]
        loss_val_review_per_epoch += [val_review_loss]
        acc_val_review_per_epoch += [val_review_acc]

        loss_train_mit_per_epoch += [train_mit_loss]
        acc_train_mit_per_epoch += [train_mit_acc]
        loss_val_mit_per_epoch += [val_mit_loss]
        acc_val_mit_per_epoch += [val_mit_acc]

    return loss_train_review_per_epoch, loss_val_review_per_epoch, acc_train_review_per_epoch, acc_val_review_per_epoch, loss_train_mit_per_epoch, loss_val_mit_per_epoch, acc_train_mit_per_epoch, acc_val_mit_per_epoch



model = BERT_classifier(NB_CLASSES).to(device)
model_mit = Bert_mitigator(model, business_price).to(device)

optimizer1 = optim.Adam(model.parameters(), lr=5e-3)
optimizer2 = optim.Adam(model.parameters(), lr=1e-10)
criterion1 = nn.CrossEntropyLoss(weight=WEIGHT)
criterion2 = nn.MSELoss()
epochs=30
results = fit(model, model_mit, train_loader_r, val_loader_r, train_loader_q, val_loader_q, epochs, optimizer1, optimizer2, criterion1, criterion2, device)
loss_train_review_per_epoch, loss_val_review_per_epoch, acc_train_review_per_epoch, acc_val_review_per_epoch, loss_train_question_per_epoch, loss_val_question_per_epoch, acc_train_question_per_epoch, acc_val_question_per_epoch = results






