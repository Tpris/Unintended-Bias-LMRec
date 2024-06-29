import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from torch.autograd import Variable

from utils import mit_utils, model_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

CITY = 'Atlanta'
N = 1

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

trainset = model_utils.MyMitDataset(train_q, name_lab_train)
valset = model_utils.MyMitDataset(val_q, name_lab_val)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)


### Model

model = model_utils.BERT_classifier(NB_CLASSES)
model.load_state_dict(torch.load('models/'+CITY+'/model.pt', map_location=torch.device(device)))
model_mit = model_utils.Bert_patch_mitigator(model, business_price).to(device)

### Training

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
        output, output2 = mit_utils.mask_after_n_first(output, output2, N, device)
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
            output, output2 = mit_utils.mask_after_n_first(output, output2, N, device)
            val_loss += criterion(output, output2).item()

            pred = output.argmax(dim=1)
            lab = output2.argmax(dim=1)
            correct += pred.eq(lab).sum().item()

        val_loss /= len(val_loader)
        acc = correct/len(val_loader.dataset)

    if min_val_loss>val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'best-model_mit_firsts-parameters.pt')
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
epochs=5
loss_train_per_epoch, loss_val_per_epoch, acc_train_per_epoch, acc_val_per_epoch = fit(model_mit, train_loader, val_loader, epochs, optimizer, criterion)







