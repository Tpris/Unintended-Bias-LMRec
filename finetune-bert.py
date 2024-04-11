import torch 
import pickle
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
from datasets import load_dataset
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

CITY = 'Boston'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

df = pd.read_csv('data_reviews/'+CITY+'_reviews.csv')
with open("data_reviews/"+CITY+"_trainValidTest/labels.pickle", "rb") as f:
    x = pickle.load(f)
df_count = df.groupby('business_id').count()
df_count = df_count[df_count.index.isin(x)]
df = df[df.business_id.isin(df_count.index)]

labels = df['business_id']
print(len(labels))
LABELS = list(labels.unique())
# print(LABELS)
NB_CLASSES = len(LABELS)
print(NB_CLASSES)

NB_TOKEN = 512

def split(df,ratio = 0.8):
    msk = np.random.rand(len(df)) < ratio
    train = df[msk]
    test = df[~msk]
    return train, test

train, test = split(df)
train, val = split(train)
print(len(train))
print(len(val))
print(len(test))

class MyDataset(Dataset):
    def __init__(self, sentence, labels):
        self.wrapped_input = tokenizer(sentence, max_length=NB_TOKEN, add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        lab_idx = np.array([LABELS.index(l) for l in labels])
        self.labels = np.zeros((lab_idx.size, NB_CLASSES))
        self.labels[np.arange(lab_idx.size), lab_idx] = 1
        # self.labels = np.array(labels).astype(float)
    def __getitem__(self, idx):
        input_dict = {}
        for k in self.wrapped_input.keys():
            input_dict[k] = self.wrapped_input[k][idx]
        return input_dict, self.labels[idx]
    
    def __len__(self):
        return len(self.labels)


BATCH_SIZE = 100

trainset = MyDataset(train['text'].astype(str).tolist(), train['business_id'])
print('train done')
valset = MyDataset(val['text'].astype(str).tolist(), val['business_id'])

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

class BERT_classifier(nn.Module):
    def __init__(self, num_label):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_label),
            # nn.ReLU(),
            # nn.Linear(1500, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2500),
            # nn.ReLU(),
            # nn.Linear(2500, num_label),
            # nn.ReLU(),
            # nn.Linear(3000, num_label),
            # nn.ReLU(),
            # nn.Linear(10000, 20000),
            # nn.ReLU(),
            # nn.Linear(20000, 40000),
            # nn.ReLU(),
            # nn.Linear(40000, num_label),
            # nn.Softmax(1)
        )

    def forward(self, wrapped_input):
        hidden = self.bert(**wrapped_input)
        last_hidden_state, pooler_output = hidden[0], hidden[1]
        logits = self.classifier(pooler_output)
        # logits = self.softmax(logits)

        return logits.squeeze()

    # def forward(self, input_ids, token_type_ids, attention_mask=None):
    #     pooled_out = self.bert(input_ids, token_type_ids, attention_mask)[1]
    #     logits = self.classifier(pooled_out)
    #     return logits


model = BERT_classifier(NB_CLASSES).to(device)
print(model)

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss, correct = 0, 0
    for data, label in tqdm(train_loader):
        device_data = {}
        for k, v in data.items():
            device_data[k] = v.to(device)
        device_label = label.to(device)
        
        optimizer.zero_grad()
        output = model(device_data)
        loss = criterion(output, device_label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        lab = device_label.argmax(dim=1)
        correct += pred.eq(lab).sum().item()

    train_loss /= len(train_loader)
    acc = correct/len(train_loader.dataset)

    return train_loss, acc


def val(model, val_loader, criterion, min_val_loss):
    model.eval()
    val_loss, correct = 0, 0.
    with torch.no_grad():
        for data, label in val_loader:
            device_data = {}
            for k, v in data.items():
                device_data[k] = v.to(device)
            device_label = label.to(device)
            
            output = model(device_data)
            val_loss += criterion(output, device_label).item()

            pred = output.argmax(dim=1)
            lab = device_label.argmax(dim=1)
            correct += pred.eq(lab).sum().item()
            
        val_loss /= len(val_loader)
        acc = correct/len(val_loader.dataset)
        
    if min_val_loss>val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'model-parameters-'+CITY+'.pt')
        print('model saved')

    return val_loss, acc

def fit(model, train_loader, val_loader, epochs, optimizer, criterion):
    loss_train_per_epoch = []
    acc_train_per_epoch = []
    loss_val_per_epoch = []
    acc_val_per_epoch = []
    min_val_loss = 1000
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch+1)
        val_loss, val_acc = val(model, val_loader, criterion, min_val_loss)

        print(f'[{epoch + 1}, {len(train_loader) + 1:5d}] loss: {train_loss:.3f}, accuracy: {train_acc:.3f} loss_val: {val_loss:.3f}, accuracy_val: {val_acc:.3f}')
        
        loss_train_per_epoch += [train_loss]
        acc_train_per_epoch += [train_acc]
        loss_val_per_epoch += [val_loss]
        acc_val_per_epoch += [val_acc]
        
    return loss_train_per_epoch, loss_val_per_epoch, acc_train_per_epoch, acc_val_per_epoch
    
    

optimizer = optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()
epochs=100
loss_train_per_epoch, loss_val_per_epoch, acc_train_per_epoch, acc_val_per_epoch = fit(model, train_loader, val_loader, epochs, optimizer, criterion)


# batch_data, batch_label = next(iter(val_loader))
# for k, v in batch_data.items():
#     batch_data[k] = v.to(device)
# batch_logits = model(batch_data)
# print(batch_logits)
# print(batch_label)


# def show_loss(loss_train,loss_val):
#     plt.figure(figsize=(10,5))
#     plt.title("Training and Validation Loss")
#     plt.plot(loss_val,label="val")
#     plt.plot(loss_train,label="train")
#     plt.xlabel("iterations")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()

# def show_acuracy(acc_train,acc_val):
#     plt.figure(figsize=(10,5))
#     plt.title("Training and Validation Accuracy")
#     plt.plot(acc_val,label="val")
#     plt.plot(acc_train,label="train")
#     plt.xlabel("iterations")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.show()

# def show_matrix_confusion(model, val_dataloader):
#     model.eval()
#     val_acc = 0.0
#     true_lab = []
#     pred = []
#     with torch.no_grad():
#         for data, label in val_loader:
#             device_data = {}
#             for k, v in data.items():
#                 device_data[k] = v.to(device)
#             device_label = label.to(device)
            
#             output = model(device_data)
#             out = output.argmax(dim=1)
#             lab = device_label.argmax(dim=1)
#             val_acc += out.eq(lab).sum().item()
#             true_lab += [lab.cpu().numpy()]
#             pred += [out.cpu().numpy()]
    
#     print(f'accuracy: {val_acc / len(val_dataloader.dataset):.3f}')
#     y_pred = np.concatenate(pred)
#     y_true = np.concatenate(true_lab)

#     print("Balanced accuracy :",balanced_accuracy_score(y_true, y_pred))
    
#     confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
    
#     cm_display.plot()
#     plt.show() 


# show_acuracy(acc_train_per_epoch,acc_val_per_epoch)
# show_loss(loss_train_per_epoch,loss_val_per_epoch)
# show_matrix_confusion(model, val_loader)




