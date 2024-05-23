import numpy as np
import random
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer,BertModel
from torch import nn
from torch.autograd import Variable
import torch

def get_sup(df, nb=100):
    df_count = df.groupby('business_id').count()
    df_plus = df_count[df_count.review_id >= nb]
    return df[df.business_id.isin(df_plus.index)]

def get_groupby_business(df):
    return df.groupby('business_id').count()

def get_groupby_price(df):
    return df.groupby('price').count()

def get_sentence_with_other_labels(df, id_row, input_data, name_lab):
  id = input_data.iloc[id_row]['idx_sentence']
  name = input_data.iloc[id_row]['example_label']
  labs = name_lab[name]

  df = df[df['idx_sentence'] == id]
  df = df[~df.label.isin(labs)].reset_index()
  return df.iloc[random.randint(0,df.shape[0]-1)]['index']

def split(df,ratio = 0.9):
    msk = np.random.rand(len(df)) < ratio
    train = df[msk]
    test = df[~msk]
    return train, test

def split_by_name(df,ratio = 0.9):
  names = df['example_label'].unique()
  msk = np.random.rand(len(names)) < ratio
  n_train = names[msk]
  train = df[df.example_label.isin(n_train)].reset_index(drop=True)
  test = df[~df.example_label.isin(n_train)].reset_index(drop=True)
  return train, test

def add_index_question(input_questions):
    sentences = {}
    idx = 0
    list_idx = []
    for _,r in input_questions.iterrows():
        sents = r['input_sentence'].split(r['example_label'])
        subsent = max(sents, key=len)
        if subsent not in sentences:
            sentences[subsent] = idx
            idx+=1
        list_idx += [sentences[subsent]]

    input_questions['idx_sentence'] = list_idx
    return input_questions

def get_name_labels(input_questions):
    name_lab = input_questions[['example_label', 'label']]
    name_lab = name_lab.groupby('example_label')['label'].unique()
    return name_lab

def train_review(model, train_loader, optimizer, criterion, epoch, device):
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


def val_review(model, val_loader, criterion, device):
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

    return val_loss, acc

def train_mit(model, train_loader, optimizer, criterion, epoch, device):
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


def val_mit(model, val_loader, criterion, device):
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

    return val_loss, acc

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
        train_review_loss, train_review_acc = train_review(model1, train_loader1, optimizer1, criterion1, epoch+1, device)
        train_mit_loss, train_mit_acc = train_mit(model2, train_loader2, optimizer2, criterion2, epoch+1, device)

        val_review_loss, val_review_acc = val_review(model1, val_loader1, criterion1, device)
        val_mit_loss, val_mit_acc = val_mit(model2, val_loader2, criterion2, device)

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
        )

    def forward(self, wrapped_input):
        hidden = self.bert(**wrapped_input)
        last_hidden_state, pooler_output = hidden[0], hidden[1]
        logits = self.classifier(pooler_output)
        # logits = self.softmax(logits)

        return logits.squeeze()

class Bert_mitigator(nn.Module):
  def __init__(self, LMRec, business_price, DECODER, EMBEDDING):
    super().__init__()
    self.LMRec = LMRec
    # for param in self.LMRec.parameters():
    #     param.requires_grad = False
    # if DECODER:
    #     for param in self.LMRec.classifier.parameters():
    #         param.requires_grad = True
    # if EMBEDDING:
    #     for param in self.LMRec.bert.embeddings.parameters():
    #         param.requires_grad = True

    nb_output = self.LMRec.classifier[-1].out_features

    price_idx = business_price['price'].to_numpy().astype(int)-1
    price_weight = np.zeros((nb_output, 4))
    price_weight[np.arange(nb_output), price_idx] = 1
    price_weight = price_weight.T

    self.mitigator = nn.Sequential(
        nn.Softmax(1),
        nn.Linear(nb_output, 4),
    )

    self.mitigator[-1].weight = torch.nn.Parameter(torch.from_numpy(price_weight).type(torch.float32))
    self.mitigator[-1].bias = torch.nn.Parameter(torch.zeros(4).type(torch.float32))
    for p in self.mitigator[-1].parameters():
        p.requires_grad = False

  def forward(self, wrapped_input):
        hidden = self.LMRec(wrapped_input)
        logits = self.mitigator(hidden)
        return logits.squeeze()
  

class MyMitDataset(Dataset):
    def __init__(self, df, input_data, name_lab):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.wrapped_input = tokenizer(df['input_sentence'].astype(str).tolist(), max_length=512, add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        self.df = df
        self.input_data = input_data
        self.name_lab= name_lab

    def __getitem__(self, idx):
        input_dict = {}
        for k in self.wrapped_input.keys():
            input_dict[k] = self.wrapped_input[k][idx]

        idx2 = get_sentence_with_other_labels(self.df, idx, self.input_data, self.name_lab)
        input_dict2 = {}
        for k in self.wrapped_input.keys():
            input_dict2[k] = self.wrapped_input[k][idx2]

        return input_dict, input_dict2

    def __len__(self):
        return self.df.shape[0]
    

class MyDataset(Dataset):
    def __init__(self, sentence, labels, LABELS, NB_CLASSES):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.wrapped_input = tokenizer(sentence, max_length=512, add_special_tokens=True, truncation=True,
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