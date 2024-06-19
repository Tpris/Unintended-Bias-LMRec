import numpy as np
import random
from tqdm import tqdm
from torch.autograd import Variable
import torch
from torch import nn

def get_sup(df, nb=100):
    df_count = df.groupby('business_id').count()
    df_plus = df_count[df_count.review_id >= nb]
    return df[df.business_id.isin(df_plus.index)]

def get_groupby_business(df):
    return df.groupby('business_id').count()

def get_groupby_price(df):
    return df.groupby('price').count()

def get_sentence_with_other_labels(input_data, id_row, name_lab):
  id = input_data.iloc[id_row]['idx_sentence']
  name = input_data.iloc[id_row]['example_label']
  labs = name_lab[name]

  input_data = input_data[input_data['idx_sentence'] == id]
  input_data = input_data[~input_data.label.isin(labs)].reset_index()
  return input_data.iloc[random.randint(0,input_data.shape[0]-1)]['index']

def get_firsts_idxs(logits,n):
  order_idx = torch.argsort(logits, descending=True)
  return order_idx[:,:n]

def mask_after_n_first(logits1, logits2,n, device):
  first_idxs1 = get_firsts_idxs(logits1,n)
  first_idxs2 = get_firsts_idxs(logits2,n)
  merge = torch.cat((first_idxs1, first_idxs2), dim=1).to(device)
  h,w = logits1.shape
  n = torch.Tensor([np.arange(w)]*h).int().to(device)
  for i, (ni, mi) in enumerate(zip(n,merge)): 
    ni = ni[~torch.isin(ni,mi)]
    logits1[i, ni] = 0
    logits2[i, ni] = 0
  return logits1, logits2

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

def load_weight_mitigator(model, model_dir):
    model.load_state_dict(torch.load(model_dir))
    model.mitigator[-1] = nn.Identity()
    return model

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

def train_mit(model, train_loader, optimizer, criterion, device):
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
