from transformers import BertTokenizer,BertModel
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer,BertModel
from torch import nn
import torch
import numpy as np

from utils.mit_utils import get_sentence_with_other_labels

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

        return logits.squeeze()
    

class Bert_patch_mitigator(nn.Module):
  def __init__(self, LMRec, business_price=None):
    super().__init__()
    self.LMRec = LMRec
    for param in self.LMRec.parameters():
        param.requires_grad = False

    nb_output = self.LMRec.classifier[-1].out_features

    if business_price is not None:
        price_idx = business_price['price'].to_numpy().astype(int)-1
        price_weight = np.zeros((nb_output, 4))
        price_weight[np.arange(nb_output), price_idx] = 1
        price_weight = price_weight.T

    self.mitigator = nn.Sequential(
        nn.Linear(nb_output, nb_output),
        nn.Softmax(1),
        nn.Linear(nb_output, 4),
    )

    if business_price is not None:
        self.mitigator[-1].weight = torch.nn.Parameter(torch.from_numpy(price_weight).type(torch.float32))
        self.mitigator[-1].bias = torch.nn.Parameter(torch.zeros(4).type(torch.float32))
        for p in self.mitigator[-1].parameters():
            p.requires_grad = False

  def forward(self, wrapped_input):
        hidden = self.LMRec(wrapped_input)
        logits = self.mitigator(hidden)

        return logits.squeeze()

class Bert_mitigator(nn.Module):
  def __init__(self, LMRec, business_price, DECODER=False, EMBEDDING=False):
    super().__init__()
    self.LMRec = LMRec

    if DECODER or EMBEDDING:
        for param in self.LMRec.parameters():
            param.requires_grad = False
        if DECODER:
            for param in self.LMRec.classifier[-1].parameters():
                param.requires_grad = True
        if EMBEDDING:
            for param in self.LMRec.bert.embeddings.parameters():
                param.requires_grad = True

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
    def __init__(self, input_data, name_lab):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.wrapped_input = tokenizer(input_data['input_sentence'].astype(str).tolist(), max_length=512, add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        self.input_data = input_data
        self.name_lab= name_lab

    def __getitem__(self, idx):
        input_dict = {}
        for k in self.wrapped_input.keys():
            input_dict[k] = self.wrapped_input[k][idx]

        idx2 = get_sentence_with_other_labels(self.input_data, idx, self.name_lab)
        input_dict2 = {}
        for k in self.wrapped_input.keys():
            input_dict2[k] = self.wrapped_input[k][idx2]

        return input_dict, input_dict2

    def __len__(self):
        return self.input_data.shape[0]
    

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