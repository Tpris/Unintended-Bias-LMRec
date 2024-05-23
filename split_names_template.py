import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

Names = pd.read_json('data/bias_analysis/example_labels/names.json')
Names = Names.transpose()
Names['label'] = Names.index
Names = Names.melt(id_vars=["label"], 
        var_name="IDX", 
        value_name="name").drop(columns=['IDX'])
Names = Names.groupby('name')['label'].apply(list).reset_index(name='labels')
Names['labels'] = Names.labels.astype(str)

names_grp = Names.groupby('labels')['name'].apply(list).reset_index(name='names')


def split(row, ratio1 = 0.66, ratio2 = 0.5):
    # row = names_grp['names'].to_list()[id_row]
    row = np.array(row)
    msk = np.random.rand(len(row)) < ratio1
    train = row[msk]
    test = row[~msk]
    msk = np.random.rand(len(train)) < ratio2
    validation = train[~msk]
    train = train[msk]
    return train, validation, test

lst = []
tot_train = []
tot_validation = []
tot_test = []
for r in names_grp['names']:
    train, test = train_test_split(r, test_size=0.3, random_state=42)
    train, validation = train_test_split(train, test_size=0.5, random_state=42)
    lst += [[train, validation, test]]
    tot_train       += train
    tot_validation  += validation
    tot_test        += test

    # print(len(r)/3)
    # print(len(train))
    # print(len(validation))
    # print(len(test))
    # print()
df = pd.DataFrame(lst, columns = ['train', 'validation', 'test'], index = names_grp['labels']) 
# df.index = names_grp['labels']
df.to_csv('data/bias_analysis/yelp/input_sentences/names_ratio.csv')

input_questions = pd.read_csv('data/bias_analysis/yelp/input_sentences/save/names.csv')

train      = input_questions[input_questions.example_label.isin(tot_train)].reset_index(drop=True)
validation = input_questions[input_questions.example_label.isin(tot_validation)].reset_index(drop=True)
test       = input_questions[input_questions.example_label.isin(tot_test)].reset_index(drop=True)

# print(train)

train.to_csv('data/bias_analysis/yelp/input_sentences/names_train.csv')
validation.to_csv('data/bias_analysis/yelp/input_sentences/names_validation.csv')
test.to_csv('data/bias_analysis/yelp/input_sentences/names_test.csv')


