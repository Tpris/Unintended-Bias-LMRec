import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ONLY_NAMES = False

def check_templates(row, sentences_part):
    sents = row['input_sentence'].split(row['example_label'])
    subsent = max(sents, key=len)
    return any(subsent in s for s in sentences_part)

def get_selected_sentences(input_questions, list_tpl):
    msk = input_questions.apply(lambda r: check_templates(r,list_tpl), axis=1)
    return input_questions[msk]

Names = pd.read_json('data/bias_analysis/example_labels/names.json')
Names = Names.transpose()
Names['label'] = Names.index
Names = Names.melt(id_vars=["label"], 
        var_name="IDX", 
        value_name="name").drop(columns=['IDX'])
Names = Names.groupby('name')['label'].apply(list).reset_index(name='labels')
Names['labels'] = Names.labels.astype(str)

names_grp = Names.groupby('labels')['name'].apply(list).reset_index(name='names')

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
df.to_csv('data/bias_analysis/yelp/input_sentences/names_ratio.csv')

input_questions = pd.read_csv('data/bias_analysis/yelp/input_sentences/save/names.csv')

train      = input_questions[input_questions.example_label.isin(tot_train)].reset_index(drop=True)
validation = input_questions[input_questions.example_label.isin(tot_validation)].reset_index(drop=True)
test       = input_questions[input_questions.example_label.isin(tot_test)].reset_index(drop=True)

if ONLY_NAMES:
    train.to_csv('data/bias_analysis/yelp/input_sentences/names_train.csv')
    validation.to_csv('data/bias_analysis/yelp/input_sentences/names_validation.csv')
    test.to_csv('data/bias_analysis/yelp/input_sentences/names_test.csv')
    exit()



Templates = pd.read_json('data/bias_analysis/templates/yelp_templates.json')
Templates = Templates['names'].to_list()

train_tpl, test_tpl = train_test_split(Templates, test_size=0.2, random_state=42)
train_tpl, validation_tpl = train_test_split(train_tpl, test_size=0.25, random_state=42)

# print(3*len(Templates)/5)
# print(len(Templates)/5)
# print(len(train))
# print(len(validation))
# print(len(test))
# print()

train_tpl_df = pd.DataFrame(train_tpl, columns = ['template']) 
train_tpl_df['part'] = 'Train'

val_tpl_df = pd.DataFrame(validation_tpl, columns = ['template']) 
val_tpl_df['part'] = 'Validation'

test_tpl_df = pd.DataFrame(test_tpl, columns = ['template']) 
test_tpl_df['part'] = 'Test'

templates_save = pd.concat([train_tpl_df,val_tpl_df,test_tpl_df])
templates_save.to_csv('data/bias_analysis/yelp/input_sentences/template_split.csv')
print(templates_save)

train_sentences = get_selected_sentences(train, train_tpl)
val_sentences   = get_selected_sentences(validation, validation_tpl)
test_sentences  = get_selected_sentences(test, test_tpl)

train_sentences.to_csv('data/bias_analysis/yelp/input_sentences/names_templates_train.csv')
val_sentences.to_csv('data/bias_analysis/yelp/input_sentences/names_templates_validation.csv')
test_sentences.to_csv('data/bias_analysis/yelp/input_sentences/names_templates_test.csv')




