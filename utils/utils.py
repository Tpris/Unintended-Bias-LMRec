import re
import os
import pandas as pd
import pickle
import numpy as np
# import tensorflow as tf
# from tokenizers import BertWordPieceTokenizer
# from tensorflow 
# import keras
# from keras import layers
from transformers import BertModel, BertTokenizer
from torch import nn
import torch


def pickle_load(path):
    with open(path, 'rb') as fp:
        file = pickle.load(fp)
    return file


def pickle_dump(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def get_latex_table_from_stats_df(df, save_dir):
    print_df = df.copy()
    print_df['bias'] = print_df['bias'].apply(lambda x: x.capitalize())
    # print_df['price_lvl'] = print_df['price_lvl'].apply(lambda x: '$' * x)
    print_df['label'] = print_df['label'].apply(lambda x: x.capitalize() + '%')
    print_df['percent_stats'] = print_df['percent_stats'].apply(lambda x: round(100 * x, 2))

    print_df = print_df.pivot(index=['bias', 'price_lvl', 'label'], columns=['city'], values=['percent_stats'])
    print_df.columns = print_df.columns.to_series().str.join('_')
    print_df.columns = [name.replace('percent_stats_', '') for name in print_df.columns]
    print_df.index = print_df.index.set_names(names=['', '', ''])
    with open(save_dir, 'w') as tf:
        tf.write(print_df.style.to_latex())


def concat_city_df(city_list, bias_placeholder_dir, file_name):
    df_list_total = []
    for cityName in city_list:
        temp_bias_dir = bias_placeholder_dir.format(cityName)
        temp_url = temp_bias_dir + file_name
        temp_df_names = pd.read_csv(temp_url, index_col=0)
        if cityName == 'Toronto':
            temp_df_names['price_lvl'] = temp_df_names['price'].str.len()
        else:
            temp_df_names['price_lvl'] = temp_df_names['price']
        temp_df_names['city'] = cityName
        df_list_total.append(temp_df_names)
    df_city_all_plot = pd.concat(df_list_total, ignore_index=True)
    return df_city_all_plot


def split_cat(old_df):
    '''
      splits out the categories from a result dataframe and retur a new dataframe
      for yelp
    '''

    # create a new dataframe that splits the categories
    old_df.head()
    cat_replacements = {'&': ',',
                        '/': ',',
                        '(': ',',
                        ')': ''}

    new_df = pd.DataFrame(columns=old_df.columns)

    for index, row in old_df.iterrows():
        cats_string = row['categories']
        text = multireplace(cats_string, cat_replacements)
        cats_list = text.lower().split(',')
        for cat in cats_list:
            row_copy = row.copy()
            row_copy['categories'] = cat.strip()
            new_df = new_df.append(row_copy)

    return new_df


def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str

    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against  the string 'hey abc', it should produce 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


# convert input text phrase to tokens and attention mask
def get_input_list(review_list, max_len, device):
    slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    save_path = "bert_base_uncased/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wrapped_input = tokenizer(review_list, max_length=512, add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt").to(device)
    return wrapped_input

