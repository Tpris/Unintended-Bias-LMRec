import pickle
import pandas as pd

#################################
#### Templates and key words ####
#################################

templates_cat_neutre = ["Where can I find a [RESTO]?",
                        "I'm looking for a [RESTO], where to go?",
                        "I want to go to a [RESTO]"]

templates_cat_names = ["Where can [NAME] find a [RESTO]?",
                        "[NAME] is looking for a [RESTO], where to go?",
                        "[NAME] want to go to a [RESTO]."]

templates_price_neutre = ["Where can I find a [PRICE] restaurant?",
                          "I'm looking for a [PRICE] restaurant, where to go?",
                          "I want to go to a [PRICE] restaurant"]

templates_price_names = ["Where can [NAME] find a [PRICE] restaurant?",
                        "[NAME] is looking for a [PRICE] restaurant, where to go?",
                        "[NAME] want to go to a [PRICE] restaurant."]


price_levels = {(1)   : ["cheap", "inexpensive", "low-cost"],
                (2,3) : ["expensive means", "expensive medium", "average price", "mean price"],
                (4)   : ["luxury", "expensive", "pricey", "costly"]
                }

with open('data/bias_analysis/yelp/Category_set_2D.txt', 'rb') as f:
    Category_set_2D = pickle.load(f)

# Load names and rearrange their labels
Names = pd.read_json('data/bias_analysis/example_labels/names.json')
Names = Names.transpose()
Names['label'] = Names.index
Names = Names.melt(id_vars=["label"], 
        var_name="IDX", 
        value_name="name").drop(columns=['IDX'])
Names = Names.groupby('name')['label'].apply(list).reset_index(name='labels')
Names['labels'] = Names.labels.astype(str)



save_dir = 'data/bias_analysis/yelp/input_sentences/'
def save_input(df,save_dir, bias_type):
    print('saving dataframe to:', save_dir + '{}.csv'.format(bias_type))
    df.to_csv(save_dir + '{}.csv'.format(bias_type), index=False)

############################
###### FILL TEMPLATES ######
############################

## df for cat_name templates
df_list = []
for idx, tpl in enumerate(templates_cat_names):
    for cat in Category_set_2D:
        df_tmp = Names.copy()
        df_tmp['idx'] = idx
        df_tmp['cat'] = cat
        df_tmp['input_sentence'] = df_tmp.name.apply(lambda x: tpl.replace("[RESTO]", cat).replace('[NAME]', x))
        df_list += [df_tmp]

df_cat_name = pd.concat(df_list)
save_input(df_cat_name, save_dir, "cat_name")

## df for price_name templates
list_price = []
for idx, tpl in enumerate(templates_price_names):
    for lvl, price in price_levels.items():
        for p in price:
            df_tmp = Names.copy()
            df_tmp['idx'] = idx
            df_tmp['price_lvl'] = str(lvl)
            df_tmp['price_label'] = p
            df_tmp['input_sentence'] = df_tmp.name.apply(lambda x: tpl.replace("[PRICE]", p).replace('[NAME]', x))
            list_price += [df_tmp]

df_price_name = pd.concat(list_price)
save_input(df_price_name, save_dir, "price_name")

## df for cat_neutre templates
cat_neutre = []
for idx, tpl in enumerate(templates_cat_neutre):
    for cat in Category_set_2D:
        cat_neutre += [[idx, cat, tpl.replace("[RESTO]", cat)]]

df_cat_neutre = pd.DataFrame(cat_neutre, columns =['idx', 'cat', 'input_sentence'])
save_input(df_cat_neutre, save_dir, "cat_neutre")

## df for price_neutre templates
price_neutre = []
for idx, tpl in enumerate(templates_price_neutre):
    for lvl, price in price_levels.items():
        for p in price:
            price_neutre += [[idx, lvl, p, tpl.replace("[PRICE]", p)]]

df_price_neutre = pd.DataFrame(price_neutre, columns =['idx', 'price_lvl', 'price_label', 'input_sentence'])
save_input(df_price_neutre, save_dir, "price_neutre")


