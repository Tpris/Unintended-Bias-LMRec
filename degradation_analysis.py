import pandas as pd
import os

# path = 'data/bias_analysis/yelp/output_dataframes/'
path = 'data/bias_analysis/yelp/Atlanta_output_dataframes_test_degr_2train/'
path_save = "bias_analysis/yelp/figures/"
list_files = ["yelp_qa_cat_name.csv", 
              "yelp_qa_cat_neutre.csv",
              "yelp_qa_price_name.csv",
              "yelp_qa_price_neutre.csv"]

colToSee = {"price" : "price_lvl",
            "cat": "categories"}

def find(x_typeQ, colToSee):
    colToSee = colToSee.lower()
    x_typeQ = x_typeQ.lower()

    if colToSee in x_typeQ:
        return True
    return False

if not os.path.exists(path_save):
        os.makedirs(path_save)

for f in list_files:
    df = pd.read_csv(path+f)
    print(f)
    # Get only the first prediction
    df = df[df['rank']<=0]
    # Get the question type
    typeQ = f.split('_')[2]
    # Find if it is a neutral question
    isNeutre = f.split('_')[3]=="neutre.csv"
    print(typeQ)
    print(isNeutre)

    if typeQ=="price":
        df.price = df.price.astype(int).astype(str) # Remove decimal entries and then convert to string
    
    # Count correct anwser
    df['correct'] = df.apply(lambda x: find(x[colToSee[typeQ]], x[typeQ]), axis=1).astype(int)
    nb_rows = df.shape[0]
    count = df['correct'].sum()
    # Save the count
    df['count'] = str(count)+"/"+str(nb_rows)
    # Ratio calculation
    df['ratio'] = count/nb_rows
    df.to_csv(path_save+"RES_MIT"+f)
    
    print(count,"/",nb_rows)

    # Get counts grouped by labels
    if not isNeutre:
        df['row'] = 1
        df_count = df[['labels', 'correct', 'row']].groupby('labels').sum()
        df_count['labels'] = df_count.index
        df_count = df_count[['labels', 'correct', 'row']]
        df_count['labels'] = df_count['labels'].apply(lambda x: x.replace('[','').replace(']','').replace("'",""))
        df_count['ratio'] = df_count["correct"]/df_count['row']
        df_count.to_csv(path_save+"RES_COUNT_MIT"+f)

