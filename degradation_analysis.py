import pandas as pd

# path = 'data/bias_analysis/yelp/output_dataframes/'
path = 'data/bias_analysis/yelp/output_dataframes/Atlanta_output_dataframes_2train/'
path_save = "bias_analysis/yelp/figures/"
list_files = ["yelp_qa_cat_name.csv", 
              "yelp_qa_cat_neutre.csv",
              "yelp_qa_price_name.csv",
              "yelp_qa_price_neutre.csv"]

colToSee = {"price" : "price_lvl",
            "cat": "categories"}

def find(colToSee,typeQ):
    colToSee = colToSee.lower()
    typeQ = typeQ.lower()

    tab = colToSee.split()
    for w in tab:
        if w in typeQ:
            return True
    return False

for f in list_files:
    df = pd.read_csv(path+f)
    print(f)
    # df = df[df['rank']<=0]
    typeQ = f.split('_')[2]
    isNeutre = f.split('_')[3]=="neutre.csv"
    print(typeQ)
    print(isNeutre)
    if typeQ=="price":
        df.price = df.price.astype(int).astype(str)
    print(df)
    df['correct'] = df.apply(lambda x: find(x[colToSee[typeQ]], x[typeQ]), axis=1).astype(int)
    nb_rows = df.shape[0]
    count = df['correct'].sum()
    df['count'] = str(count)+"/"+str(nb_rows)
    df['ratio'] = count/nb_rows
    df.to_csv(path_save+"RES_MIT"+f)
    # print(df)
    print(count,"/",nb_rows)
    if not isNeutre:
        df['row'] = 1
        df_count = df[['labels', 'correct', 'row']].groupby('labels').sum()
        df_count['labels'] = df_count.index
        df_count = df_count[['labels', 'correct', 'row']]
        df_count['labels'] = df_count['labels'].apply(lambda x: x.replace('[','').replace(']','').replace("'",""))
        df_count['ratio'] = df_count["correct"]/df_count['row']
        df_count.to_csv(path_save+"RES_COUNT_MIT"+f)
        # print(df_count)

