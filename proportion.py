import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "bias_analysis/yelp/figures/"
df_eth = pd.read_csv(path+"Atlanta_raceB_price_lvl.csv")[['label', 'price_lvl', 'counts']]
df_gender = pd.read_csv(path+"Atlanta_genderB_price_lvl.csv")[['label', 'price_lvl', 'counts']]

def get_values_per_price_range(df,label):
  y = df.loc[df['label'] == label][['counts', 'price_lvl']]
  y.index = y.price_lvl
  y = y.counts
  price_lvl = df.price_lvl.unique()
  for xi in price_lvl:
    if not xi in y.index:
      y[xi] = 0
  return y

def plot_counts(df,label1,label2,title, name_saved):
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(df.price_lvl.unique()))
    bar_width = 0.4
    y1 = get_values_per_price_range(df, label1)
    y2 = get_values_per_price_range(df, label2)
    b1 = ax.bar(x, y1, width=bar_width, label=label1)
    b2 = ax.bar(x + bar_width, y2, width=bar_width, label=label2)
    
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(df.price_lvl.unique())
    
    for bar in ax.patches:
      bar_value = bar.get_height()
      text = f'{bar_value:,}'
      text_x = bar.get_x() + bar.get_width() / 2
      text_y = bar.get_y() + bar_value
      bar_color = bar.get_facecolor()
      ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
              size=12)
    
    ax.legend()
    ax.set_xlabel('Price level', labelpad=15)
    ax.set_ylabel('Number of predictions', labelpad=15)
    ax.set_title(title, pad=15)
    fig.savefig(path+name_saved)

plot_counts(df_eth,'black','white','Number of predictions by etnic group for price levels', "Atlanta_raceB_price_lvl_plot.png")
plot_counts(df_gender,'female','male','Number of predictions by gender for price levels', "Atlanta_genderB_price_lvl_plot.png")

