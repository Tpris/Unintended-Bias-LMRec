
import argparse
import os
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from scipy import stats
import warnings

from config.configs import city_list
from utils.bias_analysis_utils import get_price_ratio_df
warnings.filterwarnings("ignore")

nltk.download('stopwords')
stopwords = stopwords.words('english')
sns.set_style("darkgrid")
# sns.set_theme(style="ticks", color_codes=True)


def get_ratio_per_price_range(df_avg_city_plot, temp_df_names, label_name, bias_type):
    # you decide the sample number here
    if p.sample_number == 0:
        sample_number = len(temp_df_names[temp_df_names['label'] == label_name].example_label.unique())
    # calculate error bar for the averaged data
    summarize_df = df_avg_city_plot[df_avg_city_plot['bias'] == bias_type]
    summarize_df = summarize_df.groupby(['label', 'price_lvl'])['ratio'].mean().reset_index(name='mean_ratio')
    print(summarize_df)

    # calculate the sample standard deviation
    for idx, row in summarize_df.iterrows():
        cur_label = row['label']
        cur_priceLvl = row['price_lvl']
        mean_ratio = row['mean_ratio']

        # sample values
        city_x = df_avg_city_plot[(df_avg_city_plot['label'] == cur_label) &
                                (df_avg_city_plot['price_lvl'] == cur_priceLvl)]['ratio']
        std = (sum([(x - mean_ratio) ** 2 for x in city_x]) / sample_number) ** 0.5
        summarize_df.loc[(summarize_df['price_lvl'] == cur_priceLvl) & (summarize_df['label'] == cur_label), ['lb', 'ub']] = std

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.6))
    axes = axes.flatten()

    # loop through each city to create the plot
    for c_idx, c in enumerate(df_avg_city_plot.city.unique()):
        g = sns.barplot(x="price_lvl", y="ratio", hue='label',
                        data=df_avg_city_plot[(df_avg_city_plot['bias'] == bias_type) & (df_avg_city_plot['city'] == c)], ax=axes[c_idx])
        # Neutralization line
        g.axhline(0.5, ls='--', c='gray', label='neutral reference')

        f = sns.pointplot(x="price_lvl", y="ratio", hue='label',
                        data=df_avg_city_plot[(df_avg_city_plot['bias'] == bias_type) & (df_avg_city_plot['city'] == c)], ax=axes[c_idx],
                        legend=False)

        # calculate slope and correlation coefficient
        cur_city = df_avg_city_plot[(df_avg_city_plot['city'] == c) & (df_avg_city_plot['bias'] == bias_type)]
        cur_city.to_csv(p.saveFigure_dir + c + '_'+label_name+'B_price_lvl.csv')
        label_ratio = cur_city[cur_city['label'] == label_name][['price_lvl','ratio']]
        label_ratio.index = label_ratio.price_lvl
        label_ratio = label_ratio.ratio
        price_lvl = cur_city.price_lvl.unique()
        for p_lvl in price_lvl:
            if not p_lvl in label_ratio.index:
                label_ratio[p_lvl] = 0
        price_lvl = cur_city.price_lvl.unique()
        res = stats.linregress(price_lvl, label_ratio)

        axes[c_idx].set_title(c + '\n', fontsize=13)
        g.set(xlabel='', ylabel='', ylim=(0.1, None))
        g.legend_.set_title(None)
        g.legend_.set_visible(False)

    # calculate pearson correlation for all
    label_ratio = summarize_df[summarize_df['label'] == label_name][['price_lvl','mean_ratio']]
    label_ratio.index = label_ratio.price_lvl
    label_ratio = label_ratio.mean_ratio
    price_lvl = summarize_df.price_lvl.unique()
    for p_lvl in price_lvl:
        if not p_lvl in label_ratio.index:
            label_ratio[p_lvl] = 0
    res = stats.linregress(price_lvl, label_ratio)

    # plot average over cities
    errLo = summarize_df.pivot(index="price_lvl", columns="label", values="lb")
    errHi = summarize_df.pivot(index="price_lvl", columns="label", values="ub")
    err = []
    for col in errLo:
        err.append([errLo[col].values, errHi[col].values])
    err = np.abs(err)
    plot_df = summarize_df.pivot(index="price_lvl", columns="label", values="mean_ratio")
    ax = plot_df.plot(kind='bar', yerr=err, ax=axes[c_idx + 1], width=0.9, xlabel='', ylabel='')
    ax.set_title("All", fontdict={'fontsize': 13})
    ax.set_ylim(0.1)
    ax.axhline(0.5, color="gray", linestyle="--", label='neutral reference')

    plt.legend(loc='upper center', bbox_to_anchor=[0.5, 0.99], ncol=4,
            bbox_transform=plt.gcf().transFigure, fontsize=13)
    fig.text(0.06, 0.35, '% at price level recommended', rotation='vertical', fontsize=16)
    fig.text(0.08, 0.5, 'to '+label_name, rotation='vertical', fontsize=16)
    fig.text(0.46, 0.07, 'price level ($)', fontsize=16)

    if p.save_figure:
        temp_file_path = p.saveFigure_dir + 'price_level_ratio_All_Cities_'+bias_type+'.pdf'
        print('Saving figure to', temp_file_path)
        plt.savefig(temp_file_path, bbox_inches='tight')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Yelp_cities')
    parser.add_argument('--statistics_df_dir', type=str, default='data/bias_analysis/yelp/statistics.csv', help='the csv file that stores the '
                                                                                                                   'datset statistics, alternative '
                                                                                                                   'path for gender: '
                                                                                                                   'gender_statistics.csv')
    parser.add_argument('--city_name', type=str, default='Atlanta')
    parser.add_argument('--save_figure', action='store_true', help='whether to save the figures')
    parser.add_argument('--test_neutralization', action='store_true', help='whether to perform test-side neutralization towards the results')
    parser.add_argument('--rank_threshold_all', type=int, default=20, help='number of top recommendations to analyse')
    parser.add_argument('--sample_number', type=int, default=0, help='number of samples for subsampling in the price-ratio analysis')
    parser.add_argument('--prob_choice_2d_plot', type=str, default='difference', help="probability choice for creating the 2d categories plot, "
                                                                                      "alternatively can use 'ratio'")
    parser.add_argument('--bias_placeholder_dir', type=str, default='data/bias_analysis/yelp/output_dataframes/{}_output_dataframes_2train_LR-10/')
    parser.add_argument('--neutralizedBias_placeholder_dir', type=str,
                        default='data/bias_analysis/yelp/output_dataframes/{}_output_dataframes_Neutralized/')
    parser.add_argument('--saveFigure_dir', type=str, default='bias_analysis/yelp/figures/')
    p = parser.parse_args()

    p.save_figure = True
    p.test_neutralization = False

    # p.bias_placeholder_dir = 'data/bias_analysis/yelp/output_dataframes/save/Atlanta_output_dataframes_base/'

    """
    Settings
    """

    print("City list:", city_list)
    if not os.path.exists(p.saveFigure_dir):
        os.makedirs(p.saveFigure_dir)


    print('-' * 20 + 'Generating bar charts for price ratio' + '-' * 20)
    
    df_list = []
    for cityName in city_list:
        origin_cityName = cityName
        if '_' in cityName:
            cityName, subExp = cityName.split('_')

        # get reading directory
        temp_bias_dir = p.bias_placeholder_dir.format(cityName)
        temp_url = temp_bias_dir + 'yelp_qa_names.csv'
        temp_df_names = pd.read_csv(temp_url, index_col=0)
        if cityName == 'Toronto':
            temp_df_names['price_lvl'] = temp_df_names['price'].str.len()
        else:
            temp_df_names['price_lvl'] = temp_df_names['price']

        temp_df_price_ratio_plot = get_price_ratio_df(temp_df_names)
        temp_df_price_ratio_plot['city'] = origin_cityName
        df_list.append(temp_df_price_ratio_plot)

    # concatenate all dataframes
    df_avg_city_plot = pd.concat(df_list, ignore_index=True)

    get_ratio_per_price_range(df_avg_city_plot, temp_df_names, 'male', 'gender')
    get_ratio_per_price_range(df_avg_city_plot, temp_df_names, 'white', 'race')
