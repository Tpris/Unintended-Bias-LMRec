# import pickle

# with open('data/bias_analysis/yelp/Category_set_2D.txt', 'rb') as f:
#     Category_set_2D = pickle.load(f)
# for name in Category_set_2D:  # USE THIS FOR CLEANER VERSION
#     print(name)



import pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io._orca
import retrying
import seaborn as sns
from nltk.corpus import stopwords
from tqdm import tqdm
import warnings
from scipy.spatial import distance
import plotly.express as px

from config.configs import city_list, replacements, nightlife_list
from utils.bias_analysis_utils import  get_counts_wFold, mean_confidence_interval
from utils.utils import concat_city_df, multireplace
warnings.filterwarnings("ignore")

nltk.download('stopwords')
stopwords = stopwords.words('english')
sns.set_style("darkgrid")

prob_choice_2d_plot = 'difference'
bias_placeholder_dir_base = 'data/bias_analysis/yelp/output_dataframes/save/Atlanta_output_dataframes_base/'
bias_placeholder_dir_mit = 'data/bias_analysis/yelp/output_dataframes/save/Atlanta_output_dataframes_biased/'
saveFigure_dir = 'bias_analysis/yelp/figures/'
name_fig_base = 'all_categories_22D_base'
name_fig_mit = 'all_categories_22D_mit'

"""3.The 2D plot for association score"""
print('-' * 20 + '3/6: Generating 2D plot for association scores' + '-' * 20)

def association_score(bias_placeholder_dir, name_figure):
    fold_number = 20
    split_number = 4

    # load all cities into one dataframe
    df_city_all_plot = concat_city_df(city_list, bias_placeholder_dir, file_name='yelp_qa_names.csv')

    # indicates how many front recommended items we want to check
    rank_bar = 20
    np.random.seed(68)

    # df_city_all_plot = df_city_all_plot[df_city_all_plot['rank'] <= rank_bar]
    # df_city_all_plot['fold'] = np.random.randint(0, fold_number, df_city_all_plot.shape[0])
    # df_cat_all = df_city_all_plot.copy()
    # df_sum_all = df_city_all_plot.groupby(by=['label', 'fold']).size().reset_index(name='counts')

    # print(df_city_all_plot)

    df_cat_all = None
    df_sum_all = None

    for split_num in range(split_number):
        df_temp = df_city_all_plot[df_city_all_plot['rank'] <= rank_bar]
        df_temp['fold'] = np.random.randint(0, fold_number, df_temp.shape[0])

        if df_cat_all is None:
            df_cat_all = df_temp.copy()
        else:
            df_cat_all = pd.concat([df_cat_all, df_temp])

        df_temp_sum = df_temp.groupby(by=['label', 'fold']).size().reset_index(name='counts')

        if df_sum_all is None:
            df_sum_all = df_temp_sum.copy()
        else:
            df_sum_all = pd.concat([df_sum_all, df_temp_sum])
            # df_sum_all = pd.concat(df_sum_all,df_temp_sum)

    # nested dictionary
    # {cat1: {'fold0': {'female':0, 'male':0, 'white':0,'black':0}, 'fold1':{}...}, cat2:{}...}
    jointCount_dict = dict()
    for index, row in tqdm(df_cat_all[['categories', 'label', 'fold']].iterrows()):
        cats_string = row['categories']
        text = multireplace(cats_string, replacements)
        cats_list = text.lower().split(',')
        for cat in cats_list:
            cat = cat.strip()
            label = row['label']
            fold_n = row['fold']

            innerDict = jointCount_dict.get(cat, dict())
            mostInnerDict = innerDict.get(fold_n, dict())
            mostInnerDict[label] = mostInnerDict.get(label, 0) + 1
            innerDict[fold_n] = mostInnerDict
            jointCount_dict[cat] = innerDict

    plot_df_all = pd.DataFrame(columns=['category', 'score', 'err', 'kind', 'bottom', 'top'])

    # use the difference
    point_list = []

    with open('data/bias_analysis/yelp/Category_set_2D.txt', 'rb') as f:
        Category_set_2D = pickle.load(f)
    for name in Category_set_2D:  # USE THIS FOR CLEANER VERSION
        xVal_list = list()
        yVal_list = list()

        # loop through each fold
        for fold_num in range(fold_number):
            # get joint count
            # white
            x_white, x_white_count, x_white_baseCount = get_counts_wFold(df_sum_all, jointCount_dict,  name, label='white', fold_num=fold_num)
            # black
            x_black, x_black_count, x_black_baseCount = get_counts_wFold(df_sum_all, jointCount_dict, name, label='black', fold_num=fold_num)
            # male
            y_male, y_male_count, y_male_baseCount = get_counts_wFold(df_sum_all, jointCount_dict, name, label='male', fold_num=fold_num)
            # female
            y_female, y_female_count, y_female_baseCount = get_counts_wFold(df_sum_all, jointCount_dict, name, label='female', fold_num=fold_num)

            if prob_choice_2d_plot == 'ratio':
                x_val_temp = (1e-4 + x_white) / (x_black + 1e-4)
                y_val_temp = (1e-4 + y_female) / (y_male + 1e-4)
            elif prob_choice_2d_plot == 'difference':
                x_val_temp = (x_white - x_black) / (1e-4 + (x_white_count + x_black_count) / (1e-4 + x_white_baseCount + x_black_baseCount))
                y_val_temp = (y_female - y_male) / (1e-4 + (y_female_count + y_male_count) / (1e-4 + y_male_baseCount + y_female_baseCount))
            else:
                raise NotImplementedError('need to choose a valid probability type')
            xVal_list.append(x_val_temp)
            yVal_list.append(y_val_temp)

        # calculate the aggregated value and error bars
        x_val, x_err = mean_confidence_interval(xVal_list, confidence=0.90, n=len(xVal_list) * split_number)
        y_val, y_err = mean_confidence_interval(yVal_list, confidence=0.90, n=len(xVal_list) * split_number)

        point_list.append([name, x_val, y_val, x_err, y_err])
        plot_df_all = pd.concat([plot_df_all, pd.DataFrame([{'category': name,
                                            'score': x_val,
                                            'err': x_err,
                                            'kind': 'race',
                                            'bottom': 'black',
                                            'top': 'white'}])])

        plot_df_all = pd.concat([plot_df_all, pd.DataFrame([{'category': name,
                                            'score': y_val,
                                            'err': y_err,
                                            'kind': 'gender',
                                            'bottom': 'male',
                                            'top': 'female'}])])
        # texts.append(plt.text(x_val, y_val, name))

    # For saving the image.
    unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
    wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
    plotly.io._orca.request_image_with_retrying = wrapped

    """showing for neutralized data"""
    fig = go.Figure()
    short_dict = {'wine bars': 'wine'}
    location_dict = {'juice bars & smoothies': 'top center', 'mediterranean': 'bottom right', 'desserts': 'top center', 'cafes': 'top center',
                        'fish & chips': 'bottom right', 'pizza': 'top right', 'canadian (new)': 'bottom right', 'gastropubs': 'bottom right',
                        'argentine': 'bottom right', 'brewpubs': 'bottom right', 'smokehouse': 'bottom right', 'whiskey bars': 'bottom right',
                        'italian': 'bottom right', 'tapas bars': 'bottom right',
                        'burgers': 'bottom right', 'mauritius': 'bottom left', 'chicken wings': 'bottom left',
                        'jazz & blues': 'top center', 'breakfast & brunch': 'top right', 'filipino': 'bottom left',
                        'wine bars': 'bottom right', 'austrian': 'bottom left',
                        'salad': 'bottom right', 'irish pub': 'bottom left', 'arabian': 'bottom right', 'patisserie': 'bottom right',
                        'cideries': 'bottom left', 'food court': 'top right', 'puerto rican': 'bottom right',
                        'coffee roasteries': 'top left', 'waffles': 'bottom left',
                        'coffee & tea': 'bottom right', 'cocktail bars': 'top left', 'shaved ice': 'bottom right',
                        'nepalese': 'bottom right', 'wine': 'top left', 'south african': 'top left', 'champagne bars': 'bottom left'}

    for item in point_list:
        n, v1, v2, err1, err2 = item
        if err1 >= 0.1 or err2 >= 0.1: continue
        if n in ['catalan', 'bars', 'persian', 'fischbroetchen', 'british', 'dive bars',
                    'wineries', 'pretzels', 'mauritius', 'sandwiches', 'beer bar', 'cider',
                    'meaderies', 'sports clubs', 'puerto rican']:
            continue
        if n in short_dict.keys():
            n = short_dict[n]

        fig.add_trace(go.Scatter(
            x=[v1],
            y=[v2],
            mode="markers+text",
            name=n,
            text=[n],
            marker=dict(size=8),
            textposition="bottom center" if n not in location_dict else location_dict[n],
            showlegend=False,
            error_y=dict(
                type='data',
                array=[err2],
                visible=True),
            error_x=dict(
                type='data',
                array=[err1],
                visible=True)

        ))
    # add neutralization point
    # opacity_list = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
    # for idx, op in enumerate(opacity_list):
    #     fig.add_trace(go.Scatter(
    #         x=[0],
    #         y=[0],
    #         opacity=op,
    #         mode="markers",
    #         name='neutralization',
    #         text=[''],
    #         marker=dict(size=idx * 20, color='gray'),
    #         textposition="bottom center",
    #         showlegend=False,
    #     ))

    fig.update_layout(
        autosize=True,
        height=600,
        width=1800,
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        xaxis=dict(
            title=dict(
                text=r"$ \Large\mathrm{black} \longleftarrow \longrightarrow \mathrm{white}$",
                font=dict(size=40))
            , range=[-1, 1]
        ),
        yaxis=dict(
            title=dict(
                text=r"$ \Large\mathrm{male} \longleftarrow \longrightarrow \mathrm{female}$",
                font=dict(size=40))
            , range=[-1, 1]
        ),
        font=dict(
            family="Times New Roman",
            size=24
        ))

    fig.write_image("{}{}.pdf".format(saveFigure_dir, name_figure), scale=1, width=1850, height=600)
    return point_list


base_point = association_score(bias_placeholder_dir_base, name_fig_base)
mit_point = association_score(bias_placeholder_dir_mit, name_fig_mit)

base_point = pd.DataFrame(base_point, columns=['name', 'x_base', 'y_base', 'x_std', 'y_std'])
mit_point = pd.DataFrame(mit_point, columns=['name', 'x_mit', 'y_mit', 'x_std', 'y_std'])

base_point = base_point.drop(['x_std', 'y_std'], axis=1).set_index('name')
mit_point = mit_point.drop(['x_std', 'y_std'], axis=1).set_index('name')

print(base_point)
print('='*20)
print(mit_point)
print('='*20)

merge = pd.concat([base_point, mit_point], axis=1)

merge['distance'] = merge.apply(lambda r: distance.euclidean((r.x_base,r.y_base),(r.x_mit,r.y_mit)), axis=1)
merge['distance_x'] = merge.apply(lambda r: distance.euclidean((r.x_base,),(r.x_mit,)), axis=1)
merge['distance_y'] = merge.apply(lambda r: distance.euclidean((r.y_base,),(r.y_mit,)), axis=1)
print(merge)
print('fast food\n', merge.loc['fast food'])

df00 = merge.drop(['x_base', 'y_base','x_mit', 'y_mit'], axis=1)
df00 = df00.sort_values(by=['distance'])
df00['name'] = df00.index

fig = px.line_polar(df00, r='distance', theta='name', line_close=True)
fig.update_layout(
    height=1500,
    width=1500,
    font=dict(
        size=8
    ))
fig.write_image("{}distance.pdf".format(saveFigure_dir), scale=1)

fig = px.line_polar(df00, r='distance_x', theta='name', line_close=True)
fig.update_layout(
    height=1500,
    width=1500,
    font=dict(
        size=8
    ))
fig.write_image("{}distance_x.pdf".format(saveFigure_dir), scale=1)

fig = px.line_polar(df00, r='distance_y', theta='name', line_close=True)
fig.update_layout(
    height=1500,
    width=1500,
    font=dict(
        size=8
    ))
fig.write_image("{}distance_y.pdf".format(saveFigure_dir), scale=1)
