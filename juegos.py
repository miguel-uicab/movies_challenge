#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


tags = pickle.load(open('tag.csv', 'rb'))

data_train_no_duplicates = pickle.load(open('rating_train_whithout_duplicate.sav',
                                            'rb'))
data_train_no_duplicates.loc[6985774, 'time_day']
tag['datetime'] = pd.to_datetime(tag['timestamp'])


tag['userId-movieId'] = tag['userId'].astype(str) + '-' + tag['movieId'].astype(str)
group_tags = (tag.groupby(by=['userId-movieId'])
                 .size()
                 .reset_index(name='counts'))
group_tags.sort_values(by='counts',
                       ascending=False,
                       inplace=True)
group_tags.reset_index(drop=True,
                       inplace=True)
group_tags
group_tags.head(10)

group_tags = list(tag['userId-movieId'].unique())



def aggregate_tag(data=None, group=None):
    output_ind = pd.DataFrame(columns=['userId', 'movieId',
                                       'tag_tokens', 'length_tag_tokens',
                                       'tag_timestamp_min', 'tag_timestamp_max',
                                       'userId-movieId'],
                              index=[0]).replace({np.nan: None})
    data_ind = data[data['userId-movieId'] == group]
    data_ind.reset_index(drop=True, inplace=True)
    output_ind['userId'] = data_ind.loc[0, 'userId']
    output_ind['movieId'] = data_ind.loc[0, 'movieId']
    output_ind['userId-movieId'] = data_ind.loc[0, 'userId-movieId']
    output_ind['tag_timestamp_max'] = max(data_ind['datetime'])
    output_ind['tag_timestamp_min'] = min(data_ind['datetime'])
    lista_c = list(data_ind['tag'])
    list_append = []
    for name in lista_c:
        list_append.append(clean_string(string=name))
        # print('WORD: ', name)
        # print(clean_string(string=name))
    list_set = set().union(*list_append)
    list_set = list(list_set)
    processed_words = [word for word in list_set if word not in
                       stopwords.words('english') and word.isalpha()]
    output_ind['length_tag_tokens'] = len(processed_words)
    output_ind['tag_tokens'] = pd.Series([processed_words],
                                          index=output_ind.index)
    return output_ind

aggregate_tag(data=tag, group='58612-34405')



output = pd.DataFrame(columns=['userId', 'movieId',
                               'tag_tokens', 'length_tag_tokens',
                               'tag_timestamp_min', 'tag_timestamp_max',
                               'userId-movieId'])
 # list(group_tags['userId-movieId'])
 # ['910-5832', '910-5882']
for group in group_tags:
    output_ind = aggregate_tag(data=tag, group=group)
    output = output.append(output_ind,
                           ignore_index=True)





for group in group_tags:
    output_ind = pd.DataFrame(columns=['userId', 'movieId', 'tag_tokens',
                                       'tag_timestamp_min', 'tag_timestamp_max',
                                       'userId-movieId'],
                              index=[0]).replace({np.nan: None})
    data_ind = tag[tag['userId-movieId'] == '58612-34405']
    data_ind.reset_index(drop=True, inplace=True)
    output_ind['userId'] = data_ind.loc[0, 'userId']
    output_ind['movieId'] = data_ind.loc[0, 'movieId']
    output_ind['userId-movieId'] = data_ind.loc[0, 'userId-movieId']
    output_ind['tag_timestamp_max'] = max(data_ind['datetime'])
    output_ind['tag_timestamp_min'] = min(data_ind['datetime'])
    lista_c = list(data_ind['tag'])
    list_append = []
    for name in lista_c:
        list_append.append(clean_string(string=name))
        # print('WORD: ', name)
        # print(clean_string(string=name))
    list_set = set().union(*list_append)
    list_set = list(list_set)
    processed_words = [word for word in list_set if word not in
                       stopwords.words('english') and word.isalpha()]
    output_ind['tag_tokens'] = pd.Series([processed_words],
                                          index=output_ind.index)




len(output_ind['tag_tokens'])
len(list_set)
len(processed_words)
set(list_set)-set(processed_words)









# La última película que ha reseñado

rating.rename(columns={'timestamp': 'timestamp_rating'},
              inplace=True)
tag.rename(columns={'timestamp': 'timestamp_tag'},
           inplace=True)
rating_tag = pd.merge(rating,
                      tag,
                      how="left",
                      on=["userId", "movieId"])

counts = rating_tag.groupby(by=["userId", "movieId"]).size().reset_index(name='counts')
counts.sort_values(by='counts', ascending=False)

rating_tag_ind = rating_tag[(rating_tag.userId==130827) & (rating_tag.movieId==2318)]
rating_tag_ind





rating_tag[(rating_tag.userId==130827)]





len(rating['movieId'].unique())
27278 - 26744

info_movies = rating['movieId'].value_counts().to_frame().reset_index()
info_movies.columns = ['movieId', 'count']
info_movies[info_movies['count'] == 10]    #





###############################################################################
def info_user(data=None):
    total_user = data.userId.value_counts().to_frame('count').reset_index()
    total_user.columns = ['userId', 'count']
    print('Descriptivos de las frecuencias de revisiones de películas hechas '
          'por los usuarios.')
    return total_user['count'].describe()

info_user(data=rating)
rating.userId.value_counts().to_frame('count').reset_index()



###############################################################################




pd_individual = rating[rating.userId==118205]
pd_individual.sort_values(by='timestamp', ascending=False)

rating[rating.userId==1].sort_values(by='timestamp', ascending=False)
rating.loc[0, 'timestamp']

'2005-04-02 23:53:47'.split()[0]
