#!/usr/bin/env python
# coding: utf-8

# # Imports
import os
import re
# os.chdir(f'{os.getcwd()}/../movies_challenge')
import pandas as pd
import numpy as np
import pickle
from functools import reduce
from extras import get_config
config = get_config()

# Data de entrenamiento completa #####################################################
data_train = pickle.load(open('rating_redundant_train_2.sav',
                              'rb'))

# Data de entramiento filtrada sin "duplicadas" ######################################
data_train_no_duplicates = pickle.load(open('rating_train_whithout_duplicate.sav',
                                            'rb'))

# Cargar las cluster encontrados en las películas ####################################
movie_with_cluster = pickle.load(open('movie_with_cluster.sav',
                                      'rb'))
# data_train_no_duplicates[data_train_no_duplicates['userId']==85252]

# Se comenzará la construcción con la data completa merge las peli con cluster #######
# data_train_and_tag = pd.merge(data_train,
#                               tag_info,
#                               how="left",
#                               on=["userId", "movieId"])

data_train_and_cluster = pd.merge(data_train,
                                  movie_with_cluster,
                                  how="left",
                                  on=["movieId"])
data_train_and_cluster.reset_index(drop=True,
                                   inplace=True)
data_train_and_cluster['movies'] = 1

data_train_and_cluster['userId'].value_counts()


# Tomar un ejemplo individual ######################################################
userid=8405
data_ind = data_train_and_cluster[data_train_and_cluster['userId']==userid]

names_interest_cum = ['time_day', 'movies', 'genre_crime',
                      'genre_no genres listed', 'genre_adventure',
                      'genre_romance', 'genre_horror', 'genre_drama',
                      'genre_thriller', 'genre_comedy', 'genre_imax',
                      'genre_documentary', 'genre_action', 'genre_musical',
                      'genre_sci-fi', 'genre_animation', 'genre_fantasy',
                      'genre_children', 'genre_western', 'genre_war',
                      'genre_mystery', 'genre_film-noir']

info_movies_ind = data_ind[names_interest_cum]

# Se colpasa la información por días ##########
info_cumulative = (info_movies_ind.groupby(['time_day'])
                                  .sum().groupby(level=0)
                                  .cumsum()
                                  .reset_index())
new_names_interest_cum = []
names_interest_cum.remove('time_day')
for name in names_interest_cum:
    new_name = f"frecc_{name}_per_day"
    new_names_interest_cum.append(new_name)
    info_cumulative.rename(columns={name: new_name}, inplace=True)


# Se tienen cantidades acumuladas ##############
cumulative_names = []
for name in new_names_interest_cum:
    cumulative_name = f'cumulative{name}'
    cumulative_names.append(cumulative_name)
    info_cumulative[cumulative_name] = info_cumulative[name].cumsum()

# Porcentaje de representación del género en el acumulado de películas ########
cumulative_names.remove('cumulativefrecc_movies_per_day')
percen_cumulative_names = []
for name in cumulative_names:
    percen_cumulative_name = f'percen{name}'
    percen_cumulative_names.append(percen_cumulative_name)
    info_cumulative[percen_cumulative_name] = info_cumulative[name] / info_cumulative['cumulativefrecc_movies_per_day']

info_cumulative.sort_values(by='time_day',
                            ascending=False,
                            inplace=True)
info_cumulative.reset_index(drop=True, inplace=True)
date_column = info_cumulative[['time_day']]
info_cumulative.rename(columns={'time_day': 'last_time_day'},
                       inplace=True)

info_cumulative.drop(0,
                     axis=0,
                     inplace=True)

info_cumulative.reset_index(drop=True, inplace=True)

past_info = pd.merge(date_column,
                     info_cumulative,
                     left_index=True,
                     right_index=True,
                     how="outer")

past_info['userId'] = userid

###############################################################################
data_train_no_duplicates
data_train_no_duplicates[data_train_no_duplicates['userId']==userid]


data_train_no_duplicates_info_past =  pd.merge(data_train_no_duplicates,
                                               past_info,
                                               how="left",
                                               on=["userId", "time_day"])

data_train_no_duplicates_info_past.isna().sum()

1002389
data_train_no_duplicates_info_past[data_train_no_duplicates_info_past['userId']==userid]

# data_except_current_month = data_interesting[data_interesting['month_date_in']<date(date.today().year, date.today().month, 1)]
