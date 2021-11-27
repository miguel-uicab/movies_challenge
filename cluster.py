#!/usr/bin/env python
# coding: utf-8

# # Imports
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering
from EDA_functions import clean_string


def cluster_movies(data_movie=None, n_clusters=1000):
    data_movie['genres_list'] = list(map(clean_string,
                                    data_movie['genres']))
    list_genres = list(data_movie['genres_list'])
    total_genres = set().union(*list_genres)
    total_genres = list(total_genres)

    for genre in total_genres:
        data_movie[f'genre_{genre}'] = data_movie.apply(lambda row: 1 if genre in row['genres_list'] else 0,
                                              axis=1)

    interest_columns = ['movieId', 'genre_film-noir',
                        'genre_no genres listed', 'genre_drama',
                        'genre_mystery', 'genre_animation',
                        'genre_horror', 'genre_fantasy',
                        'genre_war', 'genre_crime', 'genre_comedy',
                        'genre_western', 'genre_adventure',
                        'genre_documentary', 'genre_imax',
                        'genre_action', 'genre_children',
                        'genre_musical', 'genre_thriller',
                        'genre_romance', 'genre_sci-fi']

    df_movie = data_movie[interest_columns]
    df_movie = df_movie.set_index(['movieId'])

    cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                      affinity='euclidean',
                                      linkage='ward')
    cluster.fit_predict(df_movie)
    df_movie['cluster'] = cluster.labels_

    df_movie.reset_index(drop=False,
                         inplace=True)

    movie_with_cluster = pd.merge(data_movie,
                                  df_movie[['movieId', 'cluster']],
                                  how="left",
                                  on=["movieId"])

    return movie_with_cluster
