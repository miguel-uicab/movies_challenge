#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import AgglomerativeClustering
from EDA_functions import clean_string
from extras import get_config


def cluster_movies(data_movie=None, n_clusters=1000):
    "Lleva a cabo un clúster jerárquico para las películas."
    "El algortimo de clúster jerárquico agrupa los datos basándose "
    "en la distancia entre cada uno y buscando que los datos "
    "que están dentro de un clúster sean los más similares entre sí."

    config = get_config()
    data_movie['genres_list'] = list(map(clean_string,
                                     data_movie['genres']))
    list_genres = list(data_movie['genres_list'])
    total_genres = set().union(*list_genres)
    total_genres = list(total_genres)

    for genre in total_genres:
        data_movie[f'genre_{genre}'] = data_movie.apply(lambda row: 1 if genre in row['genres_list'] else 0,
                                                        axis=1)

    interest_columns = config['names_for_cluster']
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
