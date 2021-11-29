#!/usr/bin/env python
# coding: utf-8

# # Imports
import os
import re
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

data_path='final_data_test.sav'
d_2 = pickle.load(open(data_path, 'rb'))


###############################################################################
movie = pd.read_csv('movie.csv')
rating = pd.read_csv('rating.csv')
tag = pd.read_csv('tag.csv')

# Ratings #####################################################################


# Preprocesamiento de tags ####################################################
total_movies = list(tag['movieId'].unique())
tag.dropna(subset=['tag'])

def tokens_movies(data=None, movieId=None):
    output_ind = pd.DataFrame(columns=['movieId', 'tag_tokens'],
                              index=[0]).replace({np.nan: None})
    list_tag_token = []
    for name in list(data[data['movieId'] == movieId]['tag']):
        list_tag_token.append(clean_string(string=name))
    list_set = set().union(*list_tag_token)
    list_set = list(list_set)
    processed_words = [word for word in list_set if word not in
                       stopwords.words('english') and word.isalpha()]
    output_ind['movieId'] = movieId
    output_ind['tag_tokens'] = pd.Series([processed_words],
                                         index=output_ind.index)
    return [output_ind, processed_words]

tokens_movies(data=tag.dropna(subset=['tag']), movieId=36537)

great_list = []
output = pd.DataFrame(columns=['movieId', 'tag_tokens'])

# path_model = 'tokens_movies.sav'
# pickle.dump(output, open(path_model, 'wb'))
# tokens_movies = pickle.load(open(path_model, 'rb'))
len(tokens_movies['movieId'])

for movie in total_movies:
    print(movie)
    output_ind = tokens_movies(data=tag.dropna(subset=['tag']),
                               movieId=movie)
    output = output.append(output_ind[0],
                           ignore_index=True)
    great_list = great_list + output_ind[1]

great_list

import collections
counter = collections.Counter(great_list)
token_count = pd.DataFrame.from_dict(counter, orient='index').reset_index()
token_count.columns = ['token', 'count']
token_count.sort_values(by='count', ascending=False, inplace=True)
token_count.reset_index(drop=True, inplace=True)
token_count.head(30)

# pickle.load(open('token_count.sav', 'rb'))

###############################################################################
movies_with_tokens = pd.merge(movie,
                              tokens_movies,
                              how="left",
                              on=["movieId"])


# Movies ######################################################################
def clean_string(string=None):
    raw_string = re.sub(r"[ : ]", " ", string)
    # replace = raw_string.replace("|", " ").replace("-", " ").replace("'", " ")
    replace = raw_string.replace("|", " ").replace("'", " ")
    clean_charac = re.sub(r"[\.\,\[\]\(\)\_\#\*\¢\$\&\:\;\·\%\|]",
                          "",
                          replace)
    clean_lower = clean_charac.lower()

    if clean_lower == 'no genres listed':
        output = [clean_lower]
    else:
        output = clean_lower.split()
        # output = [char for char in clean_lower]
    # clean_accents = unidecode(clean_lower)
    return output

clean_string(string="Bechdel Test:Fail")
clean_string(string='Adventure|Animation|Children|Comedy|Fantasy')
clean_string(string='(no genres listed)')
clean_string(string='Action')

movies_with_tokens['genres_list'] = list(map(clean_string,
                                             movies_with_tokens['genres']))

list_genres = list(movies_with_tokens['genres_list'])
total_genres = set().union(*list_genres)
total_genres = list(total_genres)

for genre in total_genres:
    movies_with_tokens[f'genre_{genre}'] = movies_with_tokens.apply(lambda row: 1 if genre in row['genres_list'] else 0,
                                                                    axis=1)
movies_with_tokens.columns


df_movies = movies_with_tokens[['title', 'genre_film-noir',
                                'genre_no genres listed', 'genre_drama',
                                'genre_mystery', 'genre_animation',
                                'genre_horror', 'genre_fantasy',
                                'genre_war', 'genre_crime', 'genre_comedy',
                                'genre_western', 'genre_adventure',
                                'genre_documentary', 'genre_imax',
                                'genre_action', 'genre_children',
                                'genre_musical', 'genre_thriller',
                                'genre_romance', 'genre_sci-fi']]
df_movies_1 = df_movies.set_index(['title'])

user_choices = [[1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 0, 0, 0]]
df_choices = pd.DataFrame(user_choices, columns=['Apples',
                          'Bananas', 'Pineapples', 'Kiwis'],
                          index=(["User A", "User B", "User C"]))
jaccard = scipy.spatial.distance.cdist(df_choices, df_choices,
                                       metric='jaccard')
user_distance = pd.DataFrame(jaccard, columns=df_choices.index.values,
                             index=df_choices.index.values)


###############################################
import scipy.spatial
jaccard = scipy.spatial.distance.cdist(df_movies_1,
                                       df_movies_1,
                                       metric='jaccard')
user_distance = pd.DataFrame(jaccard, columns=df_movies_1.index.values,
                             index=df_movies_1.index.values)

similarity = 1-user_distance

################################################
X = np.array([[5,3],
            [10,15],
            [15,12],
            [24,10],
            [30,30],
            [85,70],
            [71,80],
            [60,78],
            [70,55],
            [80,91],])
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)
print(cluster.labels_)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')

cluster = AgglomerativeClustering(n_clusters=1000, affinity='euclidean', linkage='ward')
cluster.fit_predict(df_movies_1)
df_movies_1['cluster'] = cluster.labels_

df_movies_1[df_movies_1['cluster']==622]
df_movies_1[df_movies_1['cluster']==469]

df_movies_1['cluster'].value_counts()

df_movies_1[df_movies_1['cluster']==339]
# movie.loc[2, :]
###############################################################################
genome_scores.head(3)
genome_tags.head(3)
link.head(3)
movie.head(3)
rating.head(4)
tag.head(3)

# tag #########################################################################
genome_tags
len(genome_tags['tag'].unique())
pd.DataFrame(genome_tags['tag'].unique(), columns=['tag']).to_csv('tag.csv')
