#!/usr/bin/env python
# coding: utf-8

# # Imports
import os
import re
import pandas as pd
import numpy as np
import pickle
from functools import reduce


tag = pd.read_csv('tag.csv')
tag.isna().sum()
values = {"tag": "undefined"}
tag.fillna(value=values, inplace=True)


tag['time_day'] = tag.apply(lambda row: row['timestamp'].split()[0], axis=1)
tag['time_day'] = pd.to_datetime(tag['time_day'])


tag_min = tag.sort_values(["time_day"], ascending=True)
tag_min = tag_min[['userId', 'movieId',	'time_day']]
tag_min.columns = ['userId', 'movieId',	'min_tag_time_day']
tag_min.drop_duplicates(["userId", "movieId"], inplace=True)

tag_max = tag.sort_values(["time_day"], ascending=False)
tag_max = tag_max[['userId', 'movieId',	'time_day']]
tag_max.columns = ['userId', 'movieId',	'max_tag_time_day']
tag_max.drop_duplicates(["userId", "movieId"], inplace=True)

tag_count = tag.groupby(["userId", "movieId"])["tag"].count().to_frame().reset_index()
tag_count.columns = ["userId", "movieId", "tag_count"]


dfs = [tag_count, tag_min, tag_max]
tag_info = reduce(lambda left, right: pd.merge(left, right,
                                               how="left",
                                               on=["userId", "movieId"]),
                  dfs)
tag_info.sort_values(by='min_tag_time_day', ascending=False, inplace=True)

d_ind = tag_info[tag_info['userId']==130827]

d_ind.dtypes
d_ind[d_ind['min_tag_time_day']<'2014-07-27']
