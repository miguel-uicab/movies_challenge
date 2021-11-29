#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from extras import get_config


def past_information_user(data_with_cluster=None, userId=None):
    "Dada información de cierto usuario, construye las características "
    "que se piensan tienen un importante poder predictivo. "

    config = get_config()
    data_ind = data_with_cluster[data_with_cluster['userId'] == userId]
    names_interest_cum = config['names_interest_cum']
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

    # Porcentaje de representación del género en el acumulado de películas ####
    cumulative_names.remove('cumulativefrecc_movies_per_day')
    percen_cumulative_names = []
    for name in cumulative_names:
        percen_cumulative_name = f'percen{name}'
        percen_cumulative_names.append(percen_cumulative_name)
        info_cumulative[percen_cumulative_name] = info_cumulative[name] / info_cumulative['cumulativefrecc_movies_per_day']

    info_cumulative.sort_values(by='time_day',
                                ascending=False,
                                inplace=True)
    info_cumulative.reset_index(drop=True,
                                inplace=True)

    info_cumulative['userId'] = userId

    # date_column = info_cumulative[['time_day']]
    # info_cumulative.rename(columns={'time_day': 'last_time_day'},
    #                        inplace=True)

    # info_cumulative.drop(0,
    #                      axis=0,
    #                      inplace=True)

    # info_cumulative.reset_index(drop=True,
    #                             inplace=True)

    # past_info = pd.merge(date_column,
    #                      info_cumulative,
    #                      left_index=True,
    #                      right_index=True,
    #                      how="outer")

    # past_info['userId'] = userId

    # return past_info

    return info_cumulative
