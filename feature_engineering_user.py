#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import logging
from extras import get_config
from past_information_user import past_information_user
logging_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=logging_format, datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def total_past_extractor(is_data_train=True):
    config = get_config()
    logging.info('SE CARGAN LAS DATAS NECESARIAS.')
    # Data redundante completa ################################################
    if is_data_train:
        data_total_path = 'rating_redundant_train_2.sav'

    else:
        data_total_path = 'rating_redundant_test_1.sav'
        # data_total_path = 'rating_redundant_test_1_movies_not_in_train.sav'

    data_total = pickle.load(open(data_total_path,
                                  'rb'))
    # Cargar las cluster encontrados en las películas ########################
    movie_with_cluster = pickle.load(open('movie_with_cluster.sav',
                                          'rb'))

    data_total_and_cluster = pd.merge(data_total,
                                      movie_with_cluster,
                                      how="left",
                                      on=["movieId"])
    data_total_and_cluster.reset_index(drop=True,
                                       inplace=True)
    data_total_and_cluster['movies'] = 1

    logging.info('SE UBICAN LOS USERID DE INTERÉS.')
    # list_total_user = [8405, 121535, 59477, 91577]  # <----train
    # list_total_user = [89081, 102853, 79366]  # <----test
    list_total_user = list(data_total_and_cluster['userId'].unique())
    size_total_user = len(list_total_user)
    iter_count = iter(range(1, size_total_user + 1))

    # Tomar un ejemplo individual #############################################
    logging.info('SE EXTRAE LA INFORMACIÓN DEL PASADO DE CADA USERID.')
    list_datas_users = []
    for userid in list_total_user:
        iter_userid = next(iter_count)
        past_info_f = past_information_user(data_with_cluster=data_total_and_cluster,
                                            userId=userid)
        list_datas_users.append(past_info_f)
        logging.info(f'ITERACIÓN {iter_userid} DE UN TOTAL DE {size_total_user}.')

    datas_users = pd.concat(list_datas_users)

    logging.info('¡LISTO!')
    # list_datas_users[0].shape
    # list_datas_users[1].shape
    # list_datas_users[2].shape
    # list_datas_users[3].shape

    ###########################################################################
    logging.info('SE ADJUNTA A LA DATA CON CLUSTERS.')
    if is_data_train:
        # Data de entramiento filtrada sin "duplicadas" #######################
        data_train_no_duplicates = pickle.load(open('rating_train_whithout_duplicate.sav',
                                                    'rb'))

        data_train_no_duplicates_info_past = pd.merge(data_train_no_duplicates,
                                                      datas_users,
                                                      how="left",
                                                      on=["userId", "time_day"])
        logging.info('SE GUARDA DATA DE ENTRENAMIENTO.')
        data_path = 'final_data_train.sav'
        pickle.dump(data_train_no_duplicates_info_past,
                    open(data_path, 'wb'))
    else:
        data_total_info_past = pd.merge(data_total,
                                        datas_users,
                                        how="left",
                                        on=["userId", "time_day"])
        logging.info('SE GUARDA DATA DE PRUEBA.')
        # data_path = 'final_data_test.sav'
        data_path = 'final_data_test_complete.sav'
        pickle.dump(data_total_info_past,
                    open(data_path, 'wb'))

    logging.info('FIN DE TODO EL PROCESO.')


if __name__ == '__main__':
    total_past_extractor()
