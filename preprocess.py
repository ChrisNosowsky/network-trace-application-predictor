#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 28th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Preprocess the xml files into input ready format
# for our machine learning classifier
# ==============================================================================

import pandas as pd
import numpy as np
import glob
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# service = ['airbnb', 'doordash']
shopping = ['Starbucks', 'Walmart', 'McDonalds', 'Macys', 'Zilow', 'doordash', 'airbnb']
tools = ['Google Chrome', 'Google Maps', 'Google Play', 'Gmail']
entertainment = ['Twitch', 'Youtube', 'Imdb']
social = ['Twitter', 'Snapchat', 'Reddit', 'Pinterest', 'New York Post', 'Linked In']

def preprocess(directory, packet_type):
    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.max_columns', 500000)
    frames = []
    failed = []
    apps_data_len = []
    folders = glob.glob(directory + "/*")
    for folder in folders:
        file = folder + "/packets for " + packet_type + ".csv"
        try:
            data = pd.read_csv(file)
        except:
            print("The following apps didn't have this packet " + folder.split("/")[-1].split('\\')[-1])
            continue
        # if folder.split("/")[-1].split('\\')[-1] in service:
        #     data.insert(len(data.columns), "label", "SERVICE")
        if folder.split("/")[-1].split('\\')[-1] in tools:
            data.insert(len(data.columns), "label", "TOOLS")
        if folder.split("/")[-1].split('\\')[-1] in shopping:
            data.insert(len(data.columns), "label", "SHOPPING")
        if folder.split("/")[-1].split('\\')[-1] in entertainment:
            data.insert(len(data.columns), "label", "ENTERTAINMENT")
        if folder.split("/")[-1].split('\\')[-1] in social:
            data.insert(len(data.columns), "label", "SOCIAL")
        # data.insert(len(data.columns), "label", folder.split("/")[-1].split('\\')[-1])
        apps_data_len.append("len for " + str(folder) + " " + str(len(data)))
        frames.append(data)

    # not sure about ignore index
    data = pd.concat(frames, ignore_index = True)

    # drop empty columns
    data.dropna(how='all', axis=1, inplace=True)

    # drop columns with same value for all rows
    unique_columns = data.nunique()
    cols_to_drop = unique_columns[unique_columns == 1].index
    data.drop(columns=cols_to_drop, axis=1, inplace=True)
    # print(data.head())
    # data.drop(columns=["log_msg_len","timestamp"], inplace = True)
    data.drop(columns=["timestamp"], inplace=True)
    # there might be one type of app that has a lot of empty values on a specific column, might fill instead
    print("number of packets for each app before removing empty rows ")
    print(apps_data_len)

    # drop rows with empty values
    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0, how="any", inplace=True)

    print("number of packets for each app after removing empty rows ")
    for label in data.label.unique():
        print(label + " " + str((data.label == label).sum()))

    # drop duplicated columns, non here
    data = data.loc[:, ~data.columns.duplicated()]
    # data.to_csv('Master_TEST2_' + packet_type + '.csv', index=False)
    # data.to_csv('Master_' + packet_type + '.csv', index=False)
    print("final dimensions")
    print(len(data.columns))
    print(len(data))

    # corr = data.corr()
    # # visualise the data with seaborn
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # sns.set_style(style='white')
    # f, ax = plt.subplots(figsize=(20,20))
    # cmap = sns.diverging_palette(10, 250, as_cmap=True)
    # sns.heatmap(corr, mask=mask, cmap=cmap,
    #             square=True,
    #             linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    # plt.show()
preprocess("C:/Users/FZTHWP/PycharmProjects/network-trace-application-predictor/data/succeed", "LTE_PDCP_UL_Stats")

# data = pd.read_csv('Master_LTE_PHY_PDCCH_PHICH_Indication_Report.csv')
# object_cols = data.select_dtypes(include="object").columns
# singe_value_cols = []
# two_values_cols = []
# more_than_two_values_cols = []
# test_def = pd.DataFrame()
# for col in object_cols:
#     print(col)
#     unique_values = data[col].unique()
#     test_def.insert(len(test_def.columns), col, pd.Series(unique_values))
# test_def.to_csv("unique values.csv", index=False)
# print(len(two_values_cols))
# print(len(more_than_two_values_cols))
# print(len(singe_value_cols))
# # data.drop(columns=delete_cols, axis=1, inplace=True)
