#!/usr/bin/env python
# coding=utf-8
from compiler.ast import flatten
import warnings

warnings.filterwarnings('ignore')
import sys
import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')


def split_data_to_ffm(x, feat_dict):
    a = []
    for i in x.split(' '):
        a.append(feat_dict[i])
    return a


def pd_to_ffm(df):
    field_dict = dict(zip(df.columns, range(len(df.columns))))
    print field_dict

    ffm = pd.DataFrame()
    idx = 0
    t = df.dtypes.to_dict()
    for col in df.columns:
        col_type = t[col]
        print col, col_type, col_type.kind
        if col_type.kind == 'O':  ##category数据
            col_value = df[col].unique()
            diff_col = flatten([i.split(' ') for i in col_value])
            feat_dict = dict(zip(diff_col, range(idx, idx + len(diff_col))))
            # print feat_dict
            se = df[col].apply(lambda x: (field_dict[col], split_data_to_ffm(x, feat_dict), 1))
            ffm = pd.concat([ffm, se], axis=1)
            idx += len(col_value)
        elif col_type.kind == 'i':  ##数值型数据
            pass
            min_max_scaler = preprocessing.MinMaxScaler()  ##归一化处理
            # df[col] = min_max_scaler.fit_transform(df[col])

            # si = df[col].apply(lambda x: (field_dict[col],field_dict[col],x))
            ffm = pd.concat([ffm, df[col]], axis=1)
    print len(ffm), ffm.shape
    print ffm.info()
    return ffm


def add_feat(userFeature_data, adFeature_data, train_csv, test_csv, path):
    train_data = pd.merge(train_csv, adFeature_data, on='aid', how='left')
    print "train_csv shape is {0} ,train_label shape is {1} :".format(train_csv.shape, train_data.shape)
    train_data = pd.merge(train_data, userFeature_data, on='uid', how='left')

    train_label_uid = train_data['uid'].unique()
    print "label_uid csv :", train_label_uid.shape

    test_data = pd.merge(test_csv, adFeature_data, on='aid', how='left')
    test_data = pd.merge(test_data, userFeature_data, on='uid', how='left')
    print "train_csv shape is {0} ,train_label shape is {1} :".format(test_csv.shape, test_data.shape)

    test_label_uid = test_data['uid'].unique()
    print "label_uid csv :", test_label_uid.shape

    train_data.to_csv(path + 'train_data.csv', header=True, index=False)
    test_data.to_csv(path + 'test_data.csv', header=True, index=False)
    print  "Done"


def data_str_int(df):
    for col in df.columns:
        if col not in ['uid']:
            df[col] = df[col].map(str)
        else:
            df[col] = df[col].map(int)
    return df


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print "[Error] Except 2 args ,but got %d ." .format(len(sys.argv)-1)

    argv1 = '/Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/'
    argv2 = 'userFeature_split.data'
    argv3 = 'adFeature.csv'
    argv4 = 'train.csv'
    argv5 = 'test.csv'
    path = argv1
    userFeature_data = pd.read_csv(path + argv2)
    adFeature_data = pd.read_csv(path + argv3)
    train_csv = pd.read_csv(path + argv4)
    test_csv = pd.read_csv(path + argv5)
    print "load data finished"

    add_feat(userFeature_data, adFeature_data, train_csv, test_csv, path)
    # all_example = all_data[all_data.columns.difference(['LBS','interest4'])]
    # print all_example.columns
    # all_X = pd_to_ffm(data_str_int(all_example))
    # all_X.to_csv('../data/preliminary_contest_data/onehotdata')
    # #ffm_train_data = ffm.FFMData(all_X.values, train_y)
