#!/usr/bin/env python
# coding=utf-8
import collections
import random
import sys
import os
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
# There are 13 integer features and 26 categorical features
import numpy as np

continous_features_name = ['age','creativeSize']
categorial_features_name = ['aid','uid','label','advertiserId','campaignId','creativeId','creativeSize','adCategoryId','productId',
                            'productType','age','gender','marriageStatus','education','consumptionAbility','LBS','interest1','interest2',
                            'interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','ct','os','carrier','horse']

continous_features = [6]
a = range(3,32)
categorial_features=[x for x in a if x not in [6,10]]
# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [90]

class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')

                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)
                    break

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        for every_categorial in str(features[categorial_features[i]]).split(' '):
                            self.dicts[i][every_categorial] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())

            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0
            print i,categorial_features_name[i],self.dicts[i],len(self.dicts[i])

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts))


def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.

    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir, 'train_data.csv'), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(
        os.path.join(datadir, 'train_data.csv'), categorial_features, cutoff=200)  # 200 50

    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    random.seed(0)

    # 90% of the data are used for training, and 10% of the data are used
    # for validation.
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w')
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w')

    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w')
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
            with open(os.path.join(datadir, 'train_data.csv'), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(',')
                    continous_feats = []
                    continous_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        continous_vals.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                        continous_feats.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # ('{0}'.format(val))

                    categorial_vals = []
                    categorial_lgb_vals = []
                    for i in range(0, len(categorial_features)):
                        for v in features[categorial_features[i]].split(' '):
                            val = dicts.gen(i, v) + categorial_feature_offset[i]
                            categorial_vals.append(str(val))
                            val_lgb = dicts.gen(i, v)
                            categorial_lgb_vals.append(str(val_lgb))

                    continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    label = features[2]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        train_ffm.write(label + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii, val in
                             enumerate(continous_vals.split(','))]) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 1, str(np.int32(val) + 1)) for ii, val in
                             enumerate(categorial_vals.split(','))]) + '\n')

                        train_lgb.write(label + '\t')
                        train_lgb.write('\t'.join(continous_feats) + '\t')
                        train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

                    else:
                        out_valid.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        valid_ffm.write(label + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii, val in
                             enumerate(continous_vals.split(','))]) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 1, str(np.int32(val) + 1)) for ii, val in
                             enumerate(categorial_vals.split(','))]) + '\n')

                        valid_lgb.write(label + '\t')
                        valid_lgb.write('\t'.join(continous_feats) + '\t')
                        valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    train_ffm.close()
    valid_ffm.close()

    train_lgb.close()
    valid_lgb.close()

    print "train done"

    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w')
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test_data.csv'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(',')

                continous_feats = []
                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    continous_feats.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))  # ('{0}'.format(val))

                categorial_vals = []
                categorial_lgb_vals = []
                for i in range(0, len(categorial_features)):
                    for v in features[categorial_features[i]-1].split(' '):
                        val = dicts.gen(i,v) + categorial_feature_offset[i]
                        categorial_vals.append(str(val))

                        val_lgb = dicts.gen(i, v)
                        categorial_lgb_vals.append(str(val_lgb))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)

                out.write(','.join([continous_vals, categorial_vals]) + '\n')

                test_ffm.write('\t'.join(
                    ['{}:{}:{}'.format(ii, ii, val) for ii, val in enumerate(continous_vals.split(','))]) + '\t')
                test_ffm.write('\t'.join(
                    ['{}:{}:1'.format(ii + 1, str(np.int32(val) + 1)) for ii, val in
                     enumerate(categorial_vals.split(','))]) + '\n')

                test_lgb.write('\t'.join(continous_feats) + '\t')
                test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    test_ffm.close()
    test_lgb.close()
    return dict_sizes


def split_data_to_ffm(x,feat_dict):
    a = []
    for i in  str(x).split(' '):
        a.append(feat_dict[i])
    return a

def pd_to_ffm(df):
    field_dict = dict(zip(df.columns, range(len(df.columns))))
    print field_dict

    ffm = pd.DataFrame()
    idx = 0
    t = df.dtypes.to_dict()
    for col in df.columns:
        print col
        if col in categorial_features :  ##category数据
            col_value = df[col].unique()
            flatten_set_list=[]
            for i in col_value:
                flatten_set_list.extend(str(i).split(' '))
            diff_col = set(flatten_set_list)
            # print diff_col,len(diff_col)
            feat_dict = dict(zip(diff_col, range(idx, idx + len(diff_col))))
            se = df[col].apply(lambda x: (field_dict[col], split_data_to_ffm(x, feat_dict), 1))
            ffm = pd.concat([ffm, se], axis=1)
            idx += len(diff_col)

        elif col in continous_features:  ##数值型数据
            min_max_scaler = preprocessing.MinMaxScaler()  ##归一化处理
            df[col] = min_max_scaler.fit_transform(df[col])
            si = df[col].apply(lambda x: (field_dict[col],field_dict[col],x))
            ffm = pd.concat([ffm, si], axis=1)
        else:
            ffm = pd.concat([ffm, df[col]], axis=1)

    print len(ffm), ffm.shape
    return ffm

if __name__ == '__main__':
    path = '/Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/'
    # train_data = pd.read_csv(path + 'train_data.csv')
    # test_date = pd.read_csv(path + 'test_data.csv')

    print "load data done"
    preprocess= preprocess(path,path)