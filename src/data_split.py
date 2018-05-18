#!/usr/bin/env python
# coding=utf-8

import sys

feature_name = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1',
                'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3',
                'ct', 'os', 'carrier', 'house']

file_name = sys.argv[1]
with open(file_name, 'rb') as file_name_reader:
    for line in file_name_reader.readlines():
        out_dict = {}
        data = line.strip('\n').split('|')
        a = ''
        cnt = 1
        for i in data:
            split_data = i.split(' ')
            if len(split_data) >= 2:
                target = ' '.join(split_data[1:])
            else:
                target = None
            out_dict[split_data[0]] = target

        for line in feature_name:
            if out_dict.has_key(line):
                out = out_dict[line]
            else:
                out = ''

            if cnt == 1:
                a = a + out
            else:
                a = a + ',' + out
            cnt += 1
        print a
