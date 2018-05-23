#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2018/4/26
import xlearn as xl
import pandas as pd
import numpy as np
import os
path1='/Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/new/'
test_df=pd.read_csv('/Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/test2.csv')
ffm_model = xl.create_ffm()
ffm_model.setTrain(path1+'train_ffm.csv')
ffm_model.setTest(path1+'test_ffm.csv')
ffm_model.setSigmoid()
param = {'task':'binary', 'lr':0.01, 'lambda':0.001,'metric': 'auc','opt':'ftrl','epoch':5,'k':4,
         'alpha': 1.5, 'beta': 0.01, 'lambda_1': 0.0, 'lambda_2': 0.0}
ffm_model.fit(param,path1+"./model.out")
ffm_model.predict(path1+"./model.out",path1+"output.txt")
sub = pd.DataFrame()
sub['aid']=test_df['aid']
sub['uid']=test_df['uid']
sub['score'] = np.loadtxt(path1+"./output.txt")
sub.to_csv(path1+'submission.csv',index=False)
os.system('zip baseline_ffm.zip submission.csv')