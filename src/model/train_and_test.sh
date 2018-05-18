#!/bin/bash
data=/Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data
./xlearn_train /Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/train_ffm.txt -t model -v /Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/valid_ffm.txt  -x auc -f 3 --cv -r 0.01

./xlearn_predict /Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/test_ffm.txt /Users/liliwang/Documents/mac/tengxun/data/preliminary_contest_data/train_ffm.txt.model --sigmoid

cat 1 ${data}/test_ffm.txt.out >submission.csv
zip submission.csv.zip submission.csv 
