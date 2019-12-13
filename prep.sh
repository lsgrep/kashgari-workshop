#!/bin/bash
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip

wget https://lsgrep.com/weibo_senti_100k.csv


sudo pip3 install pandas kashgari tensorflow-gpu==1.14.0

sudo pip3 install --upgrade --force-reinstall scipy jieba