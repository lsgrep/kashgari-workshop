#!/bin/bash
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip

sudo pip3 install jieba pandas kashgari tensorflow-gpu==1.14.0

# default tool old on AWS
sudo pip3 install --upgrade --force-reinstall scipy