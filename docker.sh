#!/bin/bash

docker run --rm --runtime=nvidia -p 8501:8501 \
  -v `pwd`/tf_bert_model:/models/tf_bert_model \
  -e MODEL_NAME=tf_bert_model -t tensorflow/serving:1.14.0-gpu