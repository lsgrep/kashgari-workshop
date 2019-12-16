FROM tensorflow/serving:1.14.0

ADD tf_bert_model /models/tf_bert_model

EXPOSE 8500 8501