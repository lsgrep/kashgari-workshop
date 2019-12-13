import tensorflow as tf
import pandas as pd
import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.classification import BiLSTM_Model

kashgari.config.use_cudnn_cell = True
BERT_PATH = 'chinese_L-12_H-768_A-12'

# 初始化 Embeddings
embed = BERTEmbedding(BERT_PATH,
                      task=kashgari.CLASSIFICATION,
                      sequence_length=64)

tokenizer = embed.tokenizer

df = pd.read_csv('weibo_senti_100k.csv')
# 进行分词处理
df['cutted'] = df['review'].apply(lambda x: tokenizer.tokenize(x))

# 准备训练测试数据集
train_x = list(df['cutted'][:int(len(df)*0.7)])
train_y = list(df['label'][:int(len(df)*0.7)])

valid_x = list(df['cutted'][int(len(df)*0.7):int(len(df)*0.85)])
valid_y = list(df['label'][int(len(df)*0.7):int(len(df)*0.85)])

test_x = list(df['cutted'][int(len(df)*0.85):])
test_y = list(df['label'][int(len(df)*0.85):])

# 使用 embedding 初始化模型
model = BiLSTM_Model(embed)
# 先只训练一轮
model.fit(train_x, train_y, valid_x, valid_y, batch_size=1024, epochs=1)
model.evaluate(test_x, test_y, batch_size=512)


random_stuff = ['高兴','好难过','这个好简单','真的是折腾']
model.predict([tokenizer.tokenize(i) for i in random_stuff])

model.save('bert_model')
kashgari.utils.convert_to_saved_model(model, 'tf_bert_model', version=1)
