import kashgari
import numpy as np
import requests
from kashgari import utils
from kashgari.embeddings import BERTEmbedding

random_stuff = ['好高兴', '好欢乐', '心情低落']

BERT_PATH = 'chinese_L-12_H-768_A-12'

# 初始化 Embeddings
embed = BERTEmbedding(BERT_PATH,
                      task=kashgari.CLASSIFICATION,
                      sequence_length=64)
tokenizer = embed.tokenizer
# 从保存模型加载词表
processor = utils.load_processor(model_path='tf_bert_model/1')

# 输入转换成张量
tensor = processor.process_x_dataset([tokenizer.tokenize(i) for i in random_stuff])

# 格式化为 BERT 格式
tensor = [{
    "Input-Token:0": i.tolist(),
    "Input-Segment:0": np.zeros(i.shape).tolist()
} for i in tensor]

# 进行预测
r = requests.post("http://127.0.0.1:8501/v1/models/tf_bert_model:predict",
                  json={"instances": tensor})
preds = r.json()['predictions']
label_index = np.array(preds).argmax(-1)

# 把预测结果转换成具体的标签
labels = processor.reverse_numerize_label_sequences(label_index)
print(dict(zip(random_stuff, label_index)))
