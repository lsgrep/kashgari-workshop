import requests
from kashgari import utils
import numpy as np
import jieba

random_stuff = ['好高兴', '好欢乐', '心情低落', '质量好差']

# 从保存模型加载词表
processor = utils.load_processor(model_path='tf_bert_model/1')
# 输入转换成张量
tensor = processor.process_x_dataset([list(jieba.cut(i)) for i in random_stuff])

# 格式化为 BERT 格式
tensor = [{
    "Input-Token:0": i.tolist(),
    "Input-Segment:0": np.zeros(i.shape).tolist()
} for i in tensor]

# 进行预测
r = requests.post("http://localhost:8501/v1/models/tf_bert_model:predict",
                  json={"instances": tensor})
preds = r.json()['predictions']
label_index = np.array(preds).argmax(-1)

# 把预测结果转换成具体的标签
labels = processor.reverse_numerize_label_sequences(label_index)
print(dict(zip(random_stuff, label_index)))