import jieba
import kashgari

model = kashgari.utils.load_model('bert_model')
random_stuff = ['好高兴', '好欢乐', '心情低落', '质量好差']
print(model.predict([list(jieba.cut(i)) for i in random_stuff]))
