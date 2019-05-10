
'''
one hot测试
在GTX960上，约100s一轮
经过90轮迭代，训练集准确率为96.60%，测试集准确率为89.21%
Dropout不能用太多，否则信息损失太严重
'''
import pandas as pd
import numpy as np
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import sys
import csv
import emoji
import jieba
import keras

#coding=utf-8
count=0
jieba.set_dictionary(sys.argv[4])
data=[]
with open(sys.argv[1], newline='',encoding="utf-8-sig") as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if(count>=119018):
			break;
		if count==0:
			count+=1
			continue
		data.append(row[1])
	#	data.append(emoji.demojize(row[1]))
		count+=1
newdata=[];
for i in range(len(data)):
	a=jieba.cut(data[i])
	newdata.append(list(a))
w2v_model = word2vec.Word2Vec(newdata,size=1200,min_count=5,sg=0)
#data=data.values
#data = pd.DataFrame(data)
w2v_model.save('word.model')
#w2v_model=word2vec.Word2Vec.load('word.model')

embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)
PADDING_LENGTH = 400
X = text_to_index(newdata)
X = pad_sequences(X, maxlen=PADDING_LENGTH)

y = []
count=0;


with open(sys.argv[2], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count>=119018:
			break
		if count==0:
			count+=1
			continue
		y.append(row[1])
		count+=1
y=np.array(y,dtype=int)
HIDDEN_LAYER_SIZE = 128

model = Sequential()
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=True)
model.add(embedding_layer)
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
model.summary()
BATCH_SIZE = 128
NUM_EPOCHS = 1
model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

model.save('my_model.h5')
#		python RNN.py train_x.csv train_y.csv