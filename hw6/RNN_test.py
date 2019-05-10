
import pandas as pd
import numpy as np
import keras
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

#coding=utf-8
count=0
jieba.set_dictionary('sys argv[2]')
data=[]
with open(sys.argv[1], newline='',encoding="utf-8-sig") as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		data.append(row[1])
		#data.append(emoji.demojize(row[1]))
		count+=1
newdata=[];
print(data[0])
for i in range(len(data)):
	newdata.append([])
	for ch in data[i]:
		newdata[i].append(ch)
#w2v_model = word2vec.Word2Vec(newdata,size=1000,min_count=5)
#data=data.values
#data = pd.DataFrame(data)
print(newdata[0])
w2v_model=word2vec.Word2Vec.load('word.model')

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
PADDING_LENGTH = 300
X = text_to_index(newdata)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
print("Shape:", X.shape)
print("Sample:", X[0])
"""
y = []
count=0;


with open(sys.argv[3], newline='') as csvFile:
#with open('/Users/peter yang/Downloads/train.csv', newline='') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		if count==0:
			count+=1
			continue
		y.append(row[1])
		count+=1
y=np.array(y,dtype=int)
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
"""
model=keras.models.load_model('my_model.h5')
pre=model.predict(X)
print(pre)
output=[]
with open(sys.argv[2], 'w', newline='') as csvfile:
#   建立 CSV 檔寫入器
	writer = csv.writer(csvfile)
	writer.writerow(["id","label"])
	for i in range(len(pre)):
		if pre[i]>0.5:
			output.append(1)
		else:
			output.append(0)
		writer.writerow([str(i),output[i]])
"""

MAX_FEATURES = 1000
MAX_SENTENCE_LENGTH = 100
model = Sequential()
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)
model.add(embedding_layer)
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

BATCH_SIZE = 512
NUM_EPOCHS = 10
model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_split=0.1)

model.save('my_model.h5')
"""
#			python RNN_test.py test_x.csv ans.csv
