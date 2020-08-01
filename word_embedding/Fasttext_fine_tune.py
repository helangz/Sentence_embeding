import pandas as pd
from util.process import data_process
import numpy as np
import fasttext



if __name__=='__main__':
	## 数据预处理,将数据处理成符合fasttext输入的数据
	train=pd.read_csv('../data/query_data.csv')
	with open('../data/train_fast.txt','w') as f:
	    index=list(train.index)
	    np.random.shuffle(index)
	    for i in index:
		label=train['ids'][i]
		text=' '.join(data_process(train['query'][i]))
		f.write(f'__label__{label}'+','+text+'/n')
		
		
	#train_model
	ws = 5                # size of the context window [5]
	minCount = 1         # minimal number of word occurences [1]
	minCountLabel = 1     # minimal number of label occurences [1]
	minn = 1              # min length of char ngram [0]
	maxn = 2              # max length of char ngram [0]
	neg = 5               # number of negatives sampled [5]
	wordNgrams = 2        # max length of word ngram [1]
	loss = 'softmax'              # loss function {ns, hs, softmax, ova} [softmax]
	lrUpdateRate = 100      # change the rate of updates for the learning rate [100]
	t = 0.0001                 # sampling threshold [0.0001]
	lr=0.01
	label = '__label__'
	train_fname='../data/train_fast.txt'

	model=fasttext.train_supervised(train_fname,pretrained_vectors='./model/cc.zh.300.vec',dim = 300,epoch=5)
	model.save_model('./model/finetune_fasttext.bin')
