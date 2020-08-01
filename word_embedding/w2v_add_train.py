import gensim
import pandas as pd
from util.process import data_process

if __name__=='__main__':
	#数据读取
	data=pd.read_csv('../data/query_data')
	word_list=list(data['query'].apply(lambda x:x+'  ')+data['answer'])
	sentences=list(map(data_process,word_list))
	 
	#模型读取
	model = gensim.models.Word2Vec.load('./model/word2vec_baike')
	model.build_vocab(sentences,updates=True)
	model.train(sentences, epochs=5, total_examples=model.corpus_count)
	# 存储模型
	model.wv.save_word2vec_format('./model/w2vmodel.bin',binary = True)
