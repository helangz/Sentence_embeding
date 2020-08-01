from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

if __name__=='__main__':
	glove_file = datapath('/home/su/HL/FQA/model/vectors_qa.txt') 
	tmp_file = get_tmpfile('/home/su/HL/FQA/model/w2v_glove.txt')
	_ = glove2word2vec(glove_file, tmp_file) 
	model = KeyedVectors.load_word2vec_format('./model/w2v_glove.txt') 
