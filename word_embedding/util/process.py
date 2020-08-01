import re
import jieba
def data_process(words):
    pattern=re.compile(r"[.。!！?？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.\n吗啊]")
    words=pattern.sub('',words)
    # 分词
    words=jieba.cut(words)
    stop_word=['吗','啊','的','恩','嗯','呢']
    words=[i for i in words if i not in stop_word]
    return words

def word_ngrams(tokens, stop_words=None,ngram_range=(1,1)):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]
        # handle token n-grams
        min_n, max_n = ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(" ".join(original_tokens[i: i + n]))
        return tokens

    
