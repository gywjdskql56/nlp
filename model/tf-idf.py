import pandas as pd
from collections import Counter
from konlpy.tag import Kkma
from konlpy.utils import pprint
import operator
import re
import math

def make_counter(document):
    remove_no_hangul = lambda x: re.sub('[ ]+',' ',str(re.sub('[^가-힣 ]|[표]', ' ', str(x))))
    doc_word = remove_no_hangul(document).split(' ')
    doc_count = Counter(doc_word)
    return doc_count

def cal_tf_idf(counter1, counter2):
    df = Counter(list(counter1.keys())+list(counter2.keys()))
    tf_idf1 = dict()
    tf_idf2 = dict()
    for word in list(counter1.keys()):
        score = cal_tf(word, counter1) * cal_idf(df, word)
        tf_idf1[word] = score
    for word in list(counter2.keys()):
        score = cal_tf(word, counter2) * cal_idf(df, word)
        tf_idf2[word] = score
    return sorted(tf_idf1.items(), key=operator.itemgetter(1)), sorted(tf_idf2.items(), key=operator.itemgetter(1))

def cal_tf(word, counter):
    return counter[word]

def cal_idf(df, word):
    doc_num = 2
    df_score = df[word]
    return math.log(doc_num/(1+df_score))




if __name__=='__main__':
    data = pd.read_excel('../data/naver_news_data/intermediate/코스피_jybert.xlsx')
    doc1 = '. '.join(data[data.jybert_score < 0]['title'])
    doc2 = '. '.join(data[data.jybert_score > 0]['title'])
    counter1 = make_counter(doc1)
    counter2 = make_counter(doc2)
    sorted_doc1, sorted_doc2 = cal_tf_idf(counter1, counter2)
    df1 =  pd.DataFrame(index=[i for i in range(len(sorted_doc1))], columns=['word','score'])
    for idx, item in enumerate(sorted_doc1):
        word, freq = item
        df1.loc[idx, 'word'] = word
        df1.loc[idx, 'score'] = freq
    df2 = pd.DataFrame(index=[i for i in range(len(sorted_doc1))], columns=['word', 'score'])
    for idx, item in enumerate(sorted_doc2):
        word, freq = item
        df2.loc[idx, 'word'] = word
        df2.loc[idx, 'score'] = freq
    df1.to_excel('../data/naver_news_data/result/negative.xlsx', encoding='utf-8-sig')
    df2.to_excel('../data/naver_news_data/result/positive.xlsx', encoding='utf-8-sig')
    # cal_tf_idf(doc1, doc2)
    print(1)