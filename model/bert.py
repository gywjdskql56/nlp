# import pandas as pd
import os
# import nltk
# from nltk.tokenize import sent_tokenize
from transformers import pipeline
import nltk
nltk.download('punkt')
# import re

os.environ["CURL_CA_BUNDLE"]=""
# file_list = os.listdir('/content/drive/MyDrive/Sentiment/concall_aapl_pypdf2/')
def cal_bert_score(sentence, sent_tokenize=False):

    bert_classifier = pipeline("sentiment-analysis")
    if sent_tokenize:
        sentences = nltk.sent_tokenize(str(sentence))
        scores = list()
        for sentence in sentences:
            score = bert_classifier(sentence)
            scores.append(score)
        return {'scores':scores, 'sentences':sentences}
    else:
        score = bert_classifier(sentence)
        return score
cal_bert_score('i am fine thank you')
print(1)
    # bert_classifier(sentence)
# for i in range(len(file_list)):
#     file_nm = file_list[i]
#     print(file_nm)
#     with open('/content/drive/MyDrive/Sentiment/concall_aapl_pypdf2/' + file_list[i], 'rb') as f:
#         data = f.readlines()
#         sentences = sent_tokenize(str(data))
#         scores = []
#         for sentence in sentences:
#
#             sentence = re.sub('{BIO .* <GO>}', '', sentence)
#             if len(sentence) > 512:
#                 sentiment_score = -100
#             else:
#                 sentiment_score = bert_classifier(sentence)
#                 sentiment_score = (-1) * sentiment_score[0]['score'] if sentiment_score[0]['label'] == 'NEGATIVE' else \
#                 sentiment_score[0]['score']
#             scores.append(sentiment_score)
#         pd.DataFrame({'sentence': sentences, 'scores': scores}).to_excel(
#             '/content/drive/MyDrive/Sentiment/concall_aapl_score/' + file_nm.replace('.txt', '.xlsx'))

