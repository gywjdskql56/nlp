import pandas as pd
import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
nltk.download('punkt')
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
import pandas as pd

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import finbert


os.environ["CURL_CA_BUNDLE"]=""
file_list = os.listdir('../data/concall_aapl_data/')
file_list = list(filter(lambda x: '.txt' in x,file_list))
bert_classifier = pipeline("sentiment-analysis")
def split_sent_and_add_bert_score():
    for i in range(len(file_list)):
        file_nm = file_list[i]
        print(file_nm)
        new_sentences = list()
        with open('../data/concall_aapl_data/' + file_list[i], 'rb') as f:
            data = f.readlines()
            data = (data[0]).decode("utf-8")
            sentences = sent_tokenize(data)
            scores = []
            for sentence in sentences:
                if len(re.sub('{.*}', '', sentence)) != len(sentence):
                    print(sentence)
                    sentence = re.sub('{.*}', '', sentence)
                    print(sentence)
                new_sentences.append(sentence)
                if len(sentence) > 512 or 'bloomberg' in sentence.lower():
                    sentiment_score = -100
                else:
                    sentiment_score = cal_bert_score(sentence)
                    sentiment_score = (-1) * sentiment_score[0]['score'] if sentiment_score[0]['label'] == 'NEGATIVE' else sentiment_score[0]['score']
                    sentiment_score_fin = cal_finbert_score(sentence)

                    # sentiment_score_fin = (-1) * sentiment_score[0]['score'] if sentiment_score[0]['label'] == 'NEGATIVE' else \
                    #     sentiment_score[0]['score']

                scores.append(sentiment_score)
            pd.DataFrame({'sentence': new_sentences, 'scores': scores}).to_excel(
                '../data/concall_aapl_data/' + file_nm.replace('.txt', '.xlsx'))


def add_finbert_score(file_dir):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    for file_name in os.listdir(file_dir):
        print(file_name)
        if '.xlsx' not in file_name:
            pass
        df = pd.read_excel(file_dir + file_name, index_col=0)
        for idx, sentence in zip(df.index, df['sentence'].tolist()):
            inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            positive = predictions[:, 0].tolist()[0]
            negative = predictions[:, 1].tolist()[0]
            neutral = predictions[:, 2].tolist()[0]
            print(positive)
            df.loc[idx, 'finbert_positive'] = positive
            df.loc[idx, 'finbert_negative'] = negative
            df.loc[idx, 'finbert_neutral'] = neutral
            if np.argmax([positive, negative, neutral]) == 0:
                df.loc[idx, 'finbert_score'] = positive
            elif np.argmax([positive, negative, neutral]) == 1:
                df.loc[idx, 'finbert_score'] = negative * (-1)
            else:
                df.loc[idx, 'finbert_score'] = 0
            # df.loc[idx, 'finbert_score'] = 0
            df.to_excel(file_dir + file_name)
    return df


# np.mean(scores)
file_path = '../data/concall_aapl_data/'
file_list = os.listdir('../data/concall_aapl_data/')
df_score = pd.DataFrame(columns=['date','scores'], index=[i for i in range(len(file_list))])

for idx in range(len(file_list)):
    if '.xlsx' in file_list[idx]:
        df = pd.read_excel(file_path+file_list[idx])
        df['finbert_scores'] = df['sentence'].apply(lambda x: finbert.cal_finbert_score(x))
        df.to_excel(file_path+file_list[idx])
        # df = df[df.scores!=-100]
        # df_score.loc[idx, 'date'] = file_list[idx][:8]
        # df_score.loc[idx, 'scores'] = np.mean(df.scores.tolist())
        # print(file_list[idx][:8])
        # print(np.mean(df.scores.tolist()))



# aapl = yf.Ticker("AAPL")
# aapl_pr = aapl.history(period="max")
# plt.plot(aapl_pr['Close'].loc['2015-01-01':])
# plt.show()
#
# df_score = df_score.reset_index()
# df_score['date'] = df_score['date'].apply(lambda x: datetime.strptime(x[:4]+'-'+x[4:6]+'-'+x[6:], '%Y-%m-%d'))
#
# from datetime import datetime
# plt.rcParams["figure.figsize"] = (50,20)
#
# end_dt = ''
# aapl_pr[['score']] = ''
# for i in range(len(df_score)):
#   start_dt = datetime.strptime(df_score.loc[i,'date'],'%Y-%m-%d')
#   print(start_dt)
#   plt.axvline(x=start_dt, color='purple', linewidth=3)#abs(df_score.loc[i,'scores'])*2)
#   if i<len(df_score)-1:
#     end_dt = datetime.strptime(df_score.loc[i+1,'date'],'%Y-%m-%d')
#   else:
#     end_df = aapl_pr.index[-1]
#     print(end_df)
#   if df_score.loc[i, 'scores'] < 0:
#     plt.axvspan(start_dt, end_dt, facecolor='r', alpha=0.5)
#   else:
#     plt.axvspan(start_dt, end_dt, facecolor='b', alpha=0.5)
#   # plt.axvspan(datetime.strptime('2022-03-08','%Y-%m-%d'), datetime.strptime('2022-04-29','%Y-%m-%d'), facecolor='b', alpha=0.5)
#   plt.plot(aapl_pr.Close.loc['2015-01-01':],color='black')
# plt.show()
