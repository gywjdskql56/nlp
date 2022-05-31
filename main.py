from transformers import pipeline
import pandas as pd
import os
today = '2022-04-25'
file_list = os.listdir('data/google_news_data/')

def make_daily_score(file_name):

    news = pd.read_excel('google_news_data/'+file_name, index_col=0)
    convert_time = lambda x : str(x.year)+'-'+('0'+str(x.month) if len(str(x.month))==1 else str(x.month))+'-'+('0'+str(x.day) if len(str(x.day))==1 else str(x.day))
    news['datetime'] = news['datetime'].apply(lambda x: convert_time(x))
    news['total_score'] = ''
    news['total_score'] = news.apply(lambda row :row.loc['sentiment_score']*(-1) if row.loc['sentiment']=='NEGATIVE' else row.loc['sentiment_score'], axis=1)
    news = news.sort_values(['datetime']).groupby(['datetime']).mean('total_score')
    news.to_excel('sentiment_score_bert/'+file_name, encoding='utf-8-sig')

for file_name in file_list:
    print(file_name)
    make_daily_score(file_name)
print(1)



