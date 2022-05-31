import pandas as pd
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import kobert
import re
# import yfinance as yf
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import urllib3
from matplotlib import font_manager, rc
import scipy.stats as stats


urllib3.disable_warnings()
file_path = '../data/naver_news_data/'
file_list = list(os.listdir(file_path))

def keyword_list_in(keyword_list, sentence):
    for keyword in keyword_list:
        if keyword in sentence:
            return True
    return False
def remove_unecessary(sentence):
    print(sentence)
    sentence = str(sentence)
    if len(sentence) < 5:
        return ''
    elif type(sentence) != str:
        return ''
    sentence = re.sub('&quot;','', sentence)
    sentence = re.sub('<img alt=', '', sentence)
    sentence = re.sub(' height=.*', '', sentence)
    sentence = re.sub('amp;', '', sentence)
    return sentence
def select_df_by_keyword(keyword, file_path='../data/naver_news_data/'):
    total_df = pd.DataFrame()
    file_list = list(os.listdir(file_path))
    for file_nm in file_list:
        if '.xlsx' not in file_nm:
            continue
        df = pd.read_excel(file_path+file_nm, index_col=0)
        total_df = total_df.append(df)

    total_df = total_df.dropna().drop_duplicates(subset=['title','date'])

    total_df['title'] = total_df['title'].apply(lambda x: remove_unecessary(x))

    for key in keyword:
        print(key)
        total_df['keyword_tf'] = total_df['title'].apply(lambda x: key in str(x))
        keyword_df = total_df[total_df['keyword_tf'] == True]
        keyword_df = keyword_df.sort_values(by='date')
        keyword_df.to_excel(file_path+'intermediate/{}.xlsx'.format(key), encoding='utf-8-sig')
        print(len(keyword_df))

def get_price(ticker,date):
    # return fdr.DataReader(ticker, date)
    try:
        return pd.read_excel('../data/price/{}.xlsx'.format(ticker.replace('/','_')), index_col=0)
    except:
        return pd.read_excel('../data/price/{}.xls'.format(ticker.replace('/', '_')), index_col=0)

def cal_returns(price):
    # return (price - price.shift(1)) / price.shift(1)
    return price

def cal_diff(price):
    return (price - price.shift(1))

def add_bert_scores_all_dir(file_path = '../data/naver_news_data/'):
    file_list = list(os.listdir(file_path))
    for file_nm in file_list:
        if file_nm.replace('.xlsx','_jybert.xlsx') in file_list or '_groupby' in file_nm or 'bert' in file_nm:
            continue
        else:
            print(file_nm)
            data = pd.read_excel(file_path + file_nm, index_col=0)
            data['title_new'] = data['title'].apply(lambda x: remove_unecessary(x))
            data['jybert_score'] = data['title_new'].apply(lambda x: kobert.cal_jybert_score(x))
            data.to_excel(file_path + file_nm.replace('.xlsx', '_jybert.xlsx'))
            data['count'] = 1
            data = data.groupby('date').agg('sum', 'mean').dropna()
            data.to_excel(file_path + file_nm.replace('.xlsx', '_groupby.xlsx'))

def add_bert_scores_file(file_path = '../data/naver_news_data/intermediate/',file_nm='S&P.xlsx'):
    data = pd.read_excel(file_path+file_nm, index_col=0)

    data['title_new'] = data['title'].apply(lambda x: remove_unecessary(x))
    data['jybert_score'] = data['title_new'].apply(lambda x: kobert.cal_jybert_score(x))
    data.to_excel(file_path + file_nm.replace('.xlsx', '_jybert.xlsx'))
    data['count'] = 1
    data = data.groupby('date').agg('sum', 'mean').dropna()
    data.to_excel(file_path + file_nm.replace('.xlsx', '_groupby.xlsx'))

def compare_score_pr(keywords, ticker, shift_day=0, plot=True, file_path = '../data/naver_news_data/intermediate/'):
    data = pd.DataFrame()
    for keyword in keywords:
        print(keyword)
        if keyword+'_groupby.xlsx' in list(os.listdir(file_path)):
            df = pd.read_excel(file_path + keyword + '_groupby.xlsx',index_col=0)
            data = data.append(df)

        else:
            df = pd.read_excel(file_path + keyword+'_jybert.xlsx')
            df['count'] = 1
            df = df.drop_duplicates(subset=['date', 'title'])
            data = df.groupby('date').sum().dropna()
            # data = data.set_index('date')
            data.to_excel(file_path + keyword + '_groupby.xlsx')
            data = data.append(df)


    price = get_price(ticker=ticker,
                      date='2017-09-01')  # Indexes: 'KS11'(코스피지수), 'KQ11'(코스닥지수), 'DJI'(다우존스지수), 'IXIC'(나스닥지수), 'US500'(S&P 500지수) ...
    price = price.shift(shift_day)
    returns = cal_returns(price)
    returns.index = list(map(lambda x: int(x.strftime('%Y%m%d')), returns.index))
    total_data_re = returns[['Close']].join(data[['jybert_score', 'count']], how='outer').dropna()
    total_data_re['jybert_score'] = total_data_re['jybert_score'].apply(lambda x: float(x))
    total_data_re['jybert_score_mean'] = total_data_re[['jybert_score','count']].apply(lambda row: float(row['jybert_score'])/float(row['count']), axis=1)
    total_data_re['count'] = total_data_re['count'].apply(lambda x: float(x))
    X = total_data_re.Close.values
    Y_sum = total_data_re.jybert_score.values
    Y_mean = total_data_re.jybert_score_mean.values
    cov_sum_return = np.cov(X, Y_sum)[0, 1]
    coef_sum_return = np.corrcoef(X, Y_sum)[0, 1]
    p_value_sum_return = stats.pearsonr(X,Y_sum)
    cov_mean_return = np.cov(X, Y_mean)[0, 1]
    coef_mean_return = np.corrcoef(X, Y_mean)[0, 1]
    p_value_mean_return = stats.pearsonr(X,Y_mean)[1]
    print("cov_sum: ", cov_sum_return)
    print("coef_sum: ", coef_sum_return)
    print("cov_mean: ", cov_mean_return)
    print("coef_mean: ", coef_mean_return)

    if plot:
        from datetime import datetime
        price.index = list(map(lambda x: datetime.strptime(str(x), '%Y%m%d'), price.index))
        total_data_re.index = list(map(lambda x: datetime.strptime(str(x), '%Y%m%d'), total_data_re.index))
        thres = 10
        negative = total_data_re[total_data_re.jybert_score < -1*thres].index.tolist()
        positive = total_data_re[total_data_re.jybert_score > thres].index.tolist()
        plt.plot(returns['Close'].loc['20170101':])
        for n in negative:
            print(n)
            print(total_data_re.loc[n,'Close'])
            plt.axvline(x=n, c='r', linewidth=1)
        for p in positive:
            print(p)
            print(total_data_re.loc[p,'Close'])
            plt.axvline(x=p, c='r', linewidth=1)
        plt.savefig('../data/naver_news_data/result/thres/' + str(thres)+'.png')


        plt.show()

if __name__ == '__main__':
    add_bert_scores_all_dir()