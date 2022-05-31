import pandas as pd
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from model import kobert
import re
import yfinance as yf
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
    # sp = yf.Ticker(ticker)
    # price = sp.history

def cal_returns(price):
    return (price - price.shift(1)) / price.shift(1)

def cal_diff(price):
    return (price - price.shift(1))


def compare_score_pr(keywords, data, ticker, shift_day=0, plot=True, file_path = '../data/naver_news_data/intermediate/'):


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


    diff = cal_diff(price)
    diff.index = list(map(lambda x: int(x.strftime('%Y%m%d')), diff.index))
    total_data_di = diff[['Close']].join(data[['jybert_score', 'count']], how='outer').dropna()
    total_data_di['jybert_score'] = total_data_di['jybert_score'].apply(lambda x: float(x))
    total_data_di['jybert_score_mean'] = total_data_di[['jybert_score', 'count']].apply(
        lambda row: float(row['jybert_score']) / float(row['count']), axis=1)
    total_data_di['count'] = total_data_di['count'].apply(lambda x: float(x))
    X = total_data_di.Close.values
    Y_sum = total_data_di.jybert_score.values
    Y_mean = total_data_di.jybert_score_mean.values
    cov_sum_diff = np.cov(X, Y_sum)[0, 1]
    coef_sum_diff = np.corrcoef(X, Y_sum)[0, 1]
    p_value_sum_diff = stats.pearsonr(X, Y_sum)
    cov_mean_diff = np.cov(X, Y_mean)[0, 1]
    coef_mean_diff = np.corrcoef(X, Y_mean)[0, 1]
    p_value_mean_diff = stats.pearsonr(X, Y_mean)[1]
    print("cov_sum: ",cov_sum_diff)
    print("coef_sum: ", coef_sum_diff)
    print("cov_mean: ", cov_mean_diff)
    print("coef_mean: ", coef_mean_diff)
    if plot:
        font_path = "C:/Windows/Fonts/NGULIM.TTF"
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)

        plt.scatter(total_data_di.Close.values, total_data_di.jybert_score.values, alpha=0.5, c='b', marker='s', label='diff')
        plt.title(ticker + '||' + ','.join(keywords))
        plt.savefig('../data/naver_news_data/result/'+ticker.replace('/','_') + '_' + ','.join(keywords)+'_'+str(shift_day)+'_diff.png')
        # plt.show()
        plt.clf()

        plt.scatter(total_data_re.Close.values, total_data_re.jybert_score.values, alpha=0.5, c='r', marker='o', label='return')
        plt.title(ticker+'||'+','.join(keywords))
        plt.savefig('../data/naver_news_data/result/' + ticker.replace('/','_') + '_' + ','.join(keywords) +'_'+str(shift_day) + '_return.png')
        # plt.show()
        plt.clf()
    result_df = pd.read_excel('../data/naver_news_data/result/'+'result.xlsx', index_col=0)
    result_df.loc[max(result_df.index)+1] = [','.join(keywords), ticker, shift_day, cov_sum_return, coef_sum_return,
                                     cov_mean_return, cov_mean_return, cov_sum_diff,coef_sum_diff, cov_mean_diff,
                                     coef_mean_diff, int(total_data_re['count'].sum()),p_value_sum_diff,p_value_mean_diff,p_value_sum_return,p_value_mean_return]
    # result_df.drop_duplicates(subset=list(result_df.columns)).to_excel('../data/naver_news_data/result/'+'result.xlsx',encoding='utf-8-sig')
    result_df.to_excel('../data/naver_news_data/result/'+'result.xlsx',encoding='utf-8-sig')




# def run_analysis()

if __name__ == '__main__':
    file_path = '../data/kaggle_news/'
    file_list = list(os.listdir(file_path))
    # month = {'Jul':'07', 'Mar':'03', 'Nov':'11', 'Sep':'09', 'Jan':'01', 'Apr':'04', 'Oct':'10', 'May':'06', 'Feb':'02', 'Dec':'12', 'Aug':'08', 'Jun':'06'}
    # df1 = pd.read_csv(file_path + 'reuters_headlines.csv')
    # df1['date'] = df1['Time'].apply(lambda x: x.split(' ')[2] +month[x.split(' ')[0]]+x.split(' ')[1])
    # df1['jybert_score'] = df1['Headlines'].apply(lambda x: kobert.cal_jybert_score(x))
    df = pd.read_excel('../data/kaggle_news/headlines.xlsx', index_col=0)
    df['tf'] = df['Headlines'].apply(lambda x: 'S&P' in x)
    sp = df[df.tf==True]
    sp['count'] = 1
    sp = sp.groupby('Time').sum()
    sp = sp.set_index('date')
    compare_score_pr(keywords='S&P500', data=sp, ticker='US500')
    # df1['Time'] = df1['Time'].apply(lambda x: '0'+x.split('-')[0] if len(x.split('-')[0])==1 else x.split('-')[0]+month[x.split('-')[1]]+x.split('-')[0])
    for file_nm in file_list:
        df = pd.read_csv(file_path+file_nm)
        print(df.head())
        print(file_nm)
    print(1)
