# import yfinance as yf
# import matplotlib.pyplot as plt
# import numpy as np
# from datetime import datetime
#
# aapl = yf.Ticker("AAPL")
# aapl_pr = aapl.history(period="max")
# df_score = df_score.reset_index()
# df_score['date'] = df_score['date'].apply(lambda x: datetime.strptime(x[:4]+'-'+x[4:6]+'-'+x[6:], '%Y-%m-%d'))
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