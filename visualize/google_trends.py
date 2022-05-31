from pytrends.request import TrendReq
from pytrends import dailydata

pytrend = TrendReq(hl='JST', retries=2, tz=540, requests_args={'verify': False})
keyword_list = ['kospi','코스피']
history_df = pytrend.get_historical_interest(keyword_list, year_start=2017, month_start=9, day_start=1, hour_start=0, year_end=2022, month_end=5, day_end=11, hour_end=0, cat =0, geo='')

# df = dailydata.get_daily_data('kopsi', 2017, 9, 2022, 5, geo = 'JP', requests_args={'verify':False})
print(1)
# pytrends = TrendReq(hl='JST', tz=540)