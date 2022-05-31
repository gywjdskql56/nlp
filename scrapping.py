import pandas as pd
from requests import get
import datetime
from bs4 import BeautifulSoup
import re
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

category = {
    '금융': 259,
    '증권': 258,
    '산업_제계': 261,
    '중기_벤처': 771,
    '부동산': 260,
    '글로벌경제': 262,
    '생활경제': 310,
    '경제일반': 263
}
first = ''

remove_useless = lambda x: re.sub('" height=.*', '', str(x.find('img', alt=True))).replace("<img alt=\"", '')

while (True):
    today = datetime.date.today()
    # sids = [259, 258, 261,771, 260, 262, 310, 263]
    for c_key in ['부동산', '중기_벤처', '산업_제계']:
        sid = category[c_key]
        for back_day in range(1, 365 * 4 + 240):

            date = today - datetime.timedelta(days=back_day)
            date = date.strftime("%Y%m%d")
            head_lists = []
            print('{}_{}.xlsx'.format(date, c_key))
            # if '{}_{}.xlsx'.format(date, c_key) in crawled_list:
            # else:
            #
            # try:
            if True:
                for page in range(1, 30):

                    URL = 'https://news.naver.com/main/list.naver?mode=LS2D&sid2={}&sid1=101&mid=shm&date={}&page={}'.format(
                        sid, date, page)

                    res = get(URL, headers={'User-Agent': 'Mozilla/5.0'},verify=False).text

                    data = BeautifulSoup(res, "html.parser")
                    headline_list = data.find_all("dt", class_='photo')

                    headline_list = list(map(lambda x: remove_useless(x), headline_list))
                    if len(headline_list) == 0 or first == headline_list[0]:
                        break
                    else:
                        first = headline_list[0]
                        head_lists += headline_list
                print(len(head_lists))
                pd.DataFrame({'date': [date for i in range(len(head_lists))], 'title': head_lists}).drop_duplicates(
                    subset='title').to_excel(
                    'naver_news_data/{}_{}.xlsx'.format(date, c_key), encoding='utf-8-sig')

#           except:
 #               print('error')
#                continue



