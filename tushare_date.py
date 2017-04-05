import tushare as ts
import pandas as pd
import numpy as np
from WindPy import w
import os
import threading
import datetime
import random
ISOTIMEFORMAT = '%Y-%m-%d %X'
w.start()


class DataFetch(threading.Thread):
    def __init__(self, code, trading_day, output_filepath):
        self.code = code
        self.trading_day = trading_day
        self.output_filepath = output_filepath
        threading.Thread.__init__(self)

    def run(self):
        filepath = self.output_filepath + "\\%s" % self.code
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for date in self.trading_day:
            try:
                df = ts.get_tick_data(self.code, date=date, pause=random.random()/10.0, retry_count=5)
                if not df.dropna().empty:
                    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(date + ' ' + x, ISOTIMEFORMAT))
                    df = df.set_index('time')
                    df = df.sort_index()
                    df['code'] = np.repeat(self.code, df.shape[0])
                    df.to_csv(filepath + "\\%s" % date + "%s.csv" % self.code, encoding='gbk')
#                    print "download %s" % self.code + " data in %s" % date + " to csv"
            except Exception, e:
                print e

if __name__ == '__main__':
    codelist = w.wset("sectorconstituent", "date=2017-02-22;sectorid=a001010100000000").Data[1]
    codelist_transfer = map(lambda x: x[:6], codelist)
    ipo_date = pd.Series(w.wss(codelist, 'ipo_date').Data[0], index=codelist_transfer)\
        .apply(lambda x: x.strftime("%Y-%m-%d"))
    trading_day_list = pd.Series(w.tdays("2017-02-22", "2017-03-24").Data[0]).apply(lambda x: x.strftime("%Y-%m-%d"))
    P = [DataFetch(code, trading_day=trading_day_list.ix[trading_day_list >= ipo_date[code]],
                   output_filepath="D:\\data\\tick")
         for code in codelist_transfer]

    for p in P:
        p.start()

    for p in P:
        p.join()













