#coding: utf-8

import urllib2
import datetime
import  numpy as np
from StringIO import StringIO

class StockDownloader(object):
    def download_stock(self, stock, stop_days_before_start, start_days_before_today = 0):

        today = datetime.date.today() - datetime.timedelta(days=start_days_before_today)
        other = today - datetime.timedelta(days=stop_days_before_start)

        url = 'http://ichart.yahoo.com/table.csv?s={}&g=d&a={}&b={}&c={}&d={}&e={}&f={}'.format(
            stock, other.month - 1, other.day, other.year,
            today.month - 1, today.day, today.year)

        response = urllib2.urlopen(url)
        file = response.read()

        #print file
        data = np.genfromtxt(StringIO(file), skip_header=1, usecols=(4), delimiter=',')[::-1]
        #print data

        return data

if __name__ == "__main__":
    downloader = StockDownloader()
    downloader.download_stock("AAPL", 14, 4)