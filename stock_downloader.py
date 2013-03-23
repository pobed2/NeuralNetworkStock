#coding: utf-8

import urllib2
import datetime
import  numpy as np
from StringIO import StringIO

class StockDownloader(object):
    def download_stock(self, stock, days=28):
        #Bug in API. Have to add 28 to number of days...
        days += 28


        today = datetime.date.today()
        other = today-datetime.timedelta(days=days)

        url = 'http://ichart.yahoo.com/table.csv?s={}&g=d&a={}&b={}&c={}&d={}&e={}&f={}'.format(
            stock,other.month, other.day, other.year,
            today.month, today.day, today.year)

        print url

        response = urllib2.urlopen(url)
        file = response.read()

        print file
        data = np.genfromtxt(StringIO(file), skip_header=1, usecols=(4), delimiter=',')[::-1]
        print data

if __name__ == "__main__":
    downloader = StockDownloader()
    downloader.download_stock("AAPL", 3)