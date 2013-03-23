#coding: utf-8

import numpy as np
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from datasets.stock_dataset import StockSupervisedDataSet
from matplotlib.pyplot import ion, ioff, figure, draw, clf, show, hold, plot
from stock_downloader import StockDownloader

class StockPredicter(object):

    def __init__(self, stock_to_predict):

        self.number_of_days_before = 8
        self.downloader = StockDownloader()

        #stock_previous_data = self.downloader.download_stock(stock_to_predict, 365)

        stock_previous_data = np.genfromtxt("apple-small.csv", skip_header=1, usecols=(4), delimiter=',')[::-1]

        self.starting_price = stock_previous_data[self.number_of_days_before]

        self.dataset = StockSupervisedDataSet(self.number_of_days_before, stock_previous_data)
        self.network = buildNetwork(self.dataset.indim, 10, self.dataset.outdim, recurrent=True)
        t = BackpropTrainer(self.network, learningrate = 0.00005,  momentum=0., verbose = True)
        t.trainOnDataset(self.dataset, 200)
        t.testOnData(verbose= True)

    def predict(self):
        figure()

        #Get predicted price and reverse predicted price
        reverse = []
        predicted = []
        price = self.starting_price
        reverse_price = self.starting_price
        for i in range(len(self.dataset)):
            if i >= self.number_of_days_before:
                predicted_augmentation = self.network.activate(self.dataset['input'][i-self.number_of_days_before])
                price = price * (1 + predicted_augmentation)
                reverse_price = reverse_price * (1 - predicted_augmentation)
                print "Price: " ,price
                predicted.append(price)
                reverse.append(reverse_price)

        #Get real price
        price = self.starting_price
        real = []
        for i, target in enumerate(self.dataset['target']):
            if i >= self.number_of_days_before:
                target = self.dataset['target'][i-self.number_of_days_before]
                price = price * (1+target)
                real.append(price)

        print predicted
        print reverse

        plot(real, color = "black")
        plot(predicted, color = "red")
        plot(reverse, "blue")
        show()

if __name__ == "__main__":
    predicter = StockPredicter("AAPL")
    predicter.predict()