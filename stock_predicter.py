#coding: utf-8

import numpy as np
from math import ceil
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from stock_dataset import StockSupervisedDataSet
from matplotlib.pyplot import figure, show, plot
from stock_downloader import StockDownloader

class StockPredicter(object):

    def __init__(self, stock_to_predict, days_of_prediction = 10, days_of_training = 450):

        self.number_of_days_before = 8
        self.days_of_prediction = days_of_prediction

        self.downloader = StockDownloader()

        stock_training_data = self.downloader.download_stock(stock_to_predict, days_of_training, days_of_prediction)
        self.stock_prediction_data = self.downloader.download_stock(stock_to_predict, days_of_prediction)

        #self.starting_price = stock_training_data[-self.number_of_days_before:]
        self.starting_price = self.stock_prediction_data[0]

        self.dataset = StockSupervisedDataSet(self.number_of_days_before, stock_training_data)
        self.network = buildNetwork(self.dataset.indim, 10, self.dataset.outdim, recurrent=True)
        t = BackpropTrainer(self.network, learningrate = 0.00005,  momentum=0., verbose = True)
        t.trainOnDataset(self.dataset, 200)
        t.testOnData(verbose= True)

        self.starting_prices = self.dataset['input'][-1]


    def predict_with_starting_price_only(self):

        #Get predicted price and reverse predicted price
        prices = [self.starting_price]
        augmentations = list(self.starting_prices)
        for i in range(len(self.stock_prediction_data) - 1):

            predicted_augmentation = float(self.network.activate(augmentations[-self.number_of_days_before:]))
            augmentations.append(predicted_augmentation)

            new_price = prices[-1] * (1 + predicted_augmentation)
            prices.append(new_price)

        moving_avg = self.movingaverage(np.array(prices), 5)

        #Plot real prices
        figure()
        plot(self.stock_prediction_data, color = "black")

        #Plot predicted
        plot(moving_avg, "green")
        show()

    def movingaverage(self, interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        avg = np.convolve(interval, window, 'same')

        half_window = ceil(window_size/2)

        avg[:half_window] = interval[:half_window]
        avg[-half_window:] = interval[-half_window:]
        return avg

    def predict_one_day_ahead(self):
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

        plot(real, color = "black")
        plot(predicted, color = "red")
        plot(reverse, "blue")
        show()



if __name__ == "__main__":
    predicter = StockPredicter("AAPL")
    predicter.predict_one_day_ahead()