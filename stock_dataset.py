#coding: utf-8

from pybrain.datasets import SupervisedDataSet


class StockSupervisedDataSet(SupervisedDataSet):
    """ A dataset for stock prediction."""
    def __init__(self, number_of_days_before, quotes):
        SupervisedDataSet.__init__(self, number_of_days_before, 1)

        gains = []
        for i, quote in enumerate(quotes):
            if i >= 1:
                gain = (quote - quotes[i-1])/quotes[i-1]
                gains.append(gain)

        for i, quote in enumerate(gains):
            if i >= number_of_days_before:
                first_day = i - number_of_days_before
                input = gains[first_day:i]
                output = [gains[i]]

                self.addSample(input, output)