'''
Utility class with some helper functions
'''

import pandas as pd
import numpy as np
import random
import os

class Util(object):

    def __init__(self):
        self.ratings = None
        self.to_read = None
        self.books = None

    def read_data(self, folder):
        '''
        Function to read data required to
        build the recommender system
        '''
        self.ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))
        self.to_read = pd.read_csv(os.path.join(folder, "to_read.csv"))
        self.books = pd.read_csv(os.path.join(folder, "books.csv"))

    def clean_subset(self, num_rows):
        '''
        Function to clean and subset the data according
        to individual machine power
        '''
        temp = self.ratings.sort_values(by=['user_id'], ascending=True)
        self.ratings = temp.iloc[:num_rows, :]

    def preprocess(self):
        '''
        Preprocess data for feeding into the network
        '''
        self.ratings = self.ratings.reset_index(drop=True)
        self.ratings['List Index'] = self.ratings.index
        readers_group = self.ratings.groupby("user_id")

        total = []
        for readerID, curReader in readers_group:
            temp = np.zeros(len(self.ratings))
            for num, book in curReader.iterrows():
                temp[book['List Index']] = book['rating']/5.0
            total.append(temp)

        return total

    def split_data(self):
        '''
        Function to split into training and validation sets
        '''
        total_data = self.preprocess()
        random.shuffle(total_data)
        print("total size of the data is: {0}".format(len(total_data)))
        X_train = total_data[:1500]
        X_valid = total_data[1500:]
        print("size of the training data is: {0}".format(len(X_train)))
        print("size of the validation data is: {0}".format(len(X_valid)))
        return X_train, X_valid

    def free_energy(self, v_sample, W, vb, hb):
        '''
        Function to compute the free energy
        '''
        wx_b = np.dot(v_sample, W) + hb
        vbias_term = np.dot(v_sample, vb)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis = 1)
        return -hidden_term - vbias_term
