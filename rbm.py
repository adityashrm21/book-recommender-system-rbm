import pandas as pd
import numpy as np
import tensorflow as tf
from utils import Util
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class RBM(object):
    '''
    Class definition for a simple RBM
    '''
    def __init__(self, alpha, H, num_vis):

        self.alpha = alpha
        self.num_hid = H
        self.num_vis = num_vis # might face an error here, call preprocess if you do
        self.errors = []
        self.energy_train = []
        self.energy_valid = []

    def training(self, train, valid, user, epochs, batchsize, free_energy, verbose):
        '''
        Function where RBM training takes place
        '''
        vb = tf.placeholder(tf.float32, [self.num_vis]) # Number of unique books
        hb = tf.placeholder(tf.float32, [self.num_hid]) # Number of features were going to learn
        W = tf.placeholder(tf.float32, [self.num_vis, self.num_hid])  # Weight Matrix
        v0 = tf.placeholder(tf.float32, [None, self.num_vis])

        print("Phase 1: Input Processing")
        _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
        # Gibb's Sampling
        h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
        print("Phase 2: Reconstruction")
        _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
        v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
        h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

        print("Creating the gradients")
        w_pos_grad = tf.matmul(tf.transpose(v0), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)

        # Calculate the Contrastive Divergence to maximize
        CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

        # Create methods to update the weights and biases
        update_w = W + self.alpha * CD
        update_vb = vb + self.alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = hb + self.alpha * tf.reduce_mean(h0 - h1, 0)

        # Set the error function, here we use Mean Absolute Error Function
        err = v0 - v1
        err_sum = tf.reduce_mean(err * err)

        # Initialize our Variables with Zeroes using Numpy Library
        # Current weight
        cur_w = np.zeros([self.num_vis, self.num_hid], np.float32)
        # Current visible unit biases
        cur_vb = np.zeros([self.num_vis], np.float32)

        # Current hidden unit biases
        cur_hb = np.zeros([self.num_hid], np.float32)

        # Previous weight
        prv_w = np.random.normal(loc=0, scale=0.01,
                                size=[self.num_vis, self.num_hid])
        # Previous visible unit biases
        prv_vb = np.zeros([self.num_vis], np.float32)

        # Previous hidden unit biases
        prv_hb = np.zeros([self.num_hid], np.float32)

        print("Running the session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        print("Training RBM with {0} epochs and batch size: {1}".format(epochs, batchsize))
        print("Starting the training process")
        util = Util()
        for i in range(epochs):
            for start, end in zip(range(0, len(train), batchsize), range(batchsize, len(train), batchsize)):
                batch = train[start:end]
                cur_w = sess.run(update_w, feed_dict={
                                 v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={
                                  v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb

            if valid:
                etrain = np.mean(util.free_energy(train, cur_w, cur_vb, cur_hb))
                self.energy_train.append(etrain)
                evalid = np.mean(util.free_energy(valid, cur_w, cur_vb, cur_hb))
                self.energy_valid.append(evalid)
            self.errors.append(sess.run(err_sum, feed_dict={
                          v0: train, W: cur_w, vb: cur_vb, hb: cur_hb}))
            if verbose:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))
            elif i % 10 == 9:
                print("Error after {0} epochs is: {1}".format(i+1, self.errors[i]))

        if free_energy:
            print("Exporting free energy plot")
            self.export_free_energy_plot()
        print("Exporting errors vs epochs plot")
        self.export_errors_plot()
        inputUser = [train[user]]
        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})
        return rec, prv_w, prv_vb, prv_hb

    def calculate_scores(self, ratings, books, to_read, rec, user):
        '''
        Function to obtain recommendation scores for a user
        using the trained weights
        '''
        # Creating recommendation score for books in our data
        ratings["Recommendation Score"] = rec[0]

        """ Recommend User what books he has not read yet """
        # Find the mock user's user_id from the data
        cur_user_id = ratings.iloc[user]['user_id']

        # Find all books the mock user has read before
        read_books = ratings[ratings['user_id'] == cur_user_id]['book_id']
        read_books

        # converting the pandas series object into a list
        read_books_id = read_books.tolist()

        # getting the book names and authors for the books already read by the user
        read_books_names = []
        read_books_authors = []
        for book in read_books_id:
            read_books_names.append(
                books[books['book_id'] == book]['original_title'].tolist()[0])
            read_books_authors.append(
                books[books['book_id'] == book]['authors'].tolist()[0])

        # Find all books the mock user has 'not' read before using the to_read data
        unread_books = to_read[to_read['user_id'] == cur_user_id]['book_id']
        unread_books_id = unread_books.tolist()

        # extract the ratings of all the unread books from ratings dataframe
        unread_with_score = ratings[ratings['book_id'].isin(unread_books_id)]

        # grouping the unread data on book id and taking the mean of the recommendation scores for each book_id
        grouped_unread = unread_with_score.groupby('book_id', as_index=False)[
            'Recommendation Score'].mean()

        # getting the names and authors of the unread books
        unread_books_names = []
        unread_books_authors = []
        unread_books_scores = []
        for book in grouped_unread['book_id']:
            unread_books_names.append(
                books[books['book_id'] == book]['original_title'].tolist()[0])
            unread_books_authors.append(
                books[books['book_id'] == book]['authors'].tolist()[0])
            unread_books_scores.append(
                grouped_unread[grouped_unread['book_id'] == book]['Recommendation Score'].tolist()[0])

        # creating a data frame for unread books with their names, authors and recommendation scores
        unread_books_with_scores = pd.DataFrame({
            'book_name': unread_books_names,
            'book_authors': unread_books_authors,
            'score': unread_books_scores
        })

        # creating a data frame for read books with the names and authors
        read_books_with_names = pd.DataFrame({
            'book_name': read_books_names,
            'book_authors': read_books_authors
        })

        return unread_books_with_scores, read_books_with_names

    def export(self, unread, read):
        '''
        Function to export the final result for a user into csv format
        '''
        # sort the result in descending order of the recommendation score
        sorted_result = unread.sort_values(
            by='score', ascending=False)

        # exporting the read and unread books  with scores to csv files
        read.to_csv('results/read_books_with_names.csv')
        sorted_result.to_csv('results/unread_books_with_scores.csv')
        print('The books read by the user are:')
        print(read)
        print('The books recommended to the user are:')
        print(sorted_result)

    def export_errors_plot(self):
        plt.plot(self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.savefig("error.png")

    def export_free_energy_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.energy_train, label='train')
        ax.plot(self.energy_valid, label='valid')
        leg = ax.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Free Energy")
        plt.savefig("free_energy.png")
