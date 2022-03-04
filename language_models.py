import nltk, string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import common

training_data_processed = ""
unigrams = []
stop_words = common.stop_words
punctuations = common.punctuations.replace('.', '')


class UnigramModel:

    def __init__(self, training_data_file):
        global training_data_processed
        self.training_data = open(training_data_file, encoding='utf-8').read()
        self.unigram_probabilities = {}
        self.default_probability = float
        self.preprocess()
        self.fit()

    def preprocess(self):
        global unigrams
        #   Removing the new lines from the training data.
        training_data_without_newline = ""
        for line in self.training_data:
            line_without_newline = line.replace("\n", "")
            training_data_without_newline += line_without_newline

        # Removing the special characters from the training data
        training_data_processed = "".join([char for char in training_data_without_newline
                                                      if char not in punctuations])
        '''
        print(training_data_without_punctuations)
        '''

        sentences = sent_tokenize(training_data_processed)

        # Removing the full stops from the list of words.
        unigrams = []
        for sentence in sentences:
            sentence = sentence.lower()
            sequence = word_tokenize(sentence)
            #print(sequence)
            for word in sequence:
                if word == ".":
                    sequence.remove(word)
                else:
                    unigrams.append(word)

        # Removing the stopwords from the unigrams list
        unigrams = [word for word in unigrams if word not in stop_words]

    def fit(self):
        #print(unigrams)
        total_unigrams = len(unigrams)
        unique_unigrams = set(unigrams)
        total_unique_unigrams = len(unique_unigrams)
        #print(f'Total Unigrams: {total_unigrams}')
        #print(f'Unique Unigrams: {unique_unigrams}')

        self.default_probability = 1 / (total_unigrams + total_unique_unigrams)
        # Calculating the probability of each of the unigrams along with 1-smoothening / Laplace Smoothening
        freq_unigrams = nltk.FreqDist(unigrams)
        for key, value in freq_unigrams.items():
            self.unigram_probabilities[key] = (value + 1) / (total_unigrams + total_unique_unigrams)
        #print(freq_unigrams.most_common(10))

    def get_probability(self, word):
        return self.unigram_probabilities.get(word, self.default_probability)
