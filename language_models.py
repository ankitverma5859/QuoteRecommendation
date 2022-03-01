import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_data_processed = ""
unigrams = []
stop_words = set(stopwords.words('english'))
string.punctuation = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—'
string.punctuation = string.punctuation.replace('.', '')


class UnigramModel:

    def __init__(self, training_data_file):
        global training_data_processed
        self.training_data = open(training_data_file, encoding='utf-8').read()
        self.unigram_probabilities = {}
        self.default_probability = float

    def preprocess(self):
        global unigrams
        #   Removing the new lines from the training data.
        training_data_without_newline = ""
        for line in self.training_data:
            line_without_newline = line.replace("\n", "")
            training_data_without_newline += line_without_newline

        # Removing the special characters from the training data
        training_data_processed = "".join([char for char in training_data_without_newline
                                                      if char not in string.punctuation])
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
        print(unigrams)
        total_unigrams = len(unigrams)
        unique_unigrams = set(unigrams)
        total_unique_unigrams = len(unique_unigrams)
        print(f'Total Unigrams: {total_unigrams}')
        print(f'Unique Unigrams: {unique_unigrams}')

        self.default_probability = 1 / (total_unigrams + total_unique_unigrams)
        # Calculating the probability of each of the unigrams along with 1-smoothening / Laplace Smoothening
        freq_unigrams = nltk.FreqDist(unigrams)
        for key, value in freq_unigrams.items():
            self.unigram_probabilities[key] = (value + 1) / (total_unigrams + total_unique_unigrams)
        print(freq_unigrams.most_common(10))

    def get_probability(self, word):
        return self.unigram_probabilities.get(word, self.default_probability)




    '''
    def fit(self):
        sentences = sent_tokenize(training_data_processed)
        words = word_tokenize(training_data_processed)
        average_tokens = round(len(words) / len(sentences))
        unique_tokens = set(words)

        print(f'The number of sentences:\t{len(sentences)}')
        print(f'The number of words:\t{len(words)}')
        print(f'Average tokens:\t{average_tokens}')
        print(f'Unique tokens:\t{len(unique_tokens)}')


        unigram = []
        unigram_probabilities = {}
        tokenized_text = []

        for sentence in sentences:
            sentence = sentence.lower()
            sequence = word_tokenize(sentence)
            #print(sequence)
            for word in sequence:
                if word == ".":
                    sequence.remove(word)
                else:
                    unigram.append(word)
            tokenized_text.append(sequence)

        freq_uni = nltk.FreqDist(unigram)
        print(f'Number of words in unigram with stopwords: {len(unigram)}')
        #print(freq_uni.most_common(10))
        '''

    '''
        pd_distribution = pd.Series(dict(freq_uni))
        fig, ax = plt.subplots(figsize=(10, 10))

        all_plot = sns.barplot(x=pd_distribution.index, y=pd_distribution.values, ax=ax)
        plt.xticks(rotation=30)
        plt.show()
        '''

    '''
        unigram_sw_removed = [word for word in unigram if word not in stop_words]
        unigram_sw_removed_len = len(unigram_sw_removed)
        print(f'Number of words after removing the stopwords: {unigram_sw_removed_len}')
        unique_tokens = set(unigram_sw_removed)
        print(f'Number of unique words after removing stopword: {len(unique_tokens)}')
        freq_uni = nltk.FreqDist(unigram_sw_removed)
        print(freq_uni.most_common(10))

        for key, value in freq_uni.items():
            #print(f'Key: {key} Value: {value} Probability: {value/unigram_sw_removed_len}')
            #unigram_probabilities[key] = value/unigram_sw_removed_len
            unigram_probabilities[key] = (value + 1) / (unigram_sw_removed_len + len(unique_tokens))

        dict(sorted(unigram_probabilities.items(), key=lambda item: item[1]))
        #for key, value in unigram_probabilities.items():
        #    print(f'Key: {key} Probability: {value}')

        print(f'Mr. {unigram_probabilities["mr."]}')
        print(f'Mr. {unigram_probabilities["elizabeth"]}')
        print(f'Mr. {unigram_probabilities["could"]}')
        print(f'Mr. {unigram_probabilities["would"]}')
        print(f'Mr. {unigram_probabilities.get("said", -1)}')
        '''














