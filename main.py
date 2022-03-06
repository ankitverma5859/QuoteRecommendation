import re
import numpy as np
from Resources import abstract_words
import linguistic_features
import language_models as lm
import llr

import math
import pandas as pd
import quote_prediction
#nltk.download()
from os import walk
import nltk
from nltk import ngrams
from common import punctuations, stop_words
import json
import perceptron as pc


import corpus_stat


if __name__ == '__main__':


    #book = open('Resources/Books/ATaleOfTwoCities.txt', 'r')
    #book_data = book.read()
    #book.close()

    '''
    feature_values_file = open('Resources/Quotations/train_data.csv', 'w')
    feature_values_file.writelines(f'llr, words, chars, wd_len_agg, capitals, quantifiers, sw, '
                                   f'begins_with_sw, has_diag, abs, has_quotations, has_parenthesis, '
                                   f'has_colon, has_dash, has_semicolon, nouns, verbs, adj, adv, '
                                   f'has_comp, has_super, has_pp,label\n')

    quotations_file = open('Resources/Quotations/Quotations.txt', 'r')
    quotes = quotations_file.readlines()
    quotations_file.close()

    for quote in quotes:
        mf = linguistic_features.LinguisticFeatures(quote)

        feature_values_file.writelines(f'{mf.f_llr}, {mf.n_words}, {mf.n_chars}, {mf.n_word_len_agg}, '
                                       f'{mf.n_capital}, {mf.n_quantifiers}, {mf.n_stops}, {mf.b_begin_stop}, '
                                       f'{mf.b_has_diag}, {mf.n_abs}, {mf.b_has_quotes}, {mf.b_has_parenthesis}, '
                                       f'{mf.b_has_colon}, {mf.b_has_dash}, {mf.b_has_semicolon}, {mf.n_nouns}, '
                                       f'{mf.n_verbs}, {mf.n_adj}, {mf.n_adv}, {mf.b_has_comp}, {mf.b_has_super}, '
                                       f'{mf.b_has_pp},1\n')

    nonquotations_file = open('Resources/Quotations/non_quotes.txt', 'r')
    nonquotes = nonquotations_file.readlines()
    nonquotations_file.close()

    for nquote in nonquotes:
        mf = linguistic_features.LinguisticFeatures(nquote)

        feature_values_file.writelines(f'{mf.f_llr}, {mf.n_words}, {mf.n_chars}, {mf.n_word_len_agg}, '
                                       f'{mf.n_capital}, {mf.n_quantifiers}, {mf.n_stops}, {mf.b_begin_stop}, '
                                       f'{mf.b_has_diag}, {mf.n_abs}, {mf.b_has_quotes}, {mf.b_has_parenthesis}, '
                                       f'{mf.b_has_colon}, {mf.b_has_dash}, {mf.b_has_semicolon}, {mf.n_nouns}, '
                                       f'{mf.n_verbs}, {mf.n_adj}, {mf.n_adv}, {mf.b_has_comp}, {mf.b_has_super}, '
                                       f'{mf.b_has_pp},0\n')
    feature_values_file.close()
    '''

    # Training Data
    features_df = pd.read_csv('Resources/Quotations/train_data.csv')

    # print(features_df.head())
    shuffled_df = features_df.values
    np.random.shuffle(shuffled_df)

    '''
    # split for test & train
    train_df = shuffled_df[0:5700]
    test_df = shuffled_df[5700:]
    test_set = list()
    for row in test_df:
        row_copy = list(row)
        row_copy.pop()
        test_set.append(row_copy)

    pc_obj = pc.Perceptron(train_df, test_df)
    predicted = pc_obj.perceptron(train_df, test_set, 0.01, 500)
    print(predicted)
    '''

    # Input Book Data Processing
    qp_obj = quote_prediction.QuotablePhraseDetection('Resources/Books/TheGreatGatsby.txt')
    probable_quotes = qp_obj.get_probable_quotes()
    fp = open('Resources/Quotations/probable_quotes.txt', 'w')
    for quote in probable_quotes:
        fp.write(quote)
        fp.write("\n\n")
    fp.close()

    probable_quotes_file = open('Resources/Quotations/probable_quotes_file.csv', 'w')
    probable_quotes_file.writelines(f'llr, words, chars, wd_len_agg, capitals, quantifiers, sw, '
                                f'begins_with_sw, has_diag, abs, has_quotations, has_parenthesis, '
                                f'has_colon, has_dash, has_semicolon, nouns, verbs, adj, adv, '
                                f'has_comp, has_super, has_pp\n')
    for quote in probable_quotes:
        mf = linguistic_features.LinguisticFeatures(quote)

        probable_quotes_file.writelines(f'{mf.f_llr}, {mf.n_words}, {mf.n_chars}, {mf.n_word_len_agg}, '
                                    f'{mf.n_capital}, {mf.n_quantifiers}, {mf.n_stops}, {mf.b_begin_stop}, '
                                    f'{mf.b_has_diag}, {mf.n_abs}, {mf.b_has_quotes}, {mf.b_has_parenthesis}, '
                                    f'{mf.b_has_colon}, {mf.b_has_dash}, {mf.b_has_semicolon}, {mf.n_nouns}, '
                                    f'{mf.n_verbs}, {mf.n_adj}, {mf.n_adv}, {mf.b_has_comp}, {mf.b_has_super}, '
                                    f'{mf.b_has_pp}\n')
    probable_quotes_file.close()

    probable_quotes_df = pd.read_csv('Resources/Quotations/probable_quotes_file.csv')
    probable_quotes_data = probable_quotes_df.values

    pc_obj1 = pc.Perceptron(shuffled_df, probable_quotes_data)
    predicted = pc_obj1.perceptron(shuffled_df, probable_quotes_data, 0.01, 100)
    # print(predicted)
    print("Predicted Quotes: ")
    pq = open('Resources/Quotations/predicted_quotes.txt', 'w')
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == 1.0:
            print(f'Q{i}')
            print(probable_quotes[i])
            pq.write(probable_quotes[i])
            pq.write("\n\n")
            count += 1
    pq.close()
    print(f"Total Predicted quotes : {count}")




    #cs_ob = corpus_stat.CorpusStat()
    #print(cs_ob.get_probability("india"))
    # Calculating the feature values of the quotations

    #with open('Resources/Stats/corpus_stats.json') as f:
    '''
        data = f.read()

    print(type(data))

    js = json.loads(data)

    print(type(js))

    count = 0
    for item, val in js.items():
        count += val
    print(count)
    '''



    '''
    llr_obj = llr.LLR()

    book = open('Resources/Books/PrideAndPrejudice.txt', 'r')
    book_data = book.read()

    non_quotes = open('Resources/Quotations/non_quotes_1.txt', 'w')
    sentences = nltk.sent_tokenize(book_data)
    print(f'Total Sentences {len(sentences)}')
    count = 0
    for sent in sentences:
        llr_val = llr_obj.calculate_llr(sent)
        if llr_val > 1 and llr_val < 25:
            print(f'Count: {count}')
            print(f'LLR: {llr_val}')
            non_quotes.write(sent)
            non_quotes.write("\nend\n")
            print(sent)
            print("\nend\n")
            count += 1

    non_quotes.close()
    book.close()
    '''


    '''
    quotations_file = open('Resources/Quotations/mg_quotes.txt', 'r')
    non_quotes_file = open('Resources/Quotations/non_quotes.txt')
    quotes = quotations_file.readlines()
    non_quotes = non_quotes_file.readlines()
    quotations_file.close()
    non_quotes_file.close()

    feature_values_file = open('Resources/Quotations/mg_quotes_fvalues.csv', 'w')
    feature_values_file.writelines(f'llr, words, chars, wd_len_agg, capitals, quantifiers, sw, '
                                   f'begins_with_sw, has_diag, abs, has_quotations, has_parenthesis, '
                                   f'has_colon, has_dash, has_semicolon, nouns, verbs, adj, adv, '
                                   f'has_comp, has_super, has_pp, label\n')
    for quote in quotes:
        mf = linguistic_features.LinguisticFeatures(quote)

        feature_values_file.writelines(f'{mf.f_llr}, {mf.n_words}, {mf.n_chars}, {mf.n_word_len_agg}, '
                                       f'{mf.n_capital}, {mf.n_quantifiers}, {mf.n_stops}, {mf.b_begin_stop}, '
                                       f'{mf.b_has_diag}, {mf.n_abs}, {mf.b_has_quotes}, {mf.b_has_parenthesis}, '
                                       f'{mf.b_has_colon}, {mf.b_has_dash}, {mf.b_has_semicolon}, {mf.n_nouns}, '
                                       f'{mf.n_verbs}, {mf.n_adj}, {mf.n_adv}, {mf.b_has_comp}, {mf.b_has_super}, '
                                       f'{mf.b_has_pp}, 1\n')

    for sent in non_quotes:
        mf = linguistic_features.LinguisticFeatures(sent)

        feature_values_file.writelines(f'{mf.f_llr}, {mf.n_words}, {mf.n_chars}, {mf.n_word_len_agg}, '
                                       f'{mf.n_capital}, {mf.n_quantifiers}, {mf.n_stops}, {mf.b_begin_stop}, '
                                       f'{mf.b_has_diag}, {mf.n_abs}, {mf.b_has_quotes}, {mf.b_has_parenthesis}, '
                                       f'{mf.b_has_colon}, {mf.b_has_dash}, {mf.b_has_semicolon}, {mf.n_nouns}, '
                                       f'{mf.n_verbs}, {mf.n_adj}, {mf.n_adv}, {mf.b_has_comp}, {mf.b_has_super}, '
                                       f'{mf.b_has_pp}, 0\n')

    feature_values_file.close()
    quotations_file.close()

    #features_df = pd.read_csv('Resources/Quotations/mg_quotes_fvalues.csv')
    #print(features_df.head(5))
    '''




