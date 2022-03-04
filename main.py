from Resources import abstract_words
import linguistic_features
import language_models as lm
import llr
import nltk
import math
import pandas as pd
#nltk.download()

if __name__ == '__main__':

    # Calculating the feature values of the quotations

    quotations_file = open('Resources/Quotations/mg_quotes.txt', 'r')
    quotes = quotations_file.readlines()
    quotations_file.close()

    feature_values_file = open('Resources/Quotations/mg_quotes_fvalues.csv', 'w')
    feature_values_file.writelines(f'llr, words, chars, wd_len_agg, capitals, quantifiers, sw, '
                                   f'begins_with_sw, has_diag, abs, has_quotations, has_parenthesis, '
                                   f'has_colon, has_dash, has_semicolon, nouns, verbs, adj, adv, '
                                   f'has_comp, has_super, has_pp\n')
    for quote in quotes:
        mf = linguistic_features.LinguisticFeatures(quote)

        feature_values_file.writelines(f'{mf.f_llr}, {mf.n_words}, {mf.n_chars}, {mf.n_word_len_agg}, '
                                       f'{mf.n_capital}, {mf.n_quantifiers}, {mf.n_stops}, {mf.b_begin_stop}, '
                                       f'{mf.b_has_diag}, {mf.n_abs}, {mf.b_has_quotes}, {mf.b_has_parenthesis}, '
                                       f'{mf.b_has_colon}, {mf.b_has_dash}, {mf.b_has_semicolon}, {mf.n_nouns}, '
                                       f'{mf.n_verbs}, {mf.n_adj}, {mf.n_adv}, {mf.b_has_comp}, {mf.b_has_super}, '
                                       f'{mf.b_has_pp}\n')

    feature_values_file.close()
    quotations_file.close()

    features_df = pd.read_csv('Resources/Quotations/mg_quotes_fvalues.csv')
    print(features_df.head(5))


