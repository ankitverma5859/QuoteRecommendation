import pandas as pd
import language_models as lm
import llr
import nltk, string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import common


class QuotablePhraseDetection:

    def __init__(self, data_file):
        self.data = open(data_file, encoding='utf-8').read()

    def get_sentences(self):
        data_without_newline = ""
        for line in self.data:
            line_without_newline = line.replace("\n", "")
            data_without_newline += line_without_newline

        sentences = sent_tokenize(data_without_newline)
        return sentences

    def get_probable_quotes(self):
        sentences = self.get_sentences()
        print(f'Total Sentences: {len(sentences)}')
        llr_obj = llr.LLR()
        probable_quotations = []
        for sent in sentences:
            uc_llr = llr_obj.calculate_llr(sent)
            if uc_llr >= 1 and uc_llr <= 25:
                probable_quotations.append(sent)
        print(f'Total Probable Quotations: {len(probable_quotations)}')
        return probable_quotations