import language_models as lm
from nltk.corpus import stopwords
import nltk
import string
import math

stop_words = set(stopwords.words('english'))
punctuations = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—' + '.'


class LLR:
    def __init__(self):
        self.um_q = lm.UnigramModel('Resources/Quotations/Quotations.txt')    # um_q: unigram model for quotations
        self.um_c = lm.UnigramModel('Resources/Books/PrideAndPrejudice.txt')  # um_c: unigram model for corpus
                                                                              # Corpus should contain more books
                                                                              # else it will be biased

    def calculate_llr(self, sentence):
        global punctuations
        llr = 0.00
        s_without_punc = "".join([char for char in sentence if char not in punctuations])

        words = nltk.word_tokenize(s_without_punc)
        words_without_sw = [word for word in words if word not in stop_words]

        for word in words_without_sw:
            llr += math.log(self.um_q.get_probability(word) / self.um_c.get_probability(word))
        return llr




