import nltk
import llr
from Resources import quantifiers, diag_words, abstract_words
import common

model_parameters = [
    "f_llr",            # sentence log-likelihood ratio
    "n_words",          # number of words in a sentence
    "n_chars",          # number of characters in a sentence
    "n_word len_agg",   # aggregate of word length in s Agg = {min, max, mean} prefer mean
    "n_capital",        # number of uppercase characters
    "n_quantifiers",    # number of universal quantifiers in s (Example: all, whole, nobody)
    "n_stops",          # number of stop words in s
    "b_begin_stop",     # True if sentence begins with a stop word. 1/0 val in eqn
    "b_has_diag",       # True if sentence contains a dialog word i.e ["say", "says", "said"] 1/0 val in eqn
    "n_abstract",       # number of abstract concepts Example ["adventure", "charity", "stupidity"]
                        # abstract are words that do not have any physical existence
    "b_has_p",          # True if sentence contains a punctuations
    "n_nouns",          # number of nouns
                        # NLTK POS Tags: [NN, NNS, NNP, NNPS]
    "n_verbs",          # number of verbs
                        # NLTK POS Tags: [VB, VBD, VBG, VBN, VBP, VBZ]
    "n_adjectives",     # number of adjectives
                        # NLTK POS Tags: [JJ, JJR, JJS]
    "n_adverbs",        # number of adverbs
                        # NLTK POS Tags: [RB, RBR, RBS, WRB]
    "n_pronouns",       # number of pronouns
                        # NLTK POS Tags: [PRP, PRP$, WP, WP$]
    "b_has_comp",       # True if the sentence has a comparative adjective/adverb
                        # NLTK POS Tags: JJR(adjective), RBR(adverb)
    "b_has_super",      # True if the sentence has a superlative adjective/adverb
                        # NLTK POS Tags: JJS(adjective), RBS(adverb)
    "b_has_pp",         # True if the sentence has a past participle
                        # NLTK POS Tags: VBD

                        # Ref for NLTK Tag codes: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
]


class LinguisticFeatures:

    llr_obj = llr.LLR()

    def __init__(self, sentence):
        self.sentence = sentence
        self.f_llr = 0.00
        self.n_words = 0
        self.n_chars = 0
        self.n_word_len_agg = 0.0
        self.n_capital = 0
        self.n_quantifiers = 0
        self.n_stops = 0
        self.b_begin_stop = 0
        self.b_has_diag = 0
        self.n_abs = 0
        self.b_has_quotes = 0
        self.b_has_parenthesis = 0
        self.b_has_colon = 0
        self.b_has_dash = 0
        self.b_has_semicolon = 0
        self.b_has_p = 0
        self.n_nouns = 0
        self.n_verbs = 0
        self.n_adj = 0
        self.n_adv = 0
        self.n_pro = 0
        self.b_has_comp = 0
        self.b_has_super = 0
        self.b_has_pp = 0

        self.calculate_model_feature_values()

    @staticmethod
    def word_len_agg(words):
        word_lens = [len(word) for word in words]
        if len(word_lens) > 0:
            return sum(word_lens) / len(word_lens)
        return 1.0

    @staticmethod
    def num_of_words_with_uppercase(words):
        wu_count = 0
        for word in words:
            if word[0].isupper():
                wu_count += 1
        return wu_count

    @staticmethod
    def count_quantifiers(words):
        q_count = 0
        for word in words:
            if word.lower() in quantifiers.quantifiers:
                q_count += 1
        return q_count

    @staticmethod
    def count_stop_words(words):
        s_count = 0
        for word in words:
            if word.lower() in common.stop_words:
                s_count += 1
        return  s_count

    @staticmethod
    def is_begin_stop(word):
        if word in common.stop_words:
            return 1
        return 0

    @staticmethod
    def has_diag(words):
        for word in words:
            if word in diag_words.diag_words:
                return 1
        return 0

    @staticmethod
    def count_abstract(words):
        a_count = 0
        for word in words:
            if word.lower() in abstract_words.abstract_words:
                a_count += 1
        return a_count

    @staticmethod
    def has_punc(sentence):
        quotations = ["\"", "'"]
        parenthesis = ["(", ")", "{", "}", "[", "]"]
        colon = [":"]
        dash = ["-"]
        semicolon = [";"]
        has_punc = 0
        has_quotations = 0
        has_parenthesis = 0
        has_colon = 0
        has_dash = 0
        has_semicolon = 0
        for char in sentence:
            if char in common.punctuations:
                has_punc = 1
            if char in quotations:
                has_quotations = 1
            if char in parenthesis:
                has_parenthesis = 1
            if char in colon:
                has_colon = 1
            if char in dash:
                has_dash = 1
            if char in semicolon:
                has_semicolon = 1
        return has_punc, has_quotations, has_parenthesis, has_colon, has_dash, has_semicolon

    @staticmethod
    def has_quotations(sentence):
        quotations = ["\"", "'"]
        for char in sentence:
            if char in quotations:
                return 1
        return 0

    @staticmethod
    def has_parenthesis(sentence):
        parenthesis = ["(", ")", "{", "}", "[", "]"]
        for char in sentence:
            if char in parenthesis:
                return 1
        return 0

    @staticmethod
    def has_colon(sentence):
        colon = [":"]
        for char in sentence:
            if char in colon:
                return 1
        return 0

    @staticmethod
    def has_dash(sentence):
        dash = ["-"]
        for char in sentence:
            if char in dash:
                return 1
        return 0

    @staticmethod
    def has_semicolon(sentence):
        semicolon = [";"]
        for char in sentence:
            if char in semicolon:
                return 1
        return 0

    @staticmethod
    def count_pos(words):
        word_tags = nltk.pos_tag(words)
        n_count = 0
        n_verbs = 0
        n_adj = 0
        n_adv = 0
        n_pro = 0
        b_comp = 0
        b_super = 0
        b_pp = 0
        for item in word_tags:
            # NLTK POS Tags for nouns: [NN, NNS, NNP, NNPS]
            if item[1] == 'NN' or item[1] == 'NNS' or item[1] == 'NNP' or item[1] == 'NNPS':
                n_count += 1
            # NLTK POS Tags for verbs: [VB, VBD, VBG, VBN, VBP, VBZ]
            elif item[1] == "VB" or item[1] == "VBD" or item[1] == "VBG" or item[1] == "VBN" or item[1] == "VBP" or item[1] == "VBZ":
                n_verbs += 1
                if item[1] == "VBD":
                    b_pp = 1
            # NLTK POS Tags for adjectives: [JJ, JJR, JJS]
            elif item[1] == "JJ" or item[1] == "JJR" or item[1] == "JJS":
                n_adj += 1
                if item[1] == "JJR":
                    b_comp = 1
                if item[1] == "JJS":
                    b_super = 1
            # NLTK POS Tags for adverbs: [RB, RBR, RBS, WRB]
            elif item[1] == "RB" or item[1] == "RBR" or item[1] == "RBS" or item[1] == "WRB":
                n_adv += 1
                if item[1] == "RBR":
                    b_comp = 1
                if item[1] == "RBS":
                    b_super = 1
            # NLTK POS Tags for pronouns: [PRP, PRP$, WP, WP$]
            elif item[1] == "PRP" or item[1] == "PRP$" or item[1] == "WP" or item[1] == "WP$":
                n_pro += 1
        return n_count, n_verbs, n_adj, n_adv, n_pro, b_comp, b_super, b_pp

    def calculate_model_feature_values(self):

        sentence_without_punc = "".join([char for char in self.sentence if char not in common.punctuations])
        words = nltk.word_tokenize(sentence_without_punc)

        self.f_llr = LinguisticFeatures.llr_obj.calculate_llr(self.sentence)
        self.n_words = len(words)
        self.n_chars = len(sentence_without_punc)
        self.n_word_len_agg = self.word_len_agg(words)
        self.n_capital = self.num_of_words_with_uppercase(words)
        self.n_quantifiers = self.count_quantifiers(words)
        self.n_stops = self.count_stop_words(words)
        self.b_begin_stop = self.is_begin_stop(words[0])
        self.b_has_diag = self.has_diag(words)
        self.n_abs = self.count_abstract(words)
        self.b_has_p, self.b_has_quotes, self.b_has_parenthesis, self.b_has_colon, self.b_has_dash, self.b_has_semicolon = self.has_punc(self.sentence)
        self.n_nouns,  self.n_verbs, self.n_adj, self.n_adv, self.n_pro, self.b_has_comp, self.b_has_super, self.b_has_pp = self.count_pos(words)



