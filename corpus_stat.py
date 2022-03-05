import json


class CorpusStat:

    def __init__(self):

        with open('Resources/Stats/corpus_stats.json') as f:
            data = f.read()

        self.data_dict = json.loads(data)

        count = 0
        for item, val in self.data_dict.items():
            count += val

        self.total_words = count
        self.unique_words = len(self.data_dict)
        self.default_probability = 1 / (self.total_words + self.unique_words)

    def get_probability(self, word):
        word_count = self.data_dict.get(word, 0)
        probability = (word_count + 1 ) / (self.total_words + self.unique_words)
        return probability
