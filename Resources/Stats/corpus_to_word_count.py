import nltk
import glob
import re
from nltk import ngrams
from collections import Counter
from common import punctuations, stop_words

# Calculating word counts for the corpus
# Put all extra stop words in lowercase
more_stops = ["footnote", "p", "ii", "sq", "cit", "op", "de", "sqq", "pp", "der", "cf", "iii", "2", "le",
              "et", "des", "la", "que", "v", "en", "el", "project", "gutenberg", "ebook", "wwwgutenbergorg"]
stop_words.update(more_stops)


corpus = []
filepaths = glob.glob("Resources/Books/*.txt", recursive=True)
for book in filepaths:
    file = open(book)
    book_data = file.read()
    # Removing the special characters from the book data
    book_data_without_punc = "".join([char for char in book_data
                                       if char not in punctuations])

    corpus.append(book_data_without_punc.lower())

frequencies = Counter([])
f = Counter([])
for book_id in range(0, len(corpus)):
    word_tokens = nltk.word_tokenize(corpus[book_id])
    tokens = [word for word in word_tokens if word.lower() not in stop_words and not word.isdigit()]
    unigrams = ngrams(tokens, 1)
    f += Counter(unigrams)
    if book_id % 2 == 0:
        frequencies += f
        f = Counter([])

file = open('Resources/Stats/CorpusWordCounts.txt', 'w')
chars = r'[()\',]*'
for item in frequencies:
    line = "\"" + re.sub(chars, "", str(item)) + "\": " + str(frequencies[item]) + ",\n"
    file.write(line)
    print(f'"{re.sub(chars, "", str(item))}":{frequencies[item]},')
file.close()
