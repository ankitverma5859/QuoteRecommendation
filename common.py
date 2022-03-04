from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))
punctuations = string.punctuation + '“' + '”' + '-' + '’' + '‘' + '—' + '.'
