from sklearn.feature_extraction import text

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def stem_tokens(stemmer, tokens):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))

    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    stemmed_words = stem_tokens(stemmer, tokens)

    return stemmed_words


# corpus = ["The quick brown fox jumps over the lazy dog", "Kill him not save him", "John said don't stop him"]
vocab = ["Sam loves coding so he codes all the time","The quick brown fox jumps over the lazy dog", "Kill him not save him", "John said don't stop him"]

vect = text.CountVectorizer(tokenizer=tokenize)
vec = vect.fit(vocab)

sentence = vec.transform(vocab)
print(sentence.toarray())
print(vec.get_feature_names_out())