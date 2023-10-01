import pandas as pd
import numpy as np

filename = 'https://github.com/lmassaron/datasets/releases/download/1.0/shakespeare_lines_in_plays.feather'
shakespear = pd.read_feather(filename)

# print(shakespear.head())
# index = shakespear.play + ' act: ' + shakespear.act
# print(index)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df = 1.0, min_df = 3.0, stop_words = 'english')
tfidf = vectorizer.fit_transform(shakespear.lines)

n_topics = 10
from sklearn.decomposition import NMF
nmf = NMF(n_components=n_topics, max_iter=999, random_state=101)
nmf.fit(tfidf)

def print_topic_words(features, topics, top=5):
    for idx, topic in enumerate(topics):
        words = " ".join([features[i] for i in topic.argsort()[:(-top):-1]])
        print(f"Topic #{idx:2.0f}: {words}")

print_topic_words(vectorizer.get_feature_names_out(), nmf.components_)

index = shakespear.play + ' act: ' + shakespear.act

def find_top_match(model, data, index, topic_num=8):
    topic_scores = model.transform(data)[:, topic_num]
    best_score = np.argmax(topic_scores)
    print(best_score)
    print(index[best_score])

find_top_match(nmf, tfidf, index)