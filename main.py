from sklearn.feature_extraction import text
vectorizer = text.CountVectorizer(binary=True)

corpus = [
    "The quick brown fox jumps over the lazy dog", "My dog is faster than yours", "Your dog is slower than mine"
]

vectorizer.fit(corpus)
vectorized_text = vectorizer.transform(corpus)

# print(vectorized_text.todense())    
# print(vectorizer.vocabulary_)

text_4 = "A black dog just passed by but my dog is white"
corpus.append(text_4)
vectorizer = text.CountVectorizer()

vectorizer.fit(corpus)
vectorized_text = vectorizer.transform(corpus)
# print(vectorized_text.todense())
# print(vectorizer.vocabulary_)

Tfidf = text.TfidfTransformer(norm='l1')
tfid_mtx = Tfidf.fit_transform(vectorized_text)

phrase = 3

total = 0
for word in vectorizer.vocabulary_:
    pos = vectorizer.vocabulary_[word]
    value = list(tfid_mtx.toarray()[phrase])[pos]

    if value != 0:
        print("%10s: %.3f" %(word, value))
        total += value

print("total sum: ", total)