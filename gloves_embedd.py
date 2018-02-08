# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:01:04 2018

@author: los40
"""
import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)
model1 = gensim.models.Word2Vec(wolilis, size=100,hs=1,negative=0, window=5, min_count=5, workers=4)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        # self.dim = len(word2vec.values().__next__())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(len(self.word2vec.values[w]))], axis=0)
            for words in X
        ])
            
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        #self.dim = len(word2vec.values().__next__())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(len(word2vec.values[w]))], axis=0)
                for words in X
            ])
                
                
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
                
                
                
count_train_w2v = etree_w2v.fit_transform(w2v)
count_test = count_vectorizer.transform(x_test.values)


from shallowlearn.models import GensimFastText
clf = GensimFastText(size=100, min_count=0, loss='hs', iter=3, seed=66)
clf.fit(count_train, y_train)
clf.predict(count_test)








model1.wv.doesnt_match("article subject mathematics".split())

model1.wv.similarity('woman', 'man')

model.wv.most_similar(positive=['article', 'bad'], negative=['good'])



bigram_transformer = gensim.models.Phrases(wolilis)
model = gensim.models.Word2Vec(bigram_transformer[wolilis], size=100)





