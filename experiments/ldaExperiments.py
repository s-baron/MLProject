import itertools
import numpy as np
from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

n_topics = [("n_topics", x) for x in [30, 40, 50, 60, 70, 80, 100]]
vectorizer = [("vectorizer", x) for x in ["tfidf"]]
ngram_range = [("ngram_range", (1,x)) for x in range(2,3)]
stop_wordsList = [("stop_words",x) for x in ["english"]]
max_features = [("max_features", x) for x in [100]]
#norm = [("norm", x) for x in ["l2", None]]
C = [("C", x) for x in [1.0]]
allParams = [vectorizer, stop_wordsList, max_features, ngram_range, C, n_topics]

kwargsList = [{key:value for key, value in kwargs} for kwargs in itertools.product(*allParams)]
nameList = [str([(key, value) for key, value in kwargs.iteritems()]) for kwargs in kwargsList]

def simpleLda(X_train, y_train, X_test, y_test, **kwargs):
    cv = None
    if kwargs["vectorizer"] == "tfidf":
        cv = TfidfVectorizer(ngram_range=kwargs["ngram_range"],
                             stop_words=kwargs["stop_words"],
                             max_features=kwargs["max_features"])
    else:
        cv = CountVectorizer(ngram_range=kwargs["ngram_range"],
                             stop_words=kwargs["stop_words"],
                             max_features=kwargs["max_features"])
    cv.fit(X_train)
    train = cv.transform(X_train)
    test = cv.transform(X_test)
    lda = LatentDirichletAllocation(n_topics=kwargs["n_topics"], 
                                    learning_method="batch")
    lda.fit(train)
    ldaTrain = lda.transform(train)
    ldaTest = lda.transform(test)
    clf = LogisticRegression(class_weight="balanced",
                             C=kwargs["C"])
    clf.fit(ldaTrain, y_train)
    y_pred = clf.decision_function(ldaTest)
    return y_pred

def biasedNGramsAndLda(X_train, y_train, X_test, y_test):
    # ---------------- Fit LDA ------------------------------------
    cv = TfidfVectorizer(ngram_range=(1,2),
                         stop_words="english",
                         max_features=100)
    cv.fit(X_train)
    train = cv.transform(X_train)
    test = cv.transform(X_test)
    lib_indices = np.where(y_train == 1)
    con_indices = np.where(y_train == -1)
    conLda = LatentDirichletAllocation(n_topics=75, 
                                       learning_method="batch")
    libLda = LatentDirichletAllocation(n_topics=75, 
                                       learning_method="batch")
    conLda.fit(train[con_indices])
    #print "Top Conservative words:"
    #print_top_words(conLda, cv.get_feature_names(), 10)
    libLda.fit(train[lib_indices])
    #print "\n"
    #print "Top Liberal words:"
    #print_top_words(libLda, cv.get_feature_names(), 10)
    #print "\n\n"
    # ---------------- Fit Biased NGrams ---------------------------
    X_lib = X_train[lib_indices]
    X_con = X_train[con_indices]

    dtms = []
    vocabularies = []
    for X in [X_lib, X_con]:
        cv = CountVectorizer(ngram_range=(1, 3))
        dtm = cv.fit_transform(X)
        dtms.append(dtm)
        vocabularies.append(cv.vocabulary_)
    vocabSets = [set(vocab.keys()) for vocab in vocabularies]
    dtms[0] = dtms[0].toarray()
    dtms[1] = dtms[1].toarray()
    wordsInBoth = vocabSets[0].intersection(vocabSets[1])
    vocab = []
    for word in wordsInBoth:
        a = dtms[0][:, vocabularies[0][word]]
        b = dtms[1][:, vocabularies[1][word]]
        t, prob = stats.ttest_ind(a, b)
        if prob < 0.1:
            vocab.append(word)
    cv = CountVectorizer(ngram_range=(1, 3), vocabulary=vocab)
    cv.fit(X_train)

    # ---------------- Create feature vectors and train classifier---
    # print "Start"
    # print sorted(conLda.transform(train)[0])
    # print sorted(libLda.transform(train)[0])
    # print sorted(conLda.transform(train)[1])
    # print sorted(libLda.transform(train)[1])
    # print sorted(conLda.transform(train)[2])
    # print sorted(libLda.transform(train)[2])
    ldaTrainItermediate = np.append(conLda.transform(train), 
                         libLda.transform(train), 
                         axis=1)
    ldaTrain = np.append(ldaTrainItermediate, cv.transform(X_train).toarray(), axis=1)
    ldaTestItermediate = np.append(conLda.transform(test), 
                        libLda.transform(test), 
                         axis=1)
    ldaTest = np.append(ldaTestItermediate,
                        cv.transform(X_test).toarray(), 
                        axis=1)
    clf = LogisticRegression(class_weight="balanced",
                             C=1.0)
    clf.fit(ldaTrain, y_train)
    y_pred = clf.decision_function(ldaTest)
    return y_pred

def splitLda(X_train, y_train, X_test, y_test, **kwargs):
    cv = None
    if kwargs["vectorizer"] == "tfidf":
        cv = TfidfVectorizer(ngram_range=kwargs["ngram_range"],
                             stop_words=kwargs["stop_words"],
                             max_features=kwargs["max_features"])
    else:
        cv = CountVectorizer(ngram_range=kwargs["ngram_range"],
                             stop_words=kwargs["stop_words"],
                             max_features=kwargs["max_features"])
    cv.fit(X_train)
    train = cv.transform(X_train)
    test = cv.transform(X_test)
    lib_indices = np.where(y_train == 1)
    con_indices = np.where(y_train == -1)
    conLda = LatentDirichletAllocation(n_topics=kwargs["n_topics"], 
                                       learning_method="batch")
    libLda = LatentDirichletAllocation(n_topics=kwargs["n_topics"], 
                                       learning_method="batch")
    conLda.fit(train[con_indices])
    libLda.fit(train[lib_indices])
    ldaTrain = np.append(conLda.transform(train), libLda.transform(train), axis=1)
    ldaTest = np.append(conLda.transform(test), libLda.transform(test), axis=1)
    clf = LogisticRegression(class_weight="balanced",
                             C=kwargs["C"])
    clf.fit(ldaTrain, y_train)
    y_pred = clf.decision_function(ldaTest)
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1
    #print accuracy_score(y_test, y_label)
    return y_pred


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic" +str(topic_idx)
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])