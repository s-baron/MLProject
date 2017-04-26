import itertools
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

vectorizer = [("vectorizer", x) for x in ["tfidf"]]
ngram_range = [("ngram_range", (1,x)) for x in range(2,3)]
stop_wordsList = [("stop_words",x) for x in ["english"]]
max_features = [("max_features", x) for x in [5000]]
max_depth = [("max_depth", d) for d in [2, 5, 10, 15, 20, None]]
max_tree_features = [("max_tree_features", f) for f in [None]]
allParams = [vectorizer, ngram_range, stop_wordsList, max_features, max_tree_features, max_depth]

kwargsList = [{key:value for key, value in kwargs} for kwargs in itertools.product(*allParams)]
nameList = [str([(key, value) for key, value in kwargs.iteritems()]) for kwargs in kwargsList]

def paramDecisionTree(X_train, y_train, X_test, y_test, **kwargs):
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
    train_dtm = cv.transform(X_train)
    clf = DecisionTreeClassifier(criterion="entropy",
                                 class_weight="balanced",
                                 max_depth=kwargs["max_depth"],
                                 max_features=kwargs["max_tree_features"])
    clf.fit(train_dtm, y_train)

    test_dtm = cv.transform(X_test)
    y_pred = clf.predict(test_dtm)
    return y_pred

def simpleDecisionTree(X_train, y_train, X_test, y_test):
    cv = TfidfVectorizer(ngram_range=(1,3), stop_words="english")
    cv.fit(X_train)
    train_dtm = cv.transform(X_train)
    clf = DecisionTreeClassifier(criterion="entropy",
                                 class_weight="balanced",
                                 max_depth=None,
                                 max_features=None)
    clf.fit(train_dtm, y_train)

    test_dtm = cv.transform(X_test)
    print "SIMPLE TREE"
    feat = clf.feature_importances_
    maxIndices = np.argsort(feat)[-50:]
    print np.array(cv.get_feature_names())[maxIndices]
    y_pred = clf.predict(test_dtm)
    return y_pred

def biasedNGramsLogRegModel(X_train, y_train, X_test, y_test):
    # ---------------------------- Make Vocab ------------------------------
    lib_indices = np.where(y_train == 1)
    con_indices = np.where(y_train == -1)
    X_lib = X_train[lib_indices]
    X_con = X_train[con_indices]

    dtms = []
    vocabularies = []
    for X in [X_lib, X_con]:
        cv = CountVectorizer(ngram_range=(1, 3), stop_words="english")
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
    print len(vocab)

    # ---------------------------- Training Set ------------------------------
    # Get words and their frequencies
    cv = CountVectorizer(ngram_range=(1, 3), vocabulary=vocab)
    cv.fit(X_train)
    train_dtm = cv.transform(X_train)
    clf = DecisionTreeClassifier(criterion="entropy",
                                 class_weight="balanced",
                                 max_depth=None,
                                 max_features=None)
    clf.fit(train_dtm, y_train)
    print "COMPLEX TREE"
    feat = clf.feature_importances_
    maxIndices = np.argsort(feat)[-50:]
    print np.array(cv.get_feature_names())[maxIndices]

    # ---------------------------- Testing Set ------------------------------
    test_dtm = cv.transform(X_test)
    y_pred = clf.predict(test_dtm)
    return y_pred