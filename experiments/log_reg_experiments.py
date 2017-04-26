import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import util.util as utilities
import util.visualize as vis

#vectorizer = [("vectorizer", x) for x in ["tfidf", "normal"]]
#ngram_range = [("ngram_range", (1,x)) for x in range(1,4)]
#stop_wordsList = [("stop_words",x) for x in ["english"]]
max_features = [("max_features", x) for x in [None, 5000, 1000, 200, 50]]
#norm = [("norm", x) for x in ["l2", None]]
C = [("C", x) for x in [10.0, 1.0, 0.01, 0.00001, 0.0000001]]
path = [("path", "TBD")]
allParams = [path, max_features, C]

kwargsList = [{key:value for key, value in kwargs} for kwargs in itertools.product(*allParams)]
nameList = ["RegTest" + str([(key, value) for key, value in kwargs.iteritems()]) for kwargs in kwargsList]

# Vectorizer Params: 
# ngram_range=(1, 1), stop_words="english", max_features=None, vocabulary=None
# norm="l2"
def paramLogReg(X_train, y_train, X_test, y_test, **kwargs):
    cv = None
    if kwargs["vectorizer"] == "tfidf":
        cv = TfidfVectorizer(ngram_range=kwargs["ngram_range"],
                             stop_words=kwargs["stop_words"],
                             max_features=kwargs["max_features"],
                             norm=kwargs["norm"])
    else:
        cv = CountVectorizer(ngram_range=kwargs["ngram_range"],
                             stop_words=kwargs["stop_words"],
                             max_features=kwargs["max_features"])
    cv.fit(X_train)
    print len(cv.vocabulary_)
    train_dtm = cv.transform(X_train)
    clf = LogisticRegression(class_weight="balanced",
                             C=kwargs["C"])
    clf.fit(train_dtm, y_train)

    test_dtm = cv.transform(X_test)
    y_pred = clf.decision_function(test_dtm)
    return y_pred
def getBiasedWords(X_train, y_train):
    con_indices = np.where(y_train == -1)
    lib_indices = np.where(y_train == 1)
    X_lib = X_train[lib_indices]
    X_con = X_train[con_indices]

    dtms = []
    vocabularies = []
    for X in [X_lib, X_con]:
        cv = TfidfVectorizer(ngram_range=(1,2),
                             stop_words="english",
                             max_features=None)
        dtm = cv.fit_transform(X)
        #print dtm.shape
        dtms.append(dtm)
        vocabularies.append(cv.vocabulary_)
    vocabSets = [set(vocab.keys()) for vocab in vocabularies]
    dtms[0] = dtms[0].toarray()
    dtms[1] = dtms[1].toarray()
    wordsInBoth = vocabSets[0].intersection(vocabSets[1])
    #print len(wordsInBoth)
    signficance = {}
    for word in wordsInBoth:
        a = dtms[0][:, vocabularies[0][word]]
        b = dtms[1][:, vocabularies[1][word]]
        t, prob = stats.ttest_ind(a, b)
        signficance[word] = prob
    return signficance

def printDTMWords(dtmEntry, features):
    toPrint = ""
    rows,cols = dtmEntry.nonzero()
    for row,col in zip(rows,cols):
        toPrint += features[col] + ", " 
    return toPrint[:-2]

def bestLogReg(X_train, y_train, X_test, y_test, **kwargs):
    cv = TfidfVectorizer(ngram_range=(1,2),
                         stop_words="english",
                         max_features=kwargs["max_features"])
    clf = LogisticRegression(class_weight="balanced",
                             C=kwargs["C"])
    cv.fit(X_train)
    train_dtm = cv.transform(X_train)
    test_dtm = cv.transform(X_test)
    clf.fit(train_dtm, y_train)
    y_pred = clf.decision_function(test_dtm)
    y_label = clf.predict(test_dtm)
    # Show what we got wrong
    y_prob = clf.predict_proba(test_dtm)
    utilities.mkdir_p(kwargs["path"])
    with open(kwargs["path"]+"SentenceResults.txt", "w+") as f:
        stringToWrite = ""
        for i in range(20):
            stringToWrite += str(i) + ":\n\t"
            if y_label[i] == y_test[i]:
                stringToWrite += "Predicted RIGHT: \n\t"
            else:
                stringToWrite += "Predicted WRONG: \n\t"
            stringToWrite += "Real label: {}, predicted label: {}\n\t".format(y_test[i], y_label[i])
            stringToWrite += X_test[i] + "\n\t"
            stringToWrite += printDTMWords(test_dtm[i], cv.get_feature_names()) + "\n"
        f.write(stringToWrite)
    vis.plotConfidence(y_test, y_prob[:,0], "Test "+str(clf.classes_[0]), path=kwargs["path"]+"ConfidenceVis.png")
    vis.plotLearningCurve(cv, clf, X_train, y_train, str(kwargs), path=kwargs["path"]+"LearningCurve.png")
    vis.plot_coefficients(clf, cv.get_feature_names(), top_features=20, path=kwargs["path"]+"Coefficients.png")

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
    print len(vocab)

    # ---------------------------- Training Set ------------------------------
    # Get words and their frequencies
    cv = CountVectorizer(ngram_range=(1, 3), vocabulary=vocab)
    cv.fit(X_train)
    train_dtm = cv.transform(X_train)
    clf = LogisticRegression(class_weight="balanced", 
                             C=0.01)
    clf.fit(train_dtm, y_train)

    # ---------------------------- Testing Set ------------------------------
    test_dtm = cv.transform(X_test)
    y_pred = clf.decision_function(test_dtm)
    vis.plotConfidence(y_test, y_pred, "Test", path=None)
    return y_pred

def simpleLogRegModel(X_train, y_train, X_test, y_test):
    # ---------------------------- Training Set ------------------------------
    # Get words and their frequencies
    cv = CountVectorizer()
    cv.fit(X_train)
    train_dtm = cv.transform(X_train)
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(train_dtm, y_train)

    # ---------------------------- Testing Set ------------------------------
    test_dtm = cv.transform(X_test)
    y_pred = clf.decision_function(test_dtm)
    return y_pred

def simpleTfidfLogRegModel(X_train, y_train, X_test, y_test):
    # ---------------------------- Training Set ------------------------------
    # Get words and their frequencies
    cv = TfidfVectorizer()
    cv.fit(X_train)
    train_dtm = cv.transform(X_train)
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(train_dtm, y_train)

    # ---------------------------- Testing Set ------------------------------
    test_dtm = cv.transform(X_test)
    y_pred = clf.decision_function(test_dtm)
    return y_pred