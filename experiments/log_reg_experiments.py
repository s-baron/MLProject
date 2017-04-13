import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = [("vectorizer", x) for x in ["tfidf", "normal"]]
ngram_range = [("ngram_range", (1,x)) for x in range(1,4)]
stop_wordsList = [("stop_words",x) for x in ["english"]]
max_features = [("max_features", x) for x in [None, 500, 1000, 5000]]
norm = [("norm", x) for x in ["l2", None]]
C = [("C", x) for x in [0.0001, 0.001, 0.01, 0.1, 1.0]]
allParams = [vectorizer, ngram_range, stop_wordsList, max_features, norm, C]

kwargsList = [{key:value for key, value in kwargs} for kwargs in itertools.product(*allParams)]
nameList = [str([(key, value) for key, value in kwargs.iteritems()]) for kwargs in kwargsList]

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