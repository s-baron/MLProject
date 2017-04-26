import nltk
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np


def svmModel(X_train, y_train, X_test, y_test):
	# Get words and their frequencies
	#cv = CountVectorizer()
	cv = TfidfVectorizer()
	cv.fit(X_train)
	train_dtm = cv.transform(X_train)
	clf = SVC(kernel="linear", class_weight='balanced', C=10)
	clf.fit(train_dtm, y_train)

	test_dtm = cv.transform(X_test)
	y_pred = clf.predict(test_dtm)
	print "score: ", metrics.accuracy_score(y_test, y_pred, normalize=True)
	

def allWords(X_train_k_2, y_train_k_2, X_test_k_2, y_test_k_2):
	n = y_train_k_2.shape[0]
	all_words = []
	for sentence in X_train_k_2:
		all_words.extend(word_tokenize(sentence))
	all_words = nltk.FreqDist(all_words)
	selected_words = all_words.most_common(4000)
	selected_words = [x[0] for x in selected_words]
	cv = CountVectorizer(tokenizer= word_tokenize, vocabulary=selected_words)
	cv.fit(X_train_k_2)
	X_train1 = cv.transform(X_train_k_2)
	X_test1 = cv.transform(X_test_k_2)
	print "aw", score(X_train1, y_train_k_2, X_test1, y_test_k_2)


# using nltk stop words
def removeStopWords(X_train_k_2, y_train_k_2, X_test_k_2, y_test_k_2):
	n = y_train_k_2.shape[0]
	all_words = []
	stop_words = set(stopwords.words('english'))
	for sentence in X_train_k_2:
		words = word_tokenize(sentence)
		for word in words:
			if word not in stop_words and word not in all_words:
				all_words.append(word)
	cv = CountVectorizer(tokenizer= word_tokenize, vocabulary=all_words)
	cv.fit(X_train_k_2)
	X_train1 = cv.transform(X_train_k_2)
	X_test1 = cv.transform(X_test_k_2)
	clf = SVC(kernel="linear", class_weight='balanced')
	clf.fit(X_train1, y_train_k_2)
	# plot_coefficients(clf, cv.get_feature_names())
	y_pred = clf.predict(X_test1)
	# print "cv", score(X_train1, y_train_k_2, X_test1, y_test_k_2)
	return y_pred

def simpleSVM(X_train, y_train, X_test, y_test):
	cv = TfidfVectorizer(ngram_range=(1,4),
						stop_words="english", 
						max_features=5000)
	cv.fit(X_train)
	train = cv.transform(X_train)
	test = cv.transform(X_test)
	clf = SVC(kernel="linear", class_weight='balanced')
	clf.fit(train, y_train)
	y_pred = clf.predict(test)
	plot_coefficients(clf, cv.get_feature_names())
	return y_pred

def plot_coefficients(classifier, feature_names, top_features=20):
	coef = classifier.coef_.toarray()
	coef = np.ravel(np.asarray(coef))
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
	# create plot
	colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
	plt.figure(figsize=(15, 5))
	plt.show()

# using nltk stop words and stem words
def stemWords(X_train_k_2, y_train_k_2, X_test_k_2, y_test_k_2):
	n = y_train_k_2.shape[0]
	all_words = []
	ps = PorterStemmer()
	stop_words = set(stopwords.words('english'))
	for sentence in X_train_k_2:
		words = word_tokenize(sentence)
		for word in words:
			if word not in stop_words:
				all_words.append(ps.stem(word))
	cv = CountVectorizer(tokenizer= word_tokenize, vocabulary=set(all_words))
	cv.fit(X_train_k_2)
	X_train1 = cv.transform(X_train_k_2)
	X_test1 = cv.transform(X_test_k_2)
	print "cv", score(X_train1, y_train_k_2, X_test1, y_test_k_2)


def score(X_train, y_train, X_test, y_test):
	clf = SVC(kernel="linear", class_weight='balanced')
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, y_pred, normalize=True)
	return score