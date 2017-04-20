from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC

def simpleLda(X_train, y_train, X_test, y_test):
	n_top_words = 10
	cv = TfidfVectorizer(stop_words="english")
	cv.fit(X_train)
	train = cv.transform(X_train)
	test = cv.transform(X_test)
	clf = LatentDirichletAllocation(n_topics=20, learning_method="batch")
	clf.fit(train)
	ldaTrain = clf.transform(train)
	ldaTest = clf.transform(test)
	svmClf = SVC(kernel="linear", class_weight='balanced')
	svmClf.fit(ldaTrain, y_train)
	y_pred = svmClf.predict(ldaTest)
	# print clf.components_[: , :10]
	print_top_words(clf, cv.get_feature_names(), n_top_words)
	return y_pred


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic" +str(topic_idx)
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])