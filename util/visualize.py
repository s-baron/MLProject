import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score
from textwrap import wrap

import util as utilities

# Found at http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', path=None):
	""" Inputs: arrays of true and (continous) predicted y values for 1 fold
		Outputs: A confusion matrix saved at the location given in path relative
		to the working directory
	"""
	plt.gcf().clear()
	fig = plt.figure()
	np.set_printoptions(precision=2)
	cmap = plt.cm.cool
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title('\n'.join(wrap(title,60)))
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	fig.set_tight_layout(True)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	if path == None:
		plt.show()
	else:
		utilities.mkdir_p(path)
		plt.savefig(path, format='png')
		plt.close('all')

def plotConfidence(y_true, y_pred, title, path=None):
	plt.gcf().clear()
	classes = np.unique(y_true)
	indices = []
	for c in classes:
		indices.append(np.where(y_true == c))
	arr = [y_pred[i] for i in indices]
	minProb = min(y_pred)
	maxProb = max(y_pred)
	binMax = max(0.5-minProb, maxProb-0.5)
	plt.hist(arr, bins=50, range=(0.5-binMax, 0.5+binMax), label=[str(classes[0]), str(classes[1])])
	plt.legend(loc='upper right')
	plt.title('\n'.join(wrap(title,60)))
	if path == None:
		plt.show()
	else:
		utilities.mkdir_p(path)
		plt.savefig(path, bbox_inches='tight')
		plt.close("all")

def plotLearningCurve(cv, clf, X, y, title, path=None):
	""" Inputs: countvectorizer and classifier objects, 
				X which is a list of sentences, and y
				which is a numpy array of labels
	"""
	# Learning Curve
	n = len(X)
	perf = []
	perfTrain = []
	percentage = []
	for i in xrange(10, 81, 5):
		percentage.append(i)
		splitIndex = int((i/100.0)*n)
		# Fit things and create DTMs
		cv.fit(X[:splitIndex])
		train_dtm = cv.transform(X[:splitIndex])
		test_dtm = cv.transform(X[splitIndex:])
		clf.fit(train_dtm, y[:splitIndex])

		# Predict Train Set
		y_true = y[:splitIndex]
		y_pred = clf.predict(train_dtm)
		perfTrain.append(1-accuracy_score(y_true, y_pred))

		# Predict Test Set
		y_true = y[splitIndex:]
		y_pred = clf.predict(test_dtm)
		perf.append(1-accuracy_score(y_true, y_pred))
	plt.gcf().clear()
	plt.plot(np.asarray(percentage), np.asarray(perf), 
		c='b', label='Test Error')
	plt.plot(np.asarray(percentage), np.asarray(perfTrain), 
		c='g', label='Train Error')
	plt.autoscale(enable=True)
	plt.xlabel('percentage data')
	plt.ylabel('error')
	plt.legend(loc=1,prop={'size':8})
	if path == None:
		plt.show()
	else:
		utilities.mkdir_p(path)
		plt.savefig(path, bbox_inches='tight')
		plt.close("all")	

# Taken from https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
def plot_coefficients(classifier, feature_names, top_features=20, path=None):
	coef = classifier.coef_.ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
	# create plot
	plt.figure(figsize=(15, 5))
	colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
	if path == None:
		plt.show()
	else:
		utilities.mkdir_p(path)
		plt.savefig(path, bbox_inches='tight')
		plt.close("all")

def plotROC(y_true, y_pred, title, path=None):
	""" Inputs: Arrays containing k arrays of true and continous predicted y values for 1 fold
		Outputs: An ROC plot saved at the location given in path relative the the wd
	"""
	plt.gcf().clear()
	fig = plt.figure()
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	colors = itertools.cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'pink', 'purple', 'chocolate'])
	lw = 2
	k=0
	for true, scores, color in zip(y_true, y_pred, colors):
		fpr, tpr, thresholds = roc_curve(true, scores)
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (k, roc_auc))
		k += 1
	
	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
		 label='Luck')

	mean_tpr /= len(y_true)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
			 label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	#plt.axis("equal")
	plt.title('\n'.join(wrap(title,60)))
	leg = plt.legend(bbox_to_anchor=(1,0.815), loc='center left', numpoints=1)
	fig.set_tight_layout(True)
	if path == None:
		plt.show()
	else:
		utilities.mkdir_p(path)
		plt.savefig(path, bbox_inches='tight')
		plt.close("all")
