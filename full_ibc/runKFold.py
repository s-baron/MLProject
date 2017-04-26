import numpy as np
import cPickle
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

import full_ibc.treeUtil as treeUtil
import util.util as utilities
import experiments.svcexperiments as svcexperiments
import util.visualize as vis

from experiments.test_experiment import randomModel
import experiments.log_reg_experiments as logReg
#import experiments.randomForests as randomForests
import experiments.decision_tree_experiments as decisionTree
import experiments.ldaExperiments as ldaExperiments
import experiments.neutral_bias as neutralBias

# Only change these variables (and the visualizations section if necessary)
# List of functions that predict two classes
twoClass = [logReg.bestLogReg for x in logReg.kwargsList]
twoClassNames = logReg.nameList
twoClassKwargs = logReg.kwargsList
# twoClass = [svcexperiments.simpleSVM]
# twoClassNames = ["SVM Remove Stop Words"]
# twoClassKwargs = [None]

neutral_bias = False

# List of functions that predict three classes
# threeClass = [randomModel]
# threeClassNames = ["Random3Class"]
threeClass = []
threeClassNames = []

def main():
	# Load data
	print len(twoClass)
	print len(threeClass)
	path = utilities.findFileOnPath('ibcData.pkl')
	[lib, con, neutral] = cPickle.load(open(path, 'rb'))

	# Format into X and y
	numSentences = len(lib) + len(con) + len(neutral)
	y = np.zeros((numSentences))
	X = []
	i = 0
	for tree in lib:
		y[i] = 1
		X.append(tree.get_words())
		i += 1
	for tree in con:
		y[i] = -1
		X.append(tree.get_words())
		i += 1
	for tree in neutral:
		y[i] = 0
		X.append(tree.get_words())
		i += 1
	X = np.array(X, dtype=object)

	# Separate Test Set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

	numSplits = 10
	kFold = StratifiedKFold(n_splits=numSplits, shuffle=True, random_state=42)
	y_pred_2_all = []
	y_true_2_all = []
	y_pred_3_all = []
	y_true_3_all = []
	k = 0
	for train, test in kFold.split(X_train, y_train):
		print "Fold " + str(k) + " out of " + str(numSplits)
		X_train_k_3 = X_train[train]
		y_train_k_3 = y_train[train]
		X_test_k_3 = X_train[test]
		y_test_k_3 = y_train[test]
		# Remove neutral examples
		non_neutral_train_indices = np.where(y_train_k_3 != 0)
		non_neutral_test_indices = np.where(y_test_k_3 != 0)
		neutral_train_indices = np.where(y_train_k_3 == 0)
		neutral_test_indices = np.where(y_test_k_3 == 0)
		# if neutral_bias is true then we have two classes: neutral vs liberal and conservative
		if neutral_bias:
			X_train_k_2 = X_train_k_3
			y_train_k_2 = y_train_k_3
			y_train_k_2[neutral_train_indices] = -1
			y_train_k_2[non_neutral_train_indices] = 1
			X_test_k_2 = X_test_k_3
			y_test_k_2 = y_test_k_3
			y_test_k_2[neutral_test_indices] = -1
			y_test_k_2[non_neutral_test_indices] = 1
		else:
			X_train_k_2 = X_train_k_3[non_neutral_train_indices]
			y_train_k_2 = y_train_k_3[non_neutral_train_indices]
			X_test_k_2 = X_test_k_3[non_neutral_test_indices]
			y_test_k_2 = y_test_k_3[non_neutral_test_indices]

		y_pred_2_fold = []
		y_pred_3_fold = []
		for m in range(len(twoClass)):
			print "Done with " + str(m) + " out of " + str(len(twoClass))
			if twoClassKwargs[m] != None:
				if "path" in twoClassKwargs[m]:
					path = "../visualizations/{}/{}".format(twoClassNames[m], k)
					twoClassKwargs[m]["path"] = path
				y_pred_2_fold.append(twoClass[m](X_train_k_2, y_train_k_2, X_test_k_2, y_test_k_2, **twoClassKwargs[m]))
			else:
				y_pred_2_fold.append(twoClass[m](X_train_k_2, y_train_k_2, X_test_k_2, y_test_k_2))
		for func in threeClass:
			y_pred_3_fold.append(func(X_train_k_3, y_train_k_3, X_test_k_3, y_test_k_3))
		y_pred_2_all.append(y_pred_2_fold)
		y_pred_3_all.append(y_pred_3_fold)
		y_true_2_all.append(y_test_k_2)
		y_true_3_all.append(y_test_k_3)
		k += 1

	# Evaluate and Visualize
	# Confusion Matrices and Accuracy for All Folds/Experiments
	accuracy_all = []
	for k in xrange(numSplits):
		accuracy_temp = []
		for m in xrange(len(twoClass)):
			y_true = y_true_2_all[k]
			y_pred = y_pred_2_all[k][m]
			y_label = np.sign(y_pred)
			y_label[y_label==0] = 1 # map points of hyperplane to +1
			accuracy_temp.append(accuracy_score(y_true, y_label))
			cm = confusion_matrix(y_true, y_label)
			title = twoClassNames[m]
			path = "../visualizations/{}/cm_fold{}.png".format(title, k)
			vis.plotConfusionMatrix(cm, ["Conservative", "Liberal"], path=path, title=title)
		accuracy_all.append(accuracy_temp)
	for m in xrange(len(twoClass)):
		accuracy = [accuracy_all[k][m] for k in xrange(numSplits)]
		minimum = min(accuracy)
		maximum = max(accuracy)
		average = np.average(accuracy)
		path = "../visualizations/accuracy_results.txt"
		utilities.mkdir_p(path)
		with open(path,"a+") as f:
			toWrite = time.ctime() + ": " + twoClassNames[m] + "\n"
			toWrite += "Min Accuracy: " + str(minimum) + "\n"
			toWrite += "Max Accuracy: " + str(maximum) + "\n"
			toWrite += "Ave Accuracy: " + str(average) + "\n"
			f.write(toWrite)

	# ROC Plots for All Experiments
	for m in xrange(len(twoClass)):
		y_true = y_true_2_all
		y_pred = [y_pred_2_all[k][m] for k in xrange(numSplits)]
		title = twoClassNames[m]
		path = "../visualizations/{}/ROCPlot".format(title, k)
		vis.plotROC(y_true, y_pred, title, path=path)

	# 3 class accuracy
	accuracy_all = []
	for k in xrange(numSplits):
		accuracy_temp = []
		for m in xrange(len(threeClass)):
			y_true = y_true_3_all[k]
			y_pred = y_pred_3_all[k][m]
			accuracy_temp.append(accuracy_score(y_true, y_pred))
			cm = confusion_matrix(y_true, y_pred)
			title = threeClassNames[m]
			path = "../visualizations/{}/cm_fold{}.png".format(title, k)
			vis.plotConfusionMatrix(cm, ["Conservative", "Neutral", "Liberal"], path=path, title=title)
		accuracy_all.append(accuracy_temp)
	for m in xrange(len(threeClass)):
		accuracy = [accuracy_all[k][m] for k in xrange(numSplits)]
		minimum = min(accuracy)
		maximum = max(accuracy)
		average = np.average(accuracy)
		print "min", minimum
		print "max", maximum
		print "avg", average


	# ------------------- Add more visualizations below as necessary ------------------

	# --------------------------------------------------------------------------------

	# On final evaluation, uncomment
	# Train on all training data, test on test data
	# y_pred = func(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
	main()