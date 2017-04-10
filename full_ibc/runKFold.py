import numpy as np
import cPickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix

import full_ibc.treeUtil as treeUtil
import util.util as utilities
import util.visualize as vis

from experiments.test_experiment import randomModel

# Only change these variables (and the visualizations section if necessary)
# List of functions that predict two classes
twoClass = [randomModel]
twoClassNames = ["Random2Class"]
# List of functions that predict three classes
threeClass = [randomModel]
threeClassNames = ["Random3Class"]

def main():
	# Load data
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

	# Oversample unbalanced data
	neutral_indices = np.where(y_train == 0)
	con_indices = np.where(y_train == -1)
	lib_indices = np.where(y_train == 1)
	lib_len = len(lib_indices[0])
	con_len = len(con_indices[0])
	neutral_len = len(neutral_indices[0])
	goalSize = max(neutral_len, con_len, lib_len)
	np.random.seed(42)
	X_train = np.append(X_train, np.random.choice(X_train[neutral_indices], 
		                                          size=(goalSize - neutral_len),
		                                          replace=True))
	y_train = np.append(y_train, np.repeat(0, goalSize-neutral_len))
	X_train = np.append(X_train, np.random.choice(X_train[con_indices], 
		                                          size=(goalSize - con_len),
		                                          replace=True))
	y_train = np.append(y_train, np.repeat(-1, goalSize - con_len))
	X_train = np.append(X_train, np.random.choice(X_train[lib_indices], 
		                                          size=(goalSize-lib_len),
		                                          replace=True))
	y_train = np.append(y_train, np.repeat(1, goalSize-lib_len))

	numSplits = 10
	kFold = StratifiedKFold(n_splits=numSplits, shuffle=True, random_state=42)
	y_pred_2_all = []
	y_true_2_all = []
	y_pred_3_all = []
	y_true_3_all = []
	for train, test in kFold.split(X_train, y_train):
		X_train_k_3 = X_train[train]
		y_train_k_3 = y_train[train]
		X_test_k_3 = X_train[test]
		y_test_k_3 = y_train[test]
		# Remove neutral examples
		non_neutral_train_indices = np.where(y_train_k_3 != 0)
		non_neutral_test_indices = np.where(y_test_k_3 != 0)
		X_train_k_2 = X_train_k_3[non_neutral_train_indices]
		y_train_k_2 = y_train_k_3[non_neutral_train_indices]
		X_test_k_2 = X_test_k_3[non_neutral_test_indices]
		y_test_k_2 = y_test_k_3[non_neutral_test_indices]

		y_pred_2_fold = []
		y_pred_3_fold = []
		for func in twoClass:
			y_pred_2_fold.append(func(X_train_k_2, y_train_k_2, X_test_k_2, y_test_k_2))

		for func in threeClass:
			y_pred_3_fold.append(func(X_train_k_3, y_train_k_3, X_test_k_3, y_test_k_3))
		y_pred_2_all.append(y_pred_2_fold)
		y_pred_3_all.append(y_pred_3_fold)
		y_true_2_all.append(y_test_k_2)
		y_true_3_all.append(y_test_k_3)

	# Evaluate

	# Visualize
	# Confusion Matrices for All Folds/Experiments
	for k in xrange(numSplits):
		for m in xrange(len(twoClass)):
			y_true = y_true_2_all[k]
			y_pred = y_pred_2_all[k][m]
			y_label = np.sign(y_pred)
			cm = confusion_matrix(y_true, y_label)
			title = twoClassNames[m]
			path = "../visualizations/{}/cm_fold{}.png".format(title, k)
			vis.plotConfusionMatrix(cm, ["Conservative", "Liberal"], path=path, title=title)
	# ROC Plots for All Experiments
	for m in xrange(len(twoClass)):
		y_true = y_true_2_all
		y_pred = [y_pred_2_all[k][m] for k in xrange(numSplits)]
		title = twoClassNames[m]
		path = "../visualizations/{}/ROCPlot".format(title, k)
		vis.plotROC(y_true, y_pred, title, path=path)

	# ------------------- Add more visualizations below as necessary ------------------

	# --------------------------------------------------------------------------------

	# On final evaluation, uncomment
	# Train on all training data, test on test data
	# y_pred = func(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
	main()