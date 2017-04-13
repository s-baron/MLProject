import numpy as np
import cPickle
from sklearn.model_selection import train_test_split, StratifiedKFold

import full_ibc.treeUtil as treeUtil
import util.util as utilities
import full_ibc.svcexperiments as svcexperiments

# Only change these variables (and the visualizations section if necessary)
# List of functions that predict two classes
twoClass = [svcexperiments.removeStopWords]
# List of functions that predict three classes
threeClass = []

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

	# ------------------- Add more visualizations below as necessary ------------------

	# --------------------------------------------------------------------------------

	# On final evaluation, uncomment
	# Train on all training data, test on test data
	# y_pred = func(X_train, y_train, X_test)

if __name__ == "__main__":
	main()