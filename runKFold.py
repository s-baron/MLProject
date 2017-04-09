import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# Only change these variables (and the visualizations section if necessary)
func = svm.run_model
# Set to False if classifying neutral, conservative, and liberal
twoClass = True

def main():
	# Load data
	[lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))

	# Format into X and y
	numSentences = len(lib) + len(con) + len(neutral)
	y = np.zeroes((numSentences))
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

	# Separate Test Set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

	y_predAllFolds = []
	kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	for train, test in kFold.split():
		X_train_k = X_train[train]
		y_train_k = y_train[train]
		X_test_k = X_train[test]
		y_test_k = y_train[test]
		# Two class problem
		if twoClass:
			# Remove neutral examples
			neutral_train_indices = np.where(y_train_k == 0)
			neutral_test_indices = np.where(y_test_k == 0)
			X_train_k = X_train_k[~neutral_train_indices]
			y_train_k = y_train_k[~neutral_train_indices]
			X_test_k = X_train_k[~neutral_test_indices]
			y_test_k = y_test_k[~neutral_test_indices]

			# Predict classes
			y_pred_k = func(X_train_k, y_train_k, X_test_k)
		# Three class problem
		else:
			print "Three class not yet implemented"
			exit()

	# Evaluate

	# Visualize

	# ------------------- Add more visualizations below as necessary ------------------

	# --------------------------------------------------------------------------------

	# On final evaluation, uncomment
	# Train on all training data, test on test data
	# y_pred = func(X_train, y_train, X_test)

if __name__ == "__main__":
	main()