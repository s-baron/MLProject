import numpy as np

def randomModel(X_train, y_train, X_test, y_test):
	""" Inputs: X, arrays of strings (sentences)
	            y, arrays of labels (-1/1 = conservative/liberal)
	    Outputs: y_pred, predicted labels based on the ratio of labels in train
	"""
	np.random.seed(42)
	counts = []
	classes = np.unique(y_train)
	for c in classes:
		c_indices = np.where(y_train == c)
		counts.append(len(c_indices[0]))
	total = float(sum(counts))
	norm_counts = [c / total for c in counts]
	y_pred = np.random.choice(classes, size=X_test.shape[0], p=norm_counts, replace=True)
	return y_pred

