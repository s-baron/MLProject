import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc

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
	plt.title(title)
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
	plt.axis("equal")
	plt.title(title)
	leg = plt.legend(bbox_to_anchor=(1,0.815), loc='center left', numpoints=1)
	fig.set_tight_layout(True)
	if path == None:
		plt.show()
	else:
		utilities.mkdir_p(path)
		plt.savefig(path, bbox_inches='tight')
		plt.close("all")
