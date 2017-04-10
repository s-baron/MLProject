# MLProject
Final project for CSCI158 at HMC

## Usage
Add the location of the repo to your PYTHONPATH.
The main file that should be run for evaluation purposes is runKFold.py,
although additional standalone projects in experiments is fine. 

## Repo Organization
- MLProject
-- experiments/
-- visualizations/
-- util/
-- full_ibc/
--- treeUtil.py (implicit, please do not add to github)
--- loadIBC.py (implicit, please do not add to github)
--- treeUtil.py (implicit, please do not add to github)
--- runKFold.py

### runKFold.py
This file is where all experiments should be run from for comparison purposes. 
It uses a fixed random seed to select the test set and k shuffled folds. Only 
the imports section to add an experiment, the line selecting an experiment to run,
and the section for extra visualizations should be changed.

### experiments/
Files with a single function to run an experiment should be stored here.
Inputs to function: X_train, y_train, X_test, kFoldNumber
Outputs: predicted class for X_test

### visualizations/
Store all visualizations here. Please label in a way that makes sense, don't overwrite existing files, and use subfolders for different experiments

### util/
Store utility code here. Examples include graphing functions, functions to parse data, 
and any other code that may be useful in more than one experiment. 