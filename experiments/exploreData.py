import cPickle
import matplotlib.pyplot as plt
import numpy as np
from string import punctuation
from collections import Counter
from matplotlib.ticker import FormatStrFormatter

import os, os.path
import errno
import os
import sys

# Taken from http://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    pathList = path.rsplit("/", 1)
    try:
        os.makedirs(pathList[0])
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(pathList[0]):
            pass
        else: raise

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

def findFileOnPath(name):
    pathToData = None
    for path in sys.path:
        pathToData = find(name, path)
        if pathToData != None:
            return pathToData
    print "Error: file " + name + " not found on path" 
    sys.exit(0)

def sampleCode():
    [lib, con, neutral] = cPickle.load(open(findFileOnPath('ibcData.pkl'), 'rb'))

    # how to access sentence text
    print 'Liberal examples (out of ', len(lib), ' sentences): '
    for tree in lib[0:5]:
        print tree.get_words()

    print '\nConservative examples (out of ', len(con), ' sentences): '
    for tree in con[0:5]:
        print tree.get_words()

    print '\nNeutral examples (out of ', len(neutral), ' sentences): '
    for tree in neutral[0:5]:
        print tree.get_words()

    # how to access phrase labels for a particular tree
    ex_tree = lib[0]

    print '\nPhrase labels for one tree: '

    # see treeUtil.py for the tree class definition
    for node in ex_tree:

        # remember, only certain nodes have labels (see paper for details)
        if hasattr(node, 'label'):
            print node.label, ': ', node.get_words()

# Pulled idea from answer here: http://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
def plotWordCountsFromCounter(counter, title, n, path):
    wordCounts = counter.most_common(n)
    labels = [tup[0] for tup in wordCounts]
    # Construct data in form hist can handle (ints only)
    data = []
    for i in range(n):
        data += [i] * wordCounts[i][1]
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(data, bins=n)

    # Set basic titles
    ax.set_title(title)
    ax.set_ylabel("Count")

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    ax.get_xaxis().set_ticks([])
    for label, x in zip(labels, bin_centers):
        # Label the raw counts
        ax.annotate(label, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -10), textcoords='offset points', va='top', ha='center',rotation=45)

    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    mkdir_p(path)
    fig.savefig(path)
    plt.close('all')

def plotWordCountFrequenciesFromCounter(dict, title, yTitle, xTitle, path):
    wordCounts = dict.values()
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(wordCounts, bins=100)
    ax.set_title(title)
    ax.set_ylabel(yTitle)
    ax.set_xlabel(xTitle)
    mkdir_p(path)
    fig.savefig(path)
    plt.close('all')

def main():
    # Things to plot:
    # Number of sentences of each type (maybe)
    # Number of labels for each type at each level? 
    # Top X words for each, and how many times they appear
    # Counts for how many times words appear x number of times

    [lib, con, neutral] = cPickle.load(open(findFileOnPath('ibcData.pkl'), 'rb'))
    fullDictionary = Counter()
    libDict = Counter()
    conDict = Counter()
    neutralDict = Counter()
    fullLen = Counter()
    libLen = Counter()
    conLen = Counter()
    neutralLen = Counter()
    for tree in lib:
        sentence = tree.get_words()
        wordList = sentence.split()
        for word in wordList:
            libDict[word] += 1
            fullDictionary[word] += 1
            libLen[len(wordList)] += 1
            fullLen[len(wordList)] += 1
    plotWordCountFrequenciesFromCounter(libDict, "Word Frequencies in Liberal Sentence", "Frequency", "Word Count", "vis/libFreq.png")
    plotWordCountsFromCounter(libDict, "Counts of Most Common Words in Liberal Sentences", 25, "vis/libHist.png")
    print "Number of liberal words: " + str(len(libDict.keys()))

    for tree in con:
        sentence = tree.get_words()
        wordList = sentence.split()
        for word in wordList:
            conDict[word] += 1
            fullDictionary[word] += 1
            conLen[len(wordList)] += 1
            fullLen[len(wordList)] += 1
    plotWordCountFrequenciesFromCounter(conDict, "Word Frequencies in Conservative Sentences", "Frequency", "Word Count", "vis/conFreq.png")
    plotWordCountsFromCounter(conDict, "Counts of Most Common Words in Conservative Sentence", 25, "vis/conHist.png")
    print "Number of conservative words: " + str(len(conDict.keys()))

    for tree in neutral:
        sentence = tree.get_words()
        wordList = sentence.split()
        for word in wordList:
            neutralDict[word] += 1
            fullDictionary[word] += 1
            neutralLen[len(wordList)] += 1
            fullLen[len(wordList)] += 1
    plotWordCountFrequenciesFromCounter(neutralDict, "Word Frequencies in Neutral Sentences", "Frequency", "Word Count", "vis/neutralFreq.png")
    plotWordCountsFromCounter(neutralDict, "Counts of Most Common Words in Neutral Sentences", 25, "vis/neutralHist.png")
    print "Number of neutral words: " + str(len(neutralDict.keys()))

    plotWordCountFrequenciesFromCounter(fullDictionary, "Word Frequencies in All Sentences", "Frequency", "Word Count", "vis/allFreq.png")
    plotWordCountsFromCounter(fullDictionary, "Counts of Most Common Words in All Sentences", 25, "vis/allHist.png")
    print "Number of words overall: " + str(len(fullDictionary.keys()))



if __name__ == '__main__':
    main()