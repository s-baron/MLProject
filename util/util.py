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