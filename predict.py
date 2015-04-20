# ********************************************
# Author: Nora Coler
# Date: April 20, 2015
#
#
# ********************************************
import numpy as np
import argparse
import os
from sklearn import metrics
from sklearn.semi_supervised import LabelSpreading
from scipy import sparse
import timeit
from math import sqrt

# read a dat file back into python. 
def readFileMatrix(myfile, size):
    array = np.fromfile(myfile, dtype=np.float64, count=-1, sep="")
    array = np.reshape(array,(size,-1))
    return array

def writeFileMatrix(matrix, fileName):
    matrix.tofile(fileName)

def getSizeFromFileName(myfile):
    parts = myfile.split("_")
    parts2 = parts[-1].split(".")
    return parts2[0]


def predict(clf, X, y, X_test, y_test, outname):
    results = []
    time0 = timeit.default_timer()
    results.append("\nResults:\n")

    clf.fit(X, y)
    time1 = timeit.default_timer()
    results.append("   fit done (%i secs)\n" % (time1 - time0))
    print "   fit done (%i secs)" % (time1 - time0)

    y_pred = clf.predict(X_test)
    time2 = timeit.default_timer()
    results.append("   predict done (%i secs)\n" % (time2 - time1))
    print "   predict done (%i secs)" % (time2 - time1)

    y_predictedProbMatrix = clf.predict_proba(X_test)
    time3 = timeit.default_timer()
    results.append("   predict probabilities done (%i secs)\n" % (time3 - time2))
    print "   predict probabilities done (%i secs)" % (time3 - time2)

    score = clf.score(X_test, y_test)
    time4 = timeit.default_timer()
    results.append("   score done (%i secs): %i\n" % (time4 - time3, score))
    print "   score done (%i secs): %i" % (time4 - time3, score)

    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    time5 = timeit.default_timer()
    results.append("   f1_score done (%i secs): %s\n" % (time5 - time4, str(f1_score)))
    print "   f1_score done (%i secs): %s" % (time5 - time4, str(f1_score))

    precision_score = metrics.precision_score(y_test, y_pred, average=None)
    time6 = timeit.default_timer()
    results.append("   precision_score done (%i secs): %s\n" % (time6 - time5, str(precision_score)))
    print "   precision_score done (%i secs): %s" % (time6 - time5, str(precision_score))

    recall_score = metrics.recall_score(y_test, y_pred, average=None)
    time7 = timeit.default_timer()
    results.append("   recall_score done (%i secs): %s\n" % (time7 - time6, str(recall_score)))
    print "   recall_score done (%i secs): %s" % (time7 - time6, str(recall_score))

    return results


#---------------------MAIN FUNCTION------------------------------------------------------------------------

# argument parsing.
parser = argparse.ArgumentParser(description='Predict CyborgBugs.')
parser.add_argument("-L", "--LabelSpreading", action="store_true", help="run LabelSpreading")
parser.add_argument("-F", "--Factorization", action="store_true", help="run non-negative factorization model")
parser.add_argument("-M", "--Mixature", action="store_true", help="run mixature model")

# OTHER INPUT VARIABLES
outname = "./output/" # assigned later
outLines = []
inXtrain = "./input/04_Lab_FD_031114--12_Lab_C_060514--13_Lab_Cmac_031114--17_Lab_Cmac_031214--21_Lab_Corrizo_051614--29_Lab_Corrizo_051914--31_Lab_Troyer_052114--35_Lab_Val_100714--_selectedFeatures_train_si123ze.dat"
inXtest = "./input/04_Lab_FD_031114--12_Lab_C_060514--13_Lab_Cmac_031114--17_Lab_Cmac_031214--21_Lab_Corrizo_051614--29_Lab_Corrizo_051914--31_Lab_Troyer_052114--35_Lab_Val_100714--_selectedFeatures_test_si123ze.dat"
inYtrain = "./input/04_Lab_FD_031114--12_Lab_C_060514--13_Lab_Cmac_031114--17_Lab_Cmac_031214--21_Lab_Corrizo_051614--29_Lab_Corrizo_051914--31_Lab_Troyer_052114--35_Lab_Val_100714--_labels_train_si123ze.dat"
inYtest = "./input/04_Lab_FD_031114--12_Lab_C_060514--13_Lab_Cmac_031114--17_Lab_Cmac_031214--21_Lab_Corrizo_051614--29_Lab_Corrizo_051914--31_Lab_Troyer_052114--35_Lab_Val_100714--_labels_test_si123ze.dat"

outLines.append("Input Files:\n")
outLines.append("inXtrain: %s\n" % inXtrain)
outLines.append("inXtest: %s\n" % inXtrain)
outLines.append("inYtrain: %s\n" % inXtrain)
outLines.append("inYtest: %s\n" % inXtrain)
outLines.append("\n")

args = parser.parse_args()
print args;

Xtrain = readFileMatrix(inXtrain, getSizeFromFileName(inXtrain))
Xtest  = readFileMatrix(inXtrain, getSizeFromFileName(inXtrain))
Ytrain = readFileMatrix(inXtrain, getSizeFromFileName(inXtrain))
Ytest  = readFileMatrix(inXtrain, getSizeFromFileName(inXtrain))
print "Read in Files Done"

outLines.append("X train shape %s\n" % str(Xtrain.shape))
outLines.append("X test shape %s\n" % str(Xtest.shape))
outLines.append("Y train shape %s\n" % str(Ytrain.shape))
outLines.append("Y test shape %s\n" % str(Ytest.shape))
outLines.append("\n")

print "X train shape %s" % str(Xtrain.shape)
print "X test shape %s" % str(Xtest.shape)
print "Y train shape %s" % str(Ytrain.shape)
print "Y test shape %s" % str(Ytest.shape)



# CLASSIFY!
if args.LabelSpreading:
    print "Label Spreading"
    outname += "LabelSpreading"
    clf = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2, max_iter=30, tol=0.001)

if args.Lasso:
    alphaIn=5.0
    print "Lasso: " + str(alphaIn)
    outname += "lasso"+str(alphaIn)
    clf = linear_model.Lasso(alpha=alphaIn)

if args.RANSAC:
    print "RANSAC"
    outname += "ransac"
    clf = linear_model.RANSACRegressor(linear_model.LinearRegression())


if args.LabelSpreading or args.Lasso or args.RANSAC:
    results = predict(clf, Xtrain, Ytrain, Xtest, Ytest)

    # output feature importance for graphs
    outfile = file(outname + "_results.txt", "w")

    for i in range (len(outLines)):
        outfile.write(outLines[i])

    for i in range (len(results)):
        outfile.write(results[i])

    outfile.close();





