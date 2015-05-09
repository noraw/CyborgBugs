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

# ------------- INPUT VARIABLES-------------------------------------
filesList = [
    ["04_Lab_FD_031114", 13571],
    ["12_Lab_C_060514", 40504],
    ["13_Lab_Cmac_031114", 6032],
    ["17_Lab_Cmac_031214", 4988],
    ["21_Lab_Corrizo_051614", 86619],
    ["29_Lab_Corrizo_051914", 73811],
    ["31_Lab_Troyer_052114", 14482],
    ["35_Lab_Val_100714", 47538]
]

filesListDebug = [
    ["04_Lab_FD_031114", 1358],
    ["12_Lab_C_060514", 4051],
    ["13_Lab_Cmac_031114", 604],
    ["17_Lab_Cmac_031214", 499],
    ["21_Lab_Corrizo_051614", 8662],
    ["29_Lab_Corrizo_051914", 7382],
    ["31_Lab_Troyer_052114", 1449],
    ["35_Lab_Val_100714", 4754]
]

n_neighbors = [7, 12]
alphas = [0.1, 0.5, 0.9]
max_iters = [30, 80]
tols = [0.001, 0.0001]
startIndex = 0

folder = "./input"

# used for debugging
#filesList = filesListDebug
#folder = "./input/debug"


#---------------------READ/WRITE FUNCTIONS----------------------------------------

# read a dat file back into python. 
def readFileMatrix(myfile, size):
    array = np.fromfile(myfile, dtype=np.float64, count=-1, sep="")
    array = np.reshape(array,(size,-1))
    return array

def writeFileMatrix(matrix, fileName):
    matrix.tofile(fileName)

def writeFileArray(dictionary, fileName):
    # output feature importance for graphs
    outfile = file(fileName, "w")
    keys = []
    header = ""
    for key in dictionary[0].keys():
        keys.append(key)
        header += '"'+str(key) + '",'
    outfile.write(header + '\n');

    for i in range(len(dictionary)):
        outLine = ""
        for key in keys:
            outLine += str(dictionary[i][key]) + ","
        outLine += "\n"
        outfile.write(outLine)

    outfile.close();

def writeFileList(array, fileName):
    # output feature importance for graphs
    outfile = file(fileName, "w")
    outfile.write('"ID","Value"\n');

    for i in range(len(array)):
        outLine = '"%i","%s"\n' % (i, str(array[i]))
        outfile.write(outLine)

    outfile.close();

def writeFileMatrixCSV(matrix, fileName):
    outfile = file(fileName, "w")
    outfile.write('"Confusion Matrix","1","2","3","4","5","6"\n');
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    for row in range(rows):
        outline = "%i," % (row+1)
        for col in range(cols):
            outline += "%i," % matrix[row][col]
        outline += "\n"
        outfile.write(outline)

    outfile.close()


# -------------------------------HELPER FUNCTIONS ------------------------------
def getSizeFromFileName(myfile):
    parts = myfile.split("_")
    parts2 = parts[-1].split(".")
    return parts2[0]


def createTrainingData(folder, fileList):
    for i in range(len(fileList)):
        fileInfo = fileList[i]
        inX = "%s/%s_selectedFeatures_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inY = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
#        inX = "%s/%s_selectedFeatures_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
#        inY = "%s/%s_labels_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        X  = readFileMatrix(inX, fileInfo[1])
        Y  = readFileMatrix(inY, fileInfo[1])

        if(i == 0):
            XFinal = X
            YFinal = Y
        else:
            XFinal = np.concatenate((XFinal, X), axis=0)
            YFinal = np.concatenate((YFinal, Y), axis=0)

    return [XFinal, YFinal]


def convertToDictLabels(values, classes):
    dictItem = {}
    for i in range(len(values)):
        dictItem["Label %i" % classes[i]] = values[i]
    return dictItem


def predict(clf, X, y, X_test, y_test, outname):
    results = []
    time0 = timeit.default_timer()
    results.append("\nResults:\n")

    clf.fit(X, y)
    time1 = timeit.default_timer()
    results.append("   fit done (%i secs)\n" % (time1 - time0))

    y_pred = clf.predict(X_test)
    time2 = timeit.default_timer()
    writeFileList(y_pred, "%s_predictedValues.csv" % (outname))
    results.append("   predict done (%i secs)\n" % (time2 - time1))

    y_predictedProbMatrix = clf.predict_proba(X_test)
    classes = clf.classes_
    time3 = timeit.default_timer()
    dictProb = []
    for i in range(len(y_predictedProbMatrix)):
        dictItem = convertToDictLabels(y_predictedProbMatrix[i], classes)
        dictItem["ID"] = i
        dictProb.append(dictItem)
    writeFileArray(dictProb, "%s_predictedProb.csv" % (outname))
    results.append("   predict probabilities done (%i secs)\n" % (time3 - time2))

    score = clf.score(X_test, y_test)
    time4 = timeit.default_timer()
    results.append("   score done (%i secs): %f\n" % (time4 - time3, score))

    dictScores = []

    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    dictItem = convertToDictLabels(f1_score, classes)
    dictItem["Score Type"] = "f1_score"
    dictScores.append(dictItem)
    time5 = timeit.default_timer()
    results.append("   f1_score done (%i secs): %s\n" % (time5 - time4, str(f1_score)))

    precision_score = metrics.precision_score(y_test, y_pred, average=None)
    dictItem = convertToDictLabels(precision_score, classes)
    dictItem["Score Type"] = "precision_score"
    dictScores.append(dictItem)
    time6 = timeit.default_timer()
    results.append("   precision_score done (%i secs): %s\n" % (time6 - time5, str(precision_score)))

    recall_score = metrics.recall_score(y_test, y_pred, average=None)
    dictItem = convertToDictLabels(recall_score, classes)
    dictItem["Score Type"] = "recall_score"
    dictScores.append(dictItem)
    time7 = timeit.default_timer()
    writeFileArray(dictScores, "%s_scores.csv" % (outname))
    results.append("   recall_score done (%i secs): %s\n" % (time7 - time6, str(recall_score)))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6])
    writeFileMatrixCSV(confusion_matrix, "%s_confusionMatrix.csv" % (outname))

    return results





def calculateGridSpot(index, args, neighbor, alpha, max_iter, tol):
    # want to try it out with each file selected separately as the test data
    # all the other files are combined to be the training data
    startAll = timeit.default_timer()
    if args.LabelSpreading:
        print "\nLabelSpreading(%i): %i, %f, %i, %f" % (index, neighbor, alpha, max_iter, tol)

    for fileInfo in filesList:
        outLines = []
        outname = "./output/PredictedLabelSpreading/" # assigned later
        inXtest = "%s/%s_selectedFeatures_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
#        inXtest = "%s/%s_selectedFeatures_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
#        inYtest = "%s/%s_labels_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
#        outname += "test_"
        Xtest  = readFileMatrix(inXtest, fileInfo[1])
        Ytest  = readFileMatrix(inYtest, fileInfo[1])
        YOnes = -np.ones_like(Ytest)

        trainList = list(filesList)
        trainList.remove(fileInfo)
        [Xtrain, Ytrain] = createTrainingData(folder, trainList)
        Xtrain = np.concatenate((Xtrain, Xtest), axis=0)
        Ytrain = np.concatenate((Ytrain, YOnes), axis=0)

#        print "X train shape %s" % str(Xtrain.shape)
#        print "X test shape %s" % str(Xtest.shape)
#        print "Y train shape %s" % str(Ytrain.shape)
#        print "Y test shape %s" % str(Ytest.shape)

        outLines.append("Test File:\n")
        outLines.append("inXtest: %s\n" % inXtest)
        outLines.append("inYtest: %s\n" % inYtest)

        outLines.append("\nVariables:\n")
        outLines.append("n_neighbor: %i\n" % neighbor)
        outLines.append("alpha: %f\n" % alpha)
        outLines.append("max_iter: %i\n" % max_iter)
        outLines.append("tol: %f\n" % tol)

        outLines.append("\nRead in Files Done\n")
        outLines.append("X train shape %s\n" % str(Xtrain.shape))
        outLines.append("X test shape %s\n" % str(Xtest.shape))
        outLines.append("Y train shape %s\n" % str(Ytrain.shape))
        outLines.append("Y test shape %s\n" % str(Ytest.shape))
        #outLines.append("total: %i\n" % (Xtrain.shape[0] + Xtest.shape[0]))
        outLines.append("\n")



        # CLASSIFY!
        if args.LabelSpreading:
            outname += "LabelSpreading"
            clf = LabelSpreading(kernel='knn', n_neighbors=neighbor, alpha=alpha, max_iter=max_iter, tol=tol)

        if args.Factorization:
            alphaIn=5.0
            print "Lasso: " + str(alphaIn)
            outname += "lasso"+str(alphaIn)
            clf = linear_model.Lasso(alpha=alphaIn)

        if args.LabelSpreading or args.Factorization:
            start = timeit.default_timer()
            results = predict(clf, Xtrain, np.ravel(Ytrain), Xtest, np.ravel(Ytest), "%s_%s_%i" % (outname, fileInfo[0], index))
            stop = timeit.default_timer()
            results.append("   total time: %i secs\n" % (stop - start))

            outfile = file("%s_%s_%i_results.txt" % (outname, fileInfo[0], index), "w")

            for i in range (len(outLines)):
                outfile.write(outLines[i])

            for i in range (len(results)):
                outfile.write(results[i])

            outfile.close();

    stopAll = timeit.default_timer()
    print "Done predicting: %i secs" % (stopAll - startAll)


#---------------------MAIN FUNCTION------------------------------------------------------------------------

# argument parsing.
parser = argparse.ArgumentParser(description='Predict CyborgBugs.')
parser.add_argument("-L", "--LabelSpreading", action="store_true", help="run LabelSpreading")
parser.add_argument("-F", "--Factorization", action="store_true", help="run non-negative factorization model")


args = parser.parse_args()
print args;

index = 1
for neighbor in n_neighbors:
    for alpha in alphas:
        for max_iter in max_iters:
            for tol in tols:
                if(index >= startIndex):
                    calculateGridSpot(index, args, neighbor, alpha, max_iter, tol)
                index +=1


print "------------------------ALL DONE!!!!---------------------------------"






