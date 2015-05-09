# ********************************************
# Author: Nora Coler
# Date: April 20, 2015
#
#
# ********************************************
import numpy as np
import argparse
import os
import random
from sklearn import metrics
from sklearn.semi_supervised import LabelSpreading
from sklearn.cluster import AgglomerativeClustering
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
    ["04_Lab_FD_031114", 1358, 3],
    ["12_Lab_C_060514", 4051, 6],
    ["13_Lab_Cmac_031114", 604, 5],
    ["17_Lab_Cmac_031214", 499, 5],
    ["21_Lab_Corrizo_051614", 8662, 6],
    ["29_Lab_Corrizo_051914", 7382, 6],
    ["31_Lab_Troyer_052114", 1449, 6],
    ["35_Lab_Val_100714", 4754, 6]
]

n_neighbors = [7, 12]
alphas = [0.1, 0.5, 0.9]
max_iters = [30, 80]
tols = [0.001, 0.0001]
startIndex = 0


n_inits = [10, 20]
folder = "./input"

# used for debugging
filesList = filesListDebug
folder = "./input/debug"


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
        inX = "%s/%s_selectedFeatures_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inY = "%s/%s_labels_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])

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

def convertClusterLabels(labels, cluster_labels):
    newLabels = []
    for i in range(len(labels)):
        newLabels.append(cluster_labels[labels[i]])
    return newLabels

def findClusterLabels(X, y, labels):
    countOfLabels = {0: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    1: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    2: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    3: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    4: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    5: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}
    percentOfLabels = {0: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    1: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    2: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    3: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    4: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0},
                    5: {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}}
    actualLabelCounts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

    print labels[1195]
    print labels[1054]
    print labels[248]

    for i in range(len(labels)):
        count = countOfLabels[labels[i]][y[i]]
        countOfLabels[labels[i]][y[i]] = count + 1
        actualLabelCounts[y[i]] = actualLabelCounts[y[i]] + 1

    for i in range(6):
        print "%i: %s" % (i, str(countOfLabels[i]))
    print actualLabelCounts

    for i in range(6):
        total = 0
        for j in range(1,7):
            total += countOfLabels[i][j]
        for j in range(1,7):
            if(total !=0):
                percent = float(countOfLabels[i][j])/float(total)
                percentOfLabels[i][j] = "%.2f" % (percent)
    

    return []

def getKMeansClusterInitArray(X, y, clusterNum):
    labelIndices = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    for i in range(len(y)):
        labelIndices[y[i]].append(i)

    indices = []
    for i in range(1, 7):
        if(len(labelIndices[i]) > 0):
            indices.append(random.sample(labelIndices[i], 1)[0])
    print indices

    clusterCenters = np.ndarray(shape=(len(indices), X.shape[1]), dtype=float, order='F')
    for i in range(len(indices)):
        clusterCenters.put(i, X[indices[i]])
    return clusterCenters


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

def predictKmeans(clf, X, y, X_test, y_test, outname):
    results = []
    time0 = timeit.default_timer()
    results.append("\nResults:\n")

    clf.fit_predict(X)
    time1 = timeit.default_timer()
    results.append("   fit done (%i secs)\n" % (time1 - time0))

    cluster_labels = findClusterLabels(X, y, clf.labels_)
#    y_pred = convertClusterLabels(clf.predict(X_test), cluster_labels)
    return results


def calculateGridSpotLabelSpreading(index, args, neighbor, alpha, max_iter, tol):
    # want to try it out with each file selected separately as the test data
    # all the other files are combined to be the training data
    startAll = timeit.default_timer()
    print "\nLabelSpreading(%i): %i, %f, %i, %f" % (index, neighbor, alpha, max_iter, tol)

    for fileInfo in filesList:
        outLines = []
        outname = "./output/PredictedLabelSpreading/" # assigned later
        inXtest = "%s/%s_selectedFeatures_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
        inXtest = "%s/%s_selectedFeatures_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        outname += "test_"

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
        outname += "LabelSpreading"
        clf = LabelSpreading(kernel='knn', n_neighbors=neighbor, alpha=alpha, max_iter=max_iter, tol=tol)

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


def calculateGridSpotKMeans(index, args, n_init, max_iter, tol):
    # want to try it out with each file selected separately as the test data
    # all the other files are combined to be the training data
    startAll = timeit.default_timer()
    print "\nKMeans(%i): %i, %i, %f" % (index, n_init, max_iter, tol)

    for fileInfo in filesList:
        outLines = []
        outname = "./output/PredictedKMeans/" # assigned later
        inXtest = "%s/%s_selectedFeatures_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
        inXtest = "%s/%s_selectedFeatures_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_test_%i.dat" % (folder, fileInfo[0], fileInfo[1])
        outname += "test_"

        Xtest  = readFileMatrix(inXtest, fileInfo[1])
        Ytest  = readFileMatrix(inYtest, fileInfo[1])

        trainList = list(filesList)
        trainList.remove(fileInfo)
        [Xtrain, Ytrain] = createTrainingData(folder, trainList)

        print "X train shape %s" % str(Xtrain.shape)
        print "X test shape %s" % str(Xtest.shape)
        print "Y train shape %s" % str(Ytrain.shape)
        print "Y test shape %s" % str(Ytest.shape)

        outLines.append("Test File:\n")
        outLines.append("inXtest: %s\n" % inXtest)
        outLines.append("inYtest: %s\n" % inYtest)

        outLines.append("\nVariables:\n")
        outLines.append("n_init: %i\n" % n_init)
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
        outname += "KMeans"
        init = getKMeansClusterInitArray(Xtest, np.ravel(Ytest), fileInfo[2])
        clf = AgglomerativeClustering(n_clusters=fileInfo[2], affinity='euclidean', connectivity=None, n_components=None, compute_full_tree='auto', linkage='ward')
        #clf = KMeans(n_clusters=fileInfo[2], init=init, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=24, copy_x=True, n_jobs=1)

        start = timeit.default_timer()
        results = predictKmeans(clf, Xtest, np.ravel(Ytest), Xtest, np.ravel(Ytest), "%s_%s_%i" % (outname, fileInfo[0], index))
        stop = timeit.default_timer()
        results.append("   total time: %i secs\n" % (stop - start))

        outfile = file("%s_%s_%i_results.txt" % (outname, fileInfo[0], index), "w")

        for i in range (len(outLines)):
            outfile.write(outLines[i])

        for i in range (len(results)):
            outfile.write(results[i])

        outfile.close();
        break

    stopAll = timeit.default_timer()
    print "Done predicting: %i secs" % (stopAll - startAll)

#---------------------MAIN FUNCTION------------------------------------------------------------------------

# argument parsing.
parser = argparse.ArgumentParser(description='Predict CyborgBugs.')
parser.add_argument("-L", "--LabelSpreading", action="store_true", help="run LabelSpreading")
parser.add_argument("-K", "--KMeans", action="store_true", help="run KMeans")

random.seed(24)
args = parser.parse_args()
print args;

index = 1
if args.LabelSpreading:
    for neighbor in n_neighbors:
        for alpha in alphas:
            for max_iter in max_iters:
                for tol in tols:
                    if(index >= startIndex):
                        calculateGridSpotLabelSpreading(index, args, neighbor, alpha, max_iter, tol)
                    index +=1

if args.KMeans:
    for n_init in n_inits:
        for max_iter in max_iters:
            for tol in tols:
                if(index >= startIndex):
                    calculateGridSpotKMeans(index, args, n_init, max_iter, tol)
                index +=1
                break
            break
        break


print "------------------------ALL DONE!!!!---------------------------------"






