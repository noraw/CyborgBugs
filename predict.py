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
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GMM
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

filesList10 = [
    ["04_Lab_FD_031114", 1358],
    ["12_Lab_C_060514", 4051],
    ["13_Lab_Cmac_031114", 604],
    ["17_Lab_Cmac_031214", 499],
    ["21_Lab_Corrizo_051614", 8662],
    ["29_Lab_Corrizo_051914", 7382],
    ["31_Lab_Troyer_052114", 1449],
    ["35_Lab_Val_100714", 4754]
]

filesList90 = [
    ["04_Lab_FD_031114", 12213],
    ["12_Lab_C_060514", 36453],
    ["13_Lab_Cmac_031114", 5428],
    ["17_Lab_Cmac_031214", 4489],
    ["21_Lab_Corrizo_051614", 77957],
    ["29_Lab_Corrizo_051914", 66429],
    ["31_Lab_Troyer_052114", 13033],
    ["35_Lab_Val_100714", 42784]
]

n_neighbors = [12]
alphas = [0.9]
max_iters = [30]
tols = [0.001]
startIndex = 0


n_inits = [10, 20]

covar_types = ['spherical', 'diag', 'tied', 'full']

infolders = [
    "12ImportantFeatures",
    "12LargestMag",
    "12LargestFreq",
    "12LargestMagFreq"
]

folder = "./input"
folder90 = "split90"
folder10 = "split10"
debug = "debug"

# used for debugging
#filesList = filesListDebug


#---------------------READ/WRITE FUNCTIONS----------------------------------------
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# read a dat file back into python. 
def readFileMatrix(myfile, size):
    array = np.fromfile(myfile, dtype=np.float64, count=-1, sep="")
    array = np.reshape(array,(size,-1))
    return array

def writeFileMatrix(matrix, fileName):
    ensure_dir(fileName)
    matrix.tofile(fileName)

def writeFileArray(dictionary, fileName):
    # output feature importance for graphs
    ensure_dir(fileName)
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
    ensure_dir(fileName)
    # output feature importance for graphs
    outfile = file(fileName, "w")
    outfile.write('"ID","Value"\n');

    for i in range(len(array)):
        outLine = '"%i","%s"\n' % (i, str(array[i]))
        outfile.write(outLine)

    outfile.close();

def writeFileMatrixCSV(matrix, fileName):
    ensure_dir(fileName)
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


def createTrainingData(folderIn, fileList, infolder):
    for i in range(len(fileList)):
        fileInfo = fileList[i]
        inX = "%s/%s/%s_selectedFeatures_%i.dat" % (folder, infolder, fileInfo[0], fileInfo[1])
        inY = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
#        inX = "%s/%s/%s/%s_selectedFeatures_10_%i.dat" % (folder, infolder, debug, fileInfo[0], fileInfo[1])
#        inY = "%s/%s/%s/%s_labels_10_%i.dat" % (folder, infolder, debug, fileInfo[0], fileInfo[1])

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
#    print "\nconvertToDictLabels"
#    print values
#    print classes
    for i in range(len(classes)):
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
    if(outname.find("Gaussian") != -1):
        y_pred = [x+1 for x in y_pred]

    time2 = timeit.default_timer()
    writeFileList(y_pred, "%s_predictedValues.csv" % (outname))
    results.append("   predict done (%i secs)\n" % (time2 - time1))

    y_predictedProbMatrix = clf.predict_proba(X_test)
    classes = [1, 2, 3, 4, 5, 6]
    time3 = timeit.default_timer()
    dictProb = []
    for i in range(len(y_predictedProbMatrix)):
        dictItem = convertToDictLabels(y_predictedProbMatrix[i], classes)
        dictItem["ID"] = i
        dictProb.append(dictItem)
    writeFileArray(dictProb, "%s_predictedProb.csv" % (outname))
    results.append("   predict probabilities done (%i secs)\n" % (time3 - time2))

    accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    time4 = timeit.default_timer()
    results.append("   accuracy_score done (%i secs): %f\n" % (time4 - time3, accuracy))

    dictScores = []

    f1_score = metrics.f1_score(y_test, y_pred, average=None, labels=classes)
    dictItem = convertToDictLabels(f1_score, classes)
    dictItem["Score Type"] = "f1_score"
    dictScores.append(dictItem)
    time5 = timeit.default_timer()
    results.append("   f1_score done (%i secs): %s\n" % (time5 - time4, str(f1_score)))

    precision_score = metrics.precision_score(y_test, y_pred, average=None, labels=classes)
    dictItem = convertToDictLabels(precision_score, classes)
    dictItem["Score Type"] = "precision_score"
    dictScores.append(dictItem)
    time6 = timeit.default_timer()
    results.append("   precision_score done (%i secs): %s\n" % (time6 - time5, str(precision_score)))

    recall_score = metrics.recall_score(y_test, y_pred, average=None, labels=classes)
    dictItem = convertToDictLabels(recall_score, classes)
    dictItem["Score Type"] = "recall_score"
    dictScores.append(dictItem)
    time7 = timeit.default_timer()
    writeFileArray(dictScores, "%s_scores.csv" % (outname))
    results.append("   recall_score done (%i secs): %s\n" % (time7 - time6, str(recall_score)))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=classes)
    writeFileMatrixCSV(confusion_matrix, "%s_confusionMatrix.csv" % (outname))

    return results



def calculateGridSpotLabelSpreading(index, args, neighbor, alpha, max_iter, tol, infolder):
    # want to try it out with each file selected separately as the test data
    # all the other files are combined to be the training data
    startAll = timeit.default_timer()
    print "\nLabelSpreading(%i) - %s: %i, %f, %i, %f" % (index, infolder, neighbor, alpha, max_iter, tol)

    for fileInfo in filesList:
        outLines = []
        outname = "./output/PredictedLabelSpreading/%s/" % infolder # assigned later
        inXtest = "%s/%s/%s_selectedFeatures_%i.dat" % (folder, infolder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
#        inXtest = "%s/%s/%s/%s_selectedFeatures_10_%i.dat" % (folder, infolder, debug, fileInfo[0], fileInfo[1])
#        inYtest = "%s/%s/%s/%s_labels_10_%i.dat" % (folder, infolder, debug, fileInfo[0], fileInfo[1])
#        outname += "test_"

        Xtest  = readFileMatrix(inXtest, fileInfo[1])
        Ytest  = readFileMatrix(inYtest, fileInfo[1])
        YOnes = -np.ones_like(Ytest)

        trainList = list(filesList)
        trainList.remove(fileInfo)
        [Xtrain, Ytrain] = createTrainingData(folder, trainList, infolder)
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


def calculateGridSpotRandomForest(args, include10, infolder):
    # want to try it out with each file selected separately as the test data
    # all the other files are combined to be the training data
    startAll = timeit.default_timer()
    print "\nRandomForest(%s) - %s" % (include10, infolder)

    for i in range(len(filesList90)):
        if(include10):
            fileInfo = filesList90[i]
            inXtest = "%s/%s/%s/%s_selectedFeatures_90_%i.dat" % (folder, infolder, folder90, fileInfo[0], fileInfo[1])
            inYtest = "%s/%s/%s/%s_labels_90_%i.dat" % (folder, infolder, folder90, fileInfo[0], fileInfo[1])
        else:
            fileInfo = filesList[i]
            inXtest = "%s/%s/%s_selectedFeatures_%i.dat" % (folder, infolder, fileInfo[0], fileInfo[1])
            inYtest = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])

        fileInfo10 = filesList10[i]
        outLines = []
        outname = "./output/PredictedRandomForest/%s/" % infolder # assigned later
        if(include10):
            outname += "include10_"
            inXtrain = "%s/%s/%s/%s_selectedFeatures_10_%i.dat" % (folder, infolder, folder10, fileInfo10[0], fileInfo10[1])
            inYtrain = "%s/%s/%s/%s_labels_10_%i.dat" % (folder, infolder, folder10, fileInfo10[0], fileInfo10[1])
        

        Xtest  = readFileMatrix(inXtest, fileInfo[1])
        Ytest  = readFileMatrix(inYtest, fileInfo[1])

        trainList = list(filesList)
        del trainList[i]
        [Xtrain, Ytrain] = createTrainingData(folder, trainList, infolder)

        if(include10):
            Xtrain10  = readFileMatrix(inXtrain, fileInfo10[1])
            Ytrain10  = readFileMatrix(inYtrain, fileInfo10[1])
            Xtrain = np.concatenate((Xtrain, Xtrain10), axis=0)
            Ytrain = np.concatenate((Ytrain, Ytrain10), axis=0)

#        print "X train shape %s" % str(Xtrain.shape)
#        print "X test shape %s" % str(Xtest.shape)
#        print "Y train shape %s" % str(Ytrain.shape)
#        print "Y test shape %s" % str(Ytest.shape)

        outLines.append("Test File:\n")
        outLines.append("inXtest: %s\n" % inXtest)
        outLines.append("inYtest: %s\n" % inYtest)

        outLines.append("\nVariables:\n")

        outLines.append("\nRead in Files Done\n")
        outLines.append("X train shape %s\n" % str(Xtrain.shape))
        outLines.append("X test shape %s\n" % str(Xtest.shape))
        outLines.append("Y train shape %s\n" % str(Ytrain.shape))
        outLines.append("Y test shape %s\n" % str(Ytest.shape))
        #outLines.append("total: %i\n" % (Xtrain.shape[0] + Xtest.shape[0]))
        outLines.append("\n")



        # CLASSIFY!
        outname += "RandomForest"
        clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

        start = timeit.default_timer()
        results = predict(clf, Xtrain, np.ravel(Ytrain), Xtest, np.ravel(Ytest), "%s_%s" % (outname, fileInfo[0]))
        stop = timeit.default_timer()
        results.append("   total time: %i secs\n" % (stop - start))
        results.append("   oob_score: %f\n" % (clf.oob_score_))

        outfile = file("%s_%s_results.txt" % (outname, fileInfo[0]), "w")

        for i in range (len(outLines)):
            outfile.write(outLines[i])

        for i in range (len(results)):
            outfile.write(results[i])

        outfile.close();

    stopAll = timeit.default_timer()
    print "Done predicting: %i secs" % (stopAll - startAll)


def calculateGridSpotGaussian(index, args, covar_type, infolder):
    # want to try it out with each file selected separately as the test data
    # all the other files are combined to be the training data
    startAll = timeit.default_timer()
    print "\nGaussian(%i) - %s: %s" % (index, infolder, covar_type)

    for fileInfo in filesList:
        outLines = []
        outname = "./output/PredictedGaussian/%s/" % infolder # assigned later
        inXtest = "%s/%s/%s_selectedFeatures_%i.dat" % (folder, infolder, fileInfo[0], fileInfo[1])
        inYtest = "%s/%s_labels_%i.dat" % (folder, fileInfo[0], fileInfo[1])
# used for debugging
#        inXtest = "%s/%s/%s/%s_selectedFeatures_10_%i.dat" % (folder, infolder, debug, fileInfo[0], fileInfo[1])
#        inYtest = "%s/%s/%s/%s_labels_10_%i.dat" % (folder, infolder, debug, fileInfo[0], fileInfo[1])
#        outname += "test_"

        Xtest  = readFileMatrix(inXtest, fileInfo[1])
        Ytest  = readFileMatrix(inYtest, fileInfo[1])

        trainList = list(filesList)
        trainList.remove(fileInfo)
        [Xtrain, Ytrain] = createTrainingData(folder, trainList, infolder)

#        print "X train shape %s" % str(Xtrain.shape)
#        print "X test shape %s" % str(Xtest.shape)
#        print "Y train shape %s" % str(Ytrain.shape)
#        print "Y test shape %s" % str(Ytest.shape)

        outLines.append("Test File:\n")
        outLines.append("inXtest: %s\n" % inXtest)
        outLines.append("inYtest: %s\n" % inYtest)

        outLines.append("\nVariables:\n")
        outLines.append("covar_type: %s\n" % covar_type)

        outLines.append("\nRead in Files Done\n")
        outLines.append("X train shape %s\n" % str(Xtrain.shape))
        outLines.append("X test shape %s\n" % str(Xtest.shape))
        outLines.append("Y train shape %s\n" % str(Ytrain.shape))
        outLines.append("Y test shape %s\n" % str(Ytest.shape))
        #outLines.append("total: %i\n" % (Xtrain.shape[0] + Xtest.shape[0]))
        outLines.append("\n")



        # CLASSIFY!
        outname += "Gaussian"
        clf = GMM(n_components=6, covariance_type=covar_type, init_params='wc', n_iter=20)

        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        meanMatrix = np.zeros(shape=(6,Xtrain.shape[1]), dtype=np.float64)
        for i in xrange(1,7):
            equals = np.ravel(Ytrain != i)
            m = np.zeros_like(Xtrain)
            for col in range(m.shape[1]):
                m[:, col] = equals
            separated = np.ma.masked_array(Xtrain, mask=m)
            mean = separated.mean(axis=0)
            meanMatrix[i-1] = mean

        clf.means_ = meanMatrix

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
parser.add_argument("-F", "--RandomForest", action="store_true", help="run Forest")
parser.add_argument("-G", "--GaussianMixture", action="store_true", help="run Gaussian Mixature Model")

random.seed(24)
args = parser.parse_args()
print args;

index = 1
if args.LabelSpreading:
    for infolder in infolders:
        for neighbor in n_neighbors:
            for alpha in alphas:
                for max_iter in max_iters:
                    for tol in tols:
                        if(index >= startIndex):
                            calculateGridSpotLabelSpreading(index, args, neighbor, alpha, max_iter, tol, infolder)
                        index +=1

if args.RandomForest:
    for infolder in infolders:
        calculateGridSpotRandomForest(args, False, infolder)
        calculateGridSpotRandomForest(args, True, infolder)


if args.GaussianMixture:
    for infolder in infolders:
        for i in range(len(covar_types)):
            calculateGridSpotGaussian(i, args, covar_types[i], infolder)



print "------------------------ALL DONE!!!!---------------------------------"






