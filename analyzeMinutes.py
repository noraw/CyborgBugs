#!/usr/bin/python
import os
import random
import timeit
import numpy
from sets import Set
from sklearn import metrics
from readWrite import *

#-------------Global Config variables-----------------------
LabelOptions = ['np', 'c', 'e1', 'e2', 'd', 'g']
LabelToIntConversion = {'np': 1, 'c': 2, 'e1': 3, 'e2': 4, 'd': 5, 'g': 6}

#------------Variables that can easily be changed to affect output-------------------------
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

featureFolders = [
    "12ImportantFeatures",
    "12LargestMag",
    "12LargestFreq",
    "12LargestMagFreq"
]

typeFolders = [
    ["Gaussian", [0,1,2,3]],
    ["LabelSpreading", [1]],
    ["RandomForest", [-1]]
]

folder = "./output"

trueFolder = "./input"

outFolders = [
    ["./output/minutes", False],
    ["./output/minutesTransitions", True]
]


#-------------Data functions----------------------------------------------------------
def groupIntoMinutes(secondsArray):
    minDict = []
    minutes = []
    startSecond = secondsArray[0]
    endSecond = secondsArray[-1]
    startMinute = int(startSecond/60)
    endMinute = int(endSecond/60)

    startIndex = 0
    for i in range(startMinute, endMinute+1):
        endIndex = startIndex + ((i+1)*60-secondsArray[startIndex]) - 1
        endIndex = min(endIndex, len(secondsArray)-1)
        row = {"minute":i, "startIndex": int(startIndex), "endIndex": int(endIndex)}
        minDict.append(row)
        minutes.append(i)
        startIndex = endIndex + 1
    return [minDict, minutes]


def groupValues(labels, minutesDict, doTransitions):
    newLabels = []

    for row in minutesDict:
        labelCount = [0, 0, 0, 0, 0, 0]
        for i in range(row['startIndex'], row['endIndex']+1):
            labelIndex = int(labels[i]) - 1
            currentCount = labelCount[labelIndex]
            labelCount[labelIndex] = currentCount + 1
        maxCount = max(labelCount)
        if(doTransitions and maxCount < 50):
            newLabels.append(7)
        else:
            newLabels.append(labelCount.index(maxCount) + 1)

    return newLabels


def convertToDictLabels(values, classes):
    dictItem = {}
    for i in range(len(values)):
        dictItem["Label %i" % classes[i]] = values[i]
    return dictItem



#------------------------MAIN--------------------------
def processFile(outFolderItem, typeFolderItem, featureFolder, index, fileInfo):
    start = timeit.default_timer()
    doTransitions = outFolderItem[1]
    typeFolder = typeFolderItem[0]
    insectFile = fileInfo[0]

    if(index != -1):
        insectFile += "_%i" % index
    inPredLabelsName = "%s/Predicted%s/%s/%s_%s_predictedValues.csv" % \
        (folder, typeFolder, featureFolder, typeFolder, insectFile)
    inTrueLabelsName = "%s/%s_labels_%i.dat" % (trueFolder, fileInfo[0], fileInfo[1])
    inSecondsName = "%s/%s_seconds_%i.dat" % (trueFolder, fileInfo[0], fileInfo[1])
    outname = "%s/Predicted%s/%s/%s_%s" % \
        (outFolderItem[0], typeFolder, featureFolder, typeFolder, insectFile)

#    print inPredLabelsName
#    print inTrueLabelsName
#    print inSecondsName
#    print outname
#    print "\n"
    trueLabelsArray = numpy.ravel(readFileMatrix(inTrueLabelsName, fileInfo[1]))
    predLabelsArray = readCSVFileToValueArray(inPredLabelsName)
    secondsArray = numpy.ravel(readFileMatrix(inSecondsName, fileInfo[1]))

    [minutesDict, minutes] = groupIntoMinutes(secondsArray)
    trueLabelsMinArray = groupValues(trueLabelsArray, minutesDict, doTransitions)
    predLabelsMinArray = groupValues(predLabelsArray, minutesDict, doTransitions)

    writeFileList(predLabelsMinArray, "%s_predictedValues.csv" % (outname))
    printDict = []
    for i in range(len(minutes)):
        printDict.append({"Minute": minutes[i], "True Labels": trueLabelsMinArray[i], "Predicted Labels": predLabelsMinArray[i]})

    writeFileArray(printDict, "%s_trueAndPredictedValues.csv" % (outname))


    dictScores = []
    results = []
    classes = [1,2,3,4,5,6, 7]

    results.append("inPredLabelsName: %s\n" % inPredLabelsName)
    results.append("inTrueLabelsName: %s\n" % inTrueLabelsName)
    results.append("inSecondsName: %s\n" % inSecondsName)
    results.append("\n")


    accuracy = metrics.accuracy_score(trueLabelsMinArray, predLabelsMinArray, normalize=True, sample_weight=None)
    results.append("   accuracy_score done: %s\n" % (str(accuracy)))

    f1_score = metrics.f1_score(trueLabelsMinArray, predLabelsMinArray, average=None, labels=classes)
    dictItem = convertToDictLabels(f1_score, classes)
    dictItem["Score Type"] = "f1_score"
    dictScores.append(dictItem)
    results.append("   f1_score done: %s\n" % (str(f1_score)))

    precision_score = metrics.precision_score(trueLabelsMinArray, predLabelsMinArray, average=None, labels=classes)
    dictItem = convertToDictLabels(precision_score, classes)
    dictItem["Score Type"] = "precision_score"
    dictScores.append(dictItem)
    results.append("   precision_score done: %s\n" % (str(precision_score)))

    recall_score = metrics.recall_score(trueLabelsMinArray, predLabelsMinArray, average=None, labels=classes)
    dictItem = convertToDictLabels(recall_score, classes)
    dictItem["Score Type"] = "recall_score"
    dictScores.append(dictItem)

    writeFileArray(dictScores, "%s_scores.csv" % (outname))
    results.append("   recall_score done: %s\n" % (str(recall_score)))
    stop = timeit.default_timer()
    results.append("   total time: %f secs\n" % (stop - start))

    confusion_matrix = metrics.confusion_matrix(trueLabelsMinArray, predLabelsMinArray, labels=classes)
    writeFileMatrixCSV(confusion_matrix, "%s_confusionMatrix.csv" % (outname))

    outfile = file("%s_results.txt" % (outname), "w")
    for i in range (len(results)):
        outfile.write(results[i])
    outfile.close();





def main():
    for outFolderItem in outFolders:
        if(outFolderItem[1]):
            print "\nGrouping Minutes with Transitions..."
        else:
            print "\nGrouping Minutes..."
        for typeFolderItem in typeFolders:
            print "   %s" % typeFolderItem[0]
            for featureFolder in featureFolders:
                print "      %s" % featureFolder
                indices = typeFolderItem[1]
                for index in indices:
                    for fileInfo in filesList:
                        processFile(outFolderItem, typeFolderItem, featureFolder, index, fileInfo)





# -------------------END--------------
main()










