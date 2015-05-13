#!/usr/bin/python
import os
import csv
import random
import timeit
import numpy
from sets import Set
from sklearn import metrics

#-------------Global Config variables-----------------------
LabelOptions = ['np', 'c', 'e1', 'e2', 'd', 'g']
LabelToIntConversion = {'np': 1, 'c': 2, 'e1': 3, 'e2': 4, 'd': 5, 'g': 6}

#------------Variables that can easily be changed to affect output-------------------------
typeList = [["LabelSpreading", 21]]
            
#typeList = [["RandomForest"]]


filesList = [
#    ["04_Lab_FD_031114", 13571],
#    ["12_Lab_C_060514", 40504],
#    ["13_Lab_Cmac_031114", 6032],
#    ["17_Lab_Cmac_031214", 4988],
#    ["21_Lab_Corrizo_051614", 86619],
    ["29_Lab_Corrizo_051914", 73811]
#    ["31_Lab_Troyer_052114", 14482],
#    ["35_Lab_Val_100714", 47538]
]

predictedFolder = "./analyzeMinutes/input"
trueFolder = "./input"
secondsFolder = "./analyzeMinutes/inputSeconds"

outFolder = "./analyzeMinutes/output"

#-------------Input/Output functions----------------------------------------
def readFile(fileName):
    csvfile = open(fileName)
    fileDict = csv.DictReader(csvfile)
    array = []
    for row in fileDict:
        array.append(float(row['Value']))
    return array


def readFileMatrix(myfile, size):
    array = numpy.fromfile(myfile, dtype=numpy.float64, count=-1, sep="")
    array = numpy.reshape(array,(size,-1))
    return array


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


def groupValues(labels, minutesDict):
    newLabels = []

    for row in minutesDict:
        labelCount = [0, 0, 0, 0, 0, 0]
        for i in range(row['startIndex'], row['endIndex']+1):
            labelIndex = int(labels[i]) - 1
            currentCount = labelCount[labelIndex]
            labelCount[labelIndex] = currentCount + 1
        maxCount = max(labelCount)
        newLabels.append(labelCount.index(maxCount) + 1)

    return newLabels


def convertToDictLabels(values, classes):
    dictItem = {}
    for i in range(len(values)):
        dictItem["Label %i" % classes[i]] = values[i]
    return dictItem


#------------------------MAIN--------------------------

for fileInfo in filesList:
    for typeInfo in typeList:
        inTrueLabelsName = "%s/%s_labels_%i.dat" % (trueFolder, fileInfo[0], fileInfo[1])
        inPredLabelsName = "%s/%s_%s_%i_predictedValues.csv" % (predictedFolder, typeInfo[0], fileInfo[0], typeInfo[1])
#        inPredLabelsName = "%s/%s_%s_predictedValues.csv" % (predictedFolder, typeInfo[0], fileInfo[0])
        inSecondsName = "%s/%s_seconds_%i.dat" % (secondsFolder, fileInfo[0], fileInfo[1])
        outname = "%s/%s_%s_%i" % (outFolder, typeInfo[0], fileInfo[0], typeInfo[1])
#        outname = "%s/%s_%s" % (outFolder, typeInfo[0], fileInfo[0])

        trueLabelsArray = numpy.ravel(readFileMatrix(inTrueLabelsName, fileInfo[1]))
        predLabelsArray = readFile(inPredLabelsName)
        secondsArray = numpy.ravel(readFileMatrix(inSecondsName, fileInfo[1]))

        print len(trueLabelsArray)
        print len(predLabelsArray)
        print len(secondsArray)

        [minutesDict, minutes] = groupIntoMinutes(secondsArray)
        trueLabelsMinArray = groupValues(trueLabelsArray, minutesDict)
        predLabelsMinArray = groupValues(predLabelsArray, minutesDict)

        printDict = []
        for i in range(len(minutes)):
            printDict.append({"Minute": minutes[i], "True Labels": trueLabelsMinArray[i], "Predicted Labels": predLabelsMinArray[i]})
        writeFileArray(printDict, "%s/%s_%s_minuteValues.csv" % (outFolder, typeInfo[0], fileInfo[0]))


        dictScores = []
        results = []
        classes = [1,2,3,4,5,6]
        accuracy = metrics.accuracy_score(trueLabelsMinArray, predLabelsMinArray, normalize=True, sample_weight=None)
        results.append("   accuracy_score done: %s\n" % (str(accuracy)))

        f1_score = metrics.f1_score(trueLabelsMinArray, predLabelsMinArray, average=None)
        dictItem = convertToDictLabels(f1_score, classes)
        dictItem["Score Type"] = "f1_score"
        dictScores.append(dictItem)
        results.append("   f1_score done: %s\n" % (str(f1_score)))

        precision_score = metrics.precision_score(trueLabelsMinArray, predLabelsMinArray, average=None)
        dictItem = convertToDictLabels(precision_score, classes)
        dictItem["Score Type"] = "precision_score"
        dictScores.append(dictItem)
        results.append("   precision_score done: %s\n" % (str(precision_score)))

        recall_score = metrics.recall_score(trueLabelsMinArray, predLabelsMinArray, average=None)
        dictItem = convertToDictLabels(recall_score, classes)
        dictItem["Score Type"] = "recall_score"
        dictScores.append(dictItem)

        writeFileArray(dictScores, "%s_scores.csv" % (outname))
        results.append("   recall_score done: %s\n" % (str(recall_score)))

        confusion_matrix = metrics.confusion_matrix(trueLabelsMinArray, predLabelsMinArray, labels=[1,2,3,4,5,6])
        writeFileMatrixCSV(confusion_matrix, "%s_confusionMatrix.csv" % (outname))
            


        outfile = file("%s_results.txt" % (outname), "w")

        for i in range (len(results)):
            outfile.write(results[i])

        outfile.close();









