#!/usr/bin/python
import os
import csv
import random
import timeit
import numpy
from sets import Set
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from readWrite import *

#-------------Global Config variables-----------------------
LabelOptions = ['np', 'c', 'e1', 'e2', 'd', 'g']
LabelToIntConversion = {'np': 1, 'c': 2, 'e1': 3, 'e2': 4, 'd': 5, 'g': 6}

filesList = [
    "04_Lab_FD_031114",
    "12_Lab_C_060514",
    "13_Lab_Cmac_031114",
    "17_Lab_Cmac_031214",
    "21_Lab_Corrizo_051614",
    "29_Lab_Corrizo_051914",
    "31_Lab_Troyer_052114",
    "35_Lab_Val_100714"
]

outfolders = [
    "./input/12ImportantFeatures",
    "./input/12LargestMag",
    "./input/12LargestFreq",
    "./input/12LargestMagFreq"
]

#------------Variables that can easily be changed to affect output-------------------------
lenOfFourier = 100
numFeatures = 12
finalSeed = 24
trainPercent = .9


#-------------Data functions----------------------------------------------------------
def groupIntoSeconds(inDict):
    secDict = []
    intSec = -1
    firstSec = True
    voltsArray = []
    testSet = Set()
    for row in inDict:
        if (intSec == -1):
            intSec = int(float(row['sec']))
        if (firstSec and intSec != int(float(row['sec']))):
            firstSec = False
            intSec = int(float(row['sec']))
            label = row['Label']
        if (not firstSec):
            #print "Sec: " + str(intSec) + "      " + row['sec']
            if(intSec == int(float(row['sec']))):
                voltsArray.append(float(row['Volts']))
            else:
                #print len(voltsArray)
                if(label in LabelOptions):
                    testSet.add(label)
                    secDict.append({'sec': intSec, 
                        'Label': LabelToIntConversion[label], 
                        'VoltsArray': voltsArray})
                intSec = int(float(row['sec']))
                label = row['Label']
                voltsArray = [float(row['Volts'])]
        #if(len(secDict) > 300): break
    secDict.pop()
    print testSet
    return secDict


def applyFourierTransform(secDict):
    matrix = numpy.zeros(shape=(len(secDict),lenOfFourier), dtype=numpy.float64)
    rowId = 0
    for row in secDict:
        fourierList = numpy.fft.fft(row['VoltsArray'], lenOfFourier)
        mag = numpy.abs(fourierList)
        for i in range(lenOfFourier):
            matrix[rowId][i] = mag[i]
        rowId +=1
    matrices = numpy.hsplit(matrix, [lenOfFourier/2])
    return matrices[0]

def getMatrixOfSeconds(secDict):
    matrix = numpy.zeros(shape=(len(secDict),1), dtype=numpy.float64)
    rowId = 0
    for row in secDict:
        matrix[rowId] = row['sec']
        rowId +=1
    return matrix

def getMatrixOfLabels(secDict):
    matrix = numpy.zeros(shape=(len(secDict),1), dtype=numpy.float64)
    rowId = 0
    for row in secDict:
        matrix[rowId] = row['Label']
        rowId +=1
    return matrix
 
def featureSelection(matrixX, matrixY, seed, fileName):
        clf = RandomForestClassifier(n_estimators=240,
            random_state=seed,
            oob_score=True)
        clf.fit(matrixX, numpy.ravel(matrixY))
        featureMatrix = clf.transform(matrixX)
        accuracy = clf.score(matrixX, matrixY)
        oob_score = clf.oob_score_

        # print out oob_score and accuracy
        dictionary = [{"ID":"oob_score", "Value":oob_score}]
        dictionary.append({"ID":"Accuracy", "Value":accuracy})
        for i in range(len(clf.feature_importances_)):
            dictionary.append({"ID":i+1, "Value":clf.feature_importances_[i]})
        writeFileArray(dictionary, "%s_featureImportance_seed-%i.csv" % (fileName, seed))
        return [clf, featureMatrix]

def featureSelectionImportanceTrimed(matrixX, matrixY, fileName):
    [clf, featureMatrix] = featureSelection(matrixX, matrixY, finalSeed, fileName)
    return numpy.hsplit(featureMatrix, [numFeatures])[0]

def featureSelectionLargestMag(matrixX, matrixY, fileName):
    [rows, cols] = matrixX.shape
    sortedMatrix = numpy.zeros(shape=(rows,numFeatures), dtype=numpy.float64)
    for i in range(rows):
        row = matrixX[i]
        sortedRowIndex = numpy.argsort(abs(row))[::-1]
        for j in range(numFeatures):
            sortedMatrix[i][j] = row[sortedRowIndex[j]]
    return sortedMatrix


def featureSelectionLargestFreq(matrixX, matrixY, fileName):
    [rows, cols] = matrixX.shape
    sortedMatrix = numpy.zeros(shape=(rows,numFeatures), dtype=numpy.float64)
    for i in range(rows):
        row = matrixX[i]
        sortedRowIndex = numpy.argsort(abs(row))[::-1]
        for j in range(numFeatures):
            sortedMatrix[i][j] = sortedRowIndex[j]
    return sortedMatrix


def featureSelectionLargestMagFreq(matrixX, matrixY, fileName):
    [rows, cols] = matrixX.shape
    sortedMatrix = numpy.zeros(shape=(rows,numFeatures*2), dtype=numpy.float64)
    for i in range(rows):
        row = matrixX[i]
        sortedRowIndex = numpy.argsort(abs(row))[::-1]
        for j in range(numFeatures):
            sortedMatrix[i][j] = row[sortedRowIndex[j]]
            sortedMatrix[i][j+numFeatures] = sortedRowIndex[j]
    return sortedMatrix

def featureSelectionTestOptions(matrixX, matrixY, fileName):
    seeds = [24]#[0, 7, 16, 1, 24, 72]#, 48, 96, 28, 56, 112]
    for seed in seeds:
        start = timeit.default_timer()
        [clf, featureMatrix] = featureSelection(matrixX, matrixY, seed, fileName)
        print "Size of Feature Matrix(Seed %i): (%i, %i)" % (seed, featureMatrix.shape[0], featureMatrix.shape[1])
        # check out print out each accuracy with the number of features next to it.
        # range number of features from default to 1
        # train and score on shortened feature set (fit and score function)
        dictionary = [{"Number of Features":0, "oob_score":0, "Accuracy":0, "Time (secs)":0}]
        if(numFeatures != -1):
            i = featureMatrix.shape[1]
            while i > numFeatures:
                start1 = timeit.default_timer()
                clf.fit(featureMatrix, numpy.ravel(matrixY))
                accuracy = clf.score(featureMatrix, matrixY)
                oob_score = clf.oob_score_
                stop1 = timeit.default_timer()
                dictionary.append({"Number of Features":i, "oob_score":oob_score, "Accuracy":accuracy, "Time (secs)":stop1-start1})

                featureMatrix = numpy.delete(featureMatrix, featureMatrix.shape[1]-1, 1)
                print "    Getting score %i: %i secs" % (i, stop1-start1)
                i -= 1
            writeFileArray(dictionary, "%s_featureScores_seed-%i.csv" % (fileName, seed))
        stop = timeit.default_timer()
        print "Lasted %i secs" % (stop-start)
    return featureMatrix


def selectSubsample(matrixX, matrixY, splits):
    separateX = numpy.vsplit(matrixX, splits)
    separateY = numpy.vsplit(matrixY, splits)
    trainXFinal = []
    trainYFinal = []
    testXFinal = []
    testYFinal = []
    for i in range(len(separateX)):
        x_train, x_test, y_train, y_test = train_test_split(
            separateX[i], separateY[i], train_size=trainPercent, random_state=finalSeed)
        trainXFinal.append(x_train)
        trainYFinal.append(y_train)
        testXFinal.append(x_test)
        testYFinal.append(y_test)
    return [trainXFinal, trainYFinal, testXFinal, testYFinal]



#------------------------MAIN--------------------------
# does not perform feature selection
# does scale features
def processFile(inFileName, outFileName):
    print "%s -> %s" % (inFileName, outFileName)
    # read in the csv file 
    inDict = readCSVFile(inFileName)

    # group the milliseconds into second chunks
    secDict = groupIntoSeconds(inDict)
    labelsMatrix = getMatrixOfLabels(secDict)
    print "Number of seconds observed: " + str(len(secDict))
    print "Number of voltages in an observation: " + str(len(secDict[len(secDict)-1]['VoltsArray']))
    writeFileMatrix(labelsMatrix, "%s_labels_%s.dat" % (outFileName, len(secDict)))
    writeFileMatrix(getMatrixOfSeconds(secDict), "%s_seconds_%s.dat" % (outFileName, len(secDict)))

    # apply the fourier transform on each second chunk
    fourierFeatureMatrix = applyFourierTransform(secDict)
    writeFileMatrix(fourierFeatureMatrix, "%s_allFeatures_%s.dat" % (outFileName, len(secDict)))
    print "Size of Feature Matrix: (%i, %i)" % (fourierFeatureMatrix.shape[0], fourierFeatureMatrix.shape[1])

    # scale all features to center around mean with variance 1
    featureMatrix_scaled = preprocessing.scale(fourierFeatureMatrix)

    print "\n"
    return [featureMatrix_scaled, labelsMatrix]



def readInFilesCombined(filesList):
    featuresFinal = None
    labelsFinal = None
    splits = []
    total = 0
    outCombinedFile = ""
    for fileRow in filesList:
        inFile = "./LabeledData/%s.csv" % fileRow
        outFile = "./input/%s" % (fileRow)
        outCombinedFile += "%s--" % fileRow
        if(featuresFinal == None):
            [featuresFinal, labelsFinal] = processFile(inFile, outFile)
            total += labelsFinal.shape[0]
        else:
            [features, labels] = processFile(inFile, outFile)
            total += labels.shape[0]
            featuresFinal = numpy.concatenate((featuresFinal, features), axis=0)
            labelsFinal = numpy.concatenate((labelsFinal, labels), axis=0)
        splits.append(total)

    splits.pop()
    print "Size of Combined Feature Matrix: (%i, %i)" % (featuresFinal.shape[0], featuresFinal.shape[1])
    print "Size of Combined Labels Matrix: (%i, %i)" % (labelsFinal.shape[0], labelsFinal.shape[1])

    return [featuresFinal, labelsFinal, splits, outCombinedFile]

# iterates over list and processes files
# performs feature selection after combining all the data in all the files
def mainAll():
    print "\nReading in the files..."
    [featuresFinal, labelsFinal, splits, outCombinedFile] = readInFilesCombined(filesList)

    for i in range(len(outfolders)):
        start = timeit.default_timer()
        outfolder = outfolders[i]

        # preform feature selection
        print "Performing Feature Selection...  %s" % outfolder
        # used to create different results for seeds and number of features
    #    featureMatrix = featureSelectionTestOptions(featuresFinal, labelsFinal, outCombinedFile)
        if(i == 0):
            featureMatrix = featureSelectionImportanceTrimed(featuresFinal, labelsFinal, "%s/%s" % (outfolder, outCombinedFile))
        elif(i == 1):
            featureMatrix = featureSelectionLargestMag(featuresFinal, labelsFinal, "%s/%s" % (outfolder, outCombinedFile))
        elif(i == 2):
            featureMatrix = featureSelectionLargestFreq(featuresFinal, labelsFinal, "%s/%s" % (outfolder, outCombinedFile))
        elif(i == 3):
            featureMatrix = featureSelectionLargestMagFreq(featuresFinal, labelsFinal, "%s/%s" % (outfolder, outCombinedFile))

        print "Size of Selected Feature Matrix: (%i, %i)" % (featureMatrix.shape[0], featureMatrix.shape[1])
        writeFileMatrix(featureMatrix, "%s/%s_selectedFeatures_%s.dat" % (outfolder, outCombinedFile, featureMatrix.shape[0]))

        # split into individual files
        print "Write out separate selected"
        writeOutSplitMatrix(featureMatrix, splits, filesList, outfolder)

        # split into training and test data
        print "select subsamples"
        [trainXList, trainYList, testXList, testYList] = selectSubsample(featureMatrix, labelsFinal, splits)
        print "write out separate samples"
        writeOutSamples(trainXList, trainYList, testXList, testYList, filesList, outfolder)
        print "write out combined samples"
        writeOutCombinedSamples(trainXList, trainYList, testXList, testYList, "%s/%s" % (outfolder, outCombinedFile))


        stop = timeit.default_timer()
        print "\nRuntime: " + str(stop - start)


# ----------------MAIN CALLS ---------------------------------


#processFile("./LabeledData/%s.csv" % filesList[0], "./output/%s" % filesList[0])


mainAll()





