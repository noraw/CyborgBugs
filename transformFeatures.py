#!/usr/bin/python
import os
import csv
import random
import timeit
import numpy
from sets import Set
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier


#-------------Global Config variables-----------------------
LabelOptions = ['np', 'c', 'e1', 'e2', 'd', 'g']
LabelToIntConversion = {'np': 1, 'c': 2, 'e1': 3, 'e2': 4, 'd': 5, 'g': 6}

#------------Variables that can easily be changed to affect output-------------------------
lenOfFourier = 240
numFeatures = 12


#-------------Input/Output functions----------------------------------------
def readFile(fileName):
    csvfile = open(fileName)
    fileDict = csv.DictReader(csvfile)
    return fileDict

def writeFileMatrix(matrix, fileName):
    matrix.tofile(fileName)

def writeFileArray(array, fileName):
    # output feature importance for graphs
    outfile = file(fileName, "w")
    outfile.write('"ID","Value"\n');
    for i in range (len(array)):
        outLine = str(i) + ","
        outLine += str(array[i])
        outLine += "\n"
        outfile.write(outLine)
    outfile.close();


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
    matrix = numpy.zeros(shape=(len(secDict),lenOfFourier*2), dtype=numpy.float64)
    rowId = 0
    for row in secDict:
        fourierList = numpy.fft.fft(row['VoltsArray'], lenOfFourier)
        mag = numpy.abs(fourierList)
        angle = numpy.angle(fourierList)
        for i in range(lenOfFourier):
            matrix[rowId][i] = mag[i]
            matrix[rowId][i+lenOfFourier] = angle[i]
        rowId +=1
    return matrix

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
 
def featureSelection(matrixX, matrixY, fileName):
    clf = ExtraTreesClassifier(n_estimators=240,
                          random_state=0)
    X_new = clf.fit(matrixX, numpy.ravel(matrixY))
    writeFileArray(clf.feature_importances_, "%s_featureImportance.csv" % (fileName))

    featureMatrix = clf.transform(matrixX)
    if(numFeatures != -1):
        i = featureMatrix.shape[1]
        while i > numFeatures:
            featureMatrix = numpy.delete(featureMatrix, featureMatrix.shape[1]-1, 1)
            i -= 1

    return featureMatrix




#------------------------MAIN--------------------------
def processFile(inFileName, outFileName):
    print "%s -> %s" % (inFileName, outFileName)
    # read in the csv file 
    inDict = readFile(inFileName)

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
    return [fourierFeatureMatrix, labelsMatrix]





def mainAll(filesList):
    start = timeit.default_timer()
    featuresFinal = None
    labelsFinal = None
    outCombinedFile = "./output/"
    for fileRow in filesList:
        inFile = "./LabeledData/%s.csv" % fileRow
        outFile = "./output/%s" % fileRow
        outCombinedFile += "%s--" % fileRow
        if(featuresFinal == None):
            [featuresFinal, labelsFinal] = processFile(inFile, outFile)
        else:
            [features, labels] = processFile(inFile, outFile)
            featuresFinal = numpy.concatenate((featuresFinal, features), axis=0)
            labelsFinal = numpy.concatenate((labelsFinal, labels), axis=0)
            break

    print "Size of Combined Feature Matrix: (%i, %i)" % (featuresFinal.shape[0], featuresFinal.shape[1])
    print "Size of Combined Labels Matrix: (%i, %i)" % (labelsFinal.shape[0], labelsFinal.shape[1])

    # preform feature selection
    print "Performing Feature Selection..."
    featureMatrix = featureSelection(featuresFinal, labelsFinal, outCombinedFile)
    print "Size of Selected Feature Matrix: (%i, %i)" % (featureMatrix.shape[0], featureMatrix.shape[1])
    stop = timeit.default_timer()

    writeFileMatrix(featureMatrix, "%s_selectedFeatures_%s.dat" % (outCombinedFile, featureMatrix.shape[0]))

    print "\nRuntime: " + str(stop - start)


# ----------------MAIN CALLS ---------------------------------

filesList = [
    "17_Lab_Cmac_031214",
    "17_Lab_Cmac_031214"
#    ["./LabeledData/.csv","./output/"]
#    ["./LabeledData/.csv","./output/"]
]

#processFile(filesList[0][0], filesList[0][1])

mainAll(filesList)





