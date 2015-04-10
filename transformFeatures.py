#!/usr/bin/python
import os
import csv
import random
from sets import Set
from sklearn import preprocessing
import numpy as np

LabelOptions = ['np', 'c', 'e1', 'e2', 'd', 'g']
LabelToIntConversion = {'np': 1, 'c': 2, 'e1': 3, 'e2': 4, 'd': 5, 'g': 6}

def readFile(fileName):
    csvfile = open(fileName)
    fileDict = csv.DictReader(csvfile)
    return fileDict

def writeFile(featureMatrix, fileName):
    featureMatrix.tofile(fileName)



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

    secDict.pop()
    print testSet
    return secDict


def applyFourierTransform(secDict):
    matrix = numpy.zeros(shape=(len(truth_test),1), dtype=numpy.float64)

 

def main(inFileName, outFileName):
    # read in the csv file 
    inDict = readFile(inFileName)

    # group the milliseconds into second chunks
    secDict = groupIntoSeconds(inDict)
    print "Number of seconds: " + str(len(secDict))
    print "Number of voltages in an observation: " + str(len(secDict[len(secDict)-1]['VoltsArray']))

    # apply the fourier transform on each second chunk
    fourierFeatureMatrix = applyFourierTransform(secDict)
    writeFile(fourierFeatureMatrix, "%s_allFeatures_%s.dat" % (outFileName, len(secDict)))

    # scale all features to center around mean with variance 1
    featureMatrix_scaled = preprocessing.scale(fourierFeatureMatrix)

    # preform feature selection
    writeFile(featureMatrix, "%s_selectedFeatures_%s.dat" % (outFileName, len(secDict)))





# ----------------MAIN CALLS ---------------------------------
main("./LabeledData/17_Lab_Cmac_031214.csv","./output/17_Lab_Cmac_031214.txt")







