# ********************************************
# Author: Nora Coler
# Date: April 20, 2015
#
#
# ********************************************
import numpy
import argparse
import os
import random
import timeit
from math import sqrt
from readWrite import *

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

featureFolders = [
    "12ImportantFeatures",
    "12LargestMag",
    "12LargestFreq",
    "12LargestMagFreq"
]

typeFolders = [
    ["Gaussian", [3]],
    ["LabelSpreading", [1]],
    ["RandomForest", [-1]]
]

folder = "./output/minutesTransitions"

#----------------------FUNCTIONS-------------------------

def findTotalSeconds(lines):
    for line in lines:
        if(line.find("total time") != -1):
            parts = line.split(" ")
            return float(parts[-2])


def findAccuracy(lines):
    for line in lines:
        if(line.find("accuracy_score") != -1):
            parts = line.split(" ")
            return float(parts[-1]) * 100
    
#------------------------MAIN-------------------------------------
def main():
    resultsDict = []
    outfile = "%s/Average_results.csv" % (folder)
    for typeFolderItem in typeFolders:
        typeFolder = typeFolderItem[0]
        indices = typeFolderItem[1]
        for featureFolder in featureFolders:
            for index in indices:
                secondsList = []
                accuracyList = []
                for insectPair in filesList:
                    insectFile = insectPair[0]
                    if(index != -1):
                        insectFile += "_%i" % index
                    infile = "%s/Predicted%s/%s/%s_%s_results.txt" % (folder, typeFolder, featureFolder, typeFolder, insectFile)
                    lines = readTxtFile(infile)
                    secondsList.append(findTotalSeconds(lines))
                    accuracyList.append(findAccuracy(lines))
                    
                
                resultsDict.append({
                    "Classifier":typeFolder, 
                    "Feature Set": featureFolder, 
                    "Index": index, 
                    "Average Accuracy": "%.2f" % numpy.average(accuracyList),
                    "Average Time (sec)": "%.2f" % numpy.average(secondsList),
                    "Max Accuracy": "%.2f" % max(accuracyList),
                    "Max Accuracy Pair": filesList[accuracyList.index(max(accuracyList))][0],
                    "Min Accuracy": "%.2f" % min(accuracyList),
                    "Min Accuracy Pair": filesList[accuracyList.index(min(accuracyList))][0]
                })

    writeFileArray(resultsDict, outfile)


# -------------------END--------------
main()









