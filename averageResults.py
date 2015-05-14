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

featureFolders = [
    "12ImportantFeatures",
    "12LargestMag",
    "12LargestFreq",
    "12LargestMagFreq"
]

typeFolders = [
    "LabelSpreading",
    "Gaussian",
    "RandomForest"
]

folder = "./output"

#----------------------FUNCTIONS-------------------------

def readFile(fileName):
    f = open(fileName) 
    lines = f.readlines()
    return lines


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


def findTotalSeconds(lines):
    for line in lines:
        if(line.find("total time") != -1):
            parts = line.split(" ")
            print parts
            return int(parts[-2])


def findAccuracy(lines):
    for line in lines:
        if(line.find("accuracy_score") != -1):
            parts = line.split(" ")
            print parts
            return float(parts[-1] * 100)
    
#------------------------MAIN-------------------------------------
def main():
    resultsDict = []
    outfile = "%s/Average_results.txt" % (folder)
    for typeFolder in typeFolders:
        for featureFolder in featureFolders:
            print outfile
            secondsList = []
            accuracyList = []
            for insectPair in filesList:
                if(typeFolder == "LabelSpreading"):
                    insectPair += "_1"
                infile = "%s/Predicted%s/%s/%s_%s_results.txt" % (folder, typeFolder, featureFolder, typeFolder, insectPair)
                lines = readFile(infile)
                secondsList.append(findTotalSeconds(lines))
                accuracyList.append(findAccuracy(lines))
                
            
            resultsDict.append({
                "Classifier":typeFolder, 
                "Feature Set": featureFolder, 
                "Average Accuracy": numpy.average(accuracyList),
                "Average Time (sec)": numpy.average(secondsList),
                "Max Accuracy": max(accuracyList),
                "Max Accuracy Pair": filesList[accuracyList.index(max(accuracyList))],
                "Min Accuracy": min(accuracyList),
                "Min Accuracy Pair": filesList[accuracyList.index(min(accuracyList))]
            })

    writeFileArray(resultsDict, outfile)


# -------------------END--------------
main()









