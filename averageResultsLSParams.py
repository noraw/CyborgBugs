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

featureFolders = ["12ImportantFeatures"]


typeFolders = [
    ["LabelSpreading", [1, 12]],
]

folder = "./output"

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

def findVariables(lines):
    dictVars = {}
    for line in lines:
        if(line.find("n_neighbor") != -1):
            parts = line.split(" ")
            dictVars["n_neighbor"] = parts[-1]
        if(line.find("alpha") != -1):
            parts = line.split(" ")
            dictVars["alpha"] = parts[-1]
        if(line.find("tol") != -1):
            parts = line.split(" ")
            dictVars["tol"] = parts[-1]
        if(line.find("max_iter") != -1):
            parts = line.split(" ")
            dictVars["max_iter"] = parts[-1]

    return dictVars
    
#------------------------MAIN-------------------------------------
def main():
    resultsDict = []
    outfile = "%s/LS_n-2_Average_results.csv" % (folder)
    for typeFolderItem in typeFolders:
        typeFolder = typeFolderItem[0]
        indices = typeFolderItem[1]
        for index in range(indices[0], indices[1]+1):
            secondsList = []
            accuracyList = []
            dictVars = {}
            for insectPair in filesList:
                insectFile = insectPair[0]
                if(index != -1):
                    insectFile += "_%i" % index
                infile = "%s/Predicted%s_n-2/12ImportantFeatures/%s_%s_results.txt" % (folder, typeFolder, typeFolder, insectFile)
                lines = readTxtFile(infile)
                secondsList.append(findTotalSeconds(lines))
                accuracyList.append(findAccuracy(lines))
                dictVars = findVariables(lines)
            
            resultsDict.append({
                "Classifier":typeFolder, 
                "Index": index, 
                "n neighbor": dictVars["n_neighbor"],
                "alpha": dictVars["alpha"],
                "max iterations": dictVars["max_iter"],
                "tol": dictVars["tol"],
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









