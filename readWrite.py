import numpy as np
import os
from scipy import sparse
import csv


#---------------------READ/WRITE FUNCTIONS----------------------------------------
def readCSVFileToValueArray(fileName):
    csvfile = open(fileName)
    fileDict = csv.DictReader(csvfile)
    array = []
    for row in fileDict:
        array.append(float(row['Value']))
    return array

def readCSVFile(fileName):
    csvfile = open(fileName)
    fileDict = csv.DictReader(csvfile)
    return fileDict

def readTxtFile(fileName):
    f = open(fileName) 
    lines = f.read().splitlines()
    return lines




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


def writeOutSamples(trainX, trainY, testX, testY, fileNames, outfolder):
    ensure_dir("%s/split90" % outfolder)
    ensure_dir("%s/split10" % outfolder)
    ensure_dir("%s/debug" % outfolder)
    for i in range(len(fileNames)):
        writeFileMatrix(trainX[i], "%s/split90/%s_selectedFeatures_90_%i.dat" % (outfolder, fileNames[i], trainX[i].shape[0]))
        writeFileMatrix(trainY[i], "%s/split90/%s_labels_90_%i.dat" % (outfolder, fileNames[i], trainY[i].shape[0]))
        writeFileMatrix(testX[i], "%s/split10/%s_selectedFeatures_10_%i.dat" % (outfolder, fileNames[i], testX[i].shape[0]))
        writeFileMatrix(testY[i], "%s/split10/%s_labels_10_%i.dat" % (outfolder, fileNames[i], testY[i].shape[0]))
        writeFileMatrix(testX[i], "%s/debug/%s_selectedFeatures_10_%i.dat" % (outfolder, fileNames[i], testX[i].shape[0]))
        writeFileMatrix(testY[i], "%s/debug/%s_labels_10_%i.dat" % (outfolder, fileNames[i], testY[i].shape[0]))


def writeOutCombinedSamples(trainX, trainY, testX, testY, fileName):
    ensure_dir(fileName)
    trainXCombo = trainX[0]
    trainYCombo = trainY[0]
    testXCombo = testX[0]
    testYCombo = testY[0]

    for i in range(1, len(trainX)):
        trainXCombo = numpy.concatenate((trainXCombo, trainX[i]), axis=0)
        trainYCombo = numpy.concatenate((trainYCombo, trainY[i]), axis=0)
        testXCombo = numpy.concatenate((testXCombo, testX[i]), axis=0)
        testYCombo = numpy.concatenate((testYCombo, testY[i]), axis=0)

        
    writeFileMatrix(trainXCombo, "%s_selectedFeatures_train_%i.dat" % (fileName, trainXCombo.shape[0]))
    writeFileMatrix(trainYCombo, "%s_labels_train_%i.dat" % (fileName, trainYCombo.shape[0]))
    writeFileMatrix(testXCombo, "%s_selectedFeatures_test_%i.dat" % (fileName, testXCombo.shape[0]))
    writeFileMatrix(testYCombo, "%s_labels_test_%i.dat" % (fileName, testYCombo.shape[0]))

def writeOutSplitMatrix(featureMatrix, splits, fileNames, outfolder):
    ensure_dir(outfolder)
    separateMatrices = numpy.vsplit(featureMatrix, splits)
    for i in range(len(fileNames)):
        writeFileMatrix(separateMatrices[i], "%s/%s_selectedFeatures_%i.dat" % (outfolder, fileNames[i], separateMatrices[i].shape[0]))



