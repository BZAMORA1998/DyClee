import numpy as np
from config import getTimeSeriesToyDatasetName, getTimeSeriesDatasetsPath

def getTimeSeriesDatasetFromFolder():
    fileFullName = getTimeSeriesToyDatasetName() + ".csv"
    filePath = getTimeSeriesDatasetsPath() + fileFullName
    # get the k param from header
    ##obtener el parámetro k del encabezado
    # with open(filePath, newline='') as f:
    #Con abierto (filePath, newline = '') como f:
    #     reader = csv.reader(f)
    #     header = next(reader)  # gets the first line
    # k = header[0] # 'header' returns ['k',]

    # get the data //obtener los datos
    ndarray = np.genfromtxt(filePath, delimiter=",", ) # skip_header=1 if header must be skipped
    fileNameWithoutExtension = fileFullName.split(".")[0]
    return ndarray
