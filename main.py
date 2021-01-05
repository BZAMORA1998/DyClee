from sklearn.preprocessing import StandardScaler
from utils.non_time_series_datasets_fetcher import getDatasetsFromFolder
from utils.time_series_dataset_fetcher import getTimeSeriesDatasetFromFolder
from utils.dyclee import Dyclee
from utils.persistor import storeAlgoConfig, storeTimeSeriesResult, storeNonTimeSeriesResult
from config import getClusteringResultsPath, getDycleeName, getTimeSeriesToyDatasetName, getNonTimeSeriesDatasetsPath
import numpy as np
from utils.bounding_box import BoundingBox
#librerias internas del proyecto de dyclee para su correcto uso y funcionamiento

def prepareResultFrom(currMicroClusters):
    res = []
    for mc in currMicroClusters:
        centroid = mc.getCentroid()
        label = mc.label
        row = [centroid[0], centroid[1], label]
        res.append(row)
    return np.array(res)

# GRUPOS DE DATOS QUE NO SON SERIES TEMPORALES

# PARAMENTROS
relativeSize=0.06           #TAMAÑO RELATIVO DE loss mmicroclustering del proyecto DYCLEE
uncommonDimensions = 0      #DIMENSIONES POCO COMUNES
closenessThreshold = 1.5    #UMBRAL DE CERCANIA

#dimesion o la distancia maas  cercana al conjunto de informacion
# OBTENER LOS CONJUNTOS DE DATOS DEL ARCHIVO CVD es aqui donde trae informacion de loss datoss en nontimeseries
non_time_series_datasets = getDatasetsFromFolder(getNonTimeSeriesDatasetsPath())

# CONTEXTO DE DATOS
dataContext = [BoundingBox(minimun=-2 , maximun=2),
               BoundingBox(minimun=-2 , maximun=2)]

# ITERAR SOBRE LOS CONJUNTOS DE DATOS
# len : sirve para saber cuantos item hay en non_time_series_datasets.
# range: crea la secuencia.

for datIndx in range(len(non_time_series_datasets)):
    # NUEVO DYCLEE PARA CADA CONJUNTO DE DATOS
    dyclee = Dyclee(dataContext=dataContext, relativeSize=relativeSize, uncommonDimensions=uncommonDimensions,
                    closenessThreshold=closenessThreshold)
    # Comienza
    # Este metodo se utiliza obvia algunos metodos porque no tiene la marca de tiempo
    X = non_time_series_datasets[datIndx]['dataset']
    dName = non_time_series_datasets[datIndx]['name']
    k = non_time_series_datasets[datIndx]['k']
    baseFolder = getClusteringResultsPath() + dName + '/'
    # normalize dataset for easier parameter selection
    #normalizar el conjunto de datos para facilitar la selección de parámetros
    X = StandardScaler().fit_transform(X)
    ac = 0  # processed samples
           #muestras procesadas
    # iterate over the data points
    #iterar sobre los puntos de datos
    for dataPoint in X:  # column index
        # índice de columna
        ac += 1
        #Cada data point va ha enviar al algoritmo de clustering
        dyclee.trainOnElement(dataPoint)
    currMicroClusters = dyclee.getClusteringResult()  # queremos mostrar el agrupamiento al final, solo una vez
    res = prepareResultFrom(currMicroClusters)
    folder = baseFolder + getDycleeName() + '/'
    storeNonTimeSeriesResult(res, folder)
    # store algo config
    # almacenar algo config
    algoConfig = dyclee.getConfig()
    storeAlgoConfig(algoConfig, folder)



# # TIME SERIES DATA SET CLUSTERING -------------------------------------------------------------------------
#AGRUPACIÓN DE CONJUNTOS DE DATOS DE SERIE DE TIEMPO
# # initialization
#inicialización
# relativeSize=0.02
# speed = 50
# lambd = 0.7 # if it has a value over 0, when a micro cluster is updated, tl will be checked and the diff with current time will matter
# periodicUpdateAt = 2
# timeWindow = 4
# periodicRemovalAt = 4
# closenessThreshold = 0.8
#
# #data context
##contexto de datos
# dataContext = [BoundingBox(minimun=-2 , maximun=2),
#                BoundingBox(minimun=-2 , maximun=2)]
#
# dyclee = Dyclee(dataContext = dataContext, relativeSize = relativeSize, speed = speed, lambd = lambd,
#                 periodicRemovalAt = periodicRemovalAt, periodicUpdateAt = periodicUpdateAt,
#                 timeWindow = timeWindow, closenessThreshold = closenessThreshold)
#
# tGlobal = 200
# ac = 0 # represents amount of processed elements
# folder = getClusteringResultsPath() + getTimeSeriesToyDatasetName() + '/' + getDycleeName() + '/'
#
# for point in getTimeSeriesDatasetFromFolder():
#     ac += 1
#     dyclee.trainOnElement(point)
#     if ac % tGlobal == 0:
#         currMicroClusters = dyclee.getClusteringResult()
#         res = prepareResultFrom(currMicroClusters)
#         storeTimeSeriesResult({"processedElements": ac, "result": res}, folder)
# algoConfig = dyclee.getConfig()
# storeAlgoConfig(algoConfig, folder)
