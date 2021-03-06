import json

paths = None
algoNames = None
timeSeriesToyDatasetName = None

def _getPaths():
    return paths


def _getAlgoNames():
    return algoNames


def _getTimeSeriesToyDatasetName():
    return timeSeriesToyDatasetName


def _fetchConfig():
    # we use the global key word to being able to change the values of the variables declared outside the function
    #usamos la palabra clave global para poder cambiar los valores de las variables declaradas fuera de la función v
    global paths
    global algoNames
    global timeSeriesToyDatasetName

    configFilePath = "C:/workspace_titulacion/config.json"
    with open(configFilePath) as f:
        data = json.load(f)
    # fill variables
    #llenar variables
    paths = data.get("paths")
    algoNames = data.get("algoNames")
    timeSeriesToyDatasetName = data.get("timeSeriesToyDatasetName")


def _fetchElementIfNull(_getter):
    element = _getter()
    if (element != None):
        return element
    # else
    #más
    _fetchConfig()
    return _getter()


def _getElementFromDict(key, _getter):
    dict = _fetchElementIfNull(_getter)
    return dict.get(key)


def getClusteringResultsPath():
    return _getElementFromDict(key="clusteringResultsPath", _getter=_getPaths)


def getTimeSeriesToyDatasetName():
    return _fetchElementIfNull(_getTimeSeriesToyDatasetName)


def getTimeSeriesDatasetsPath():
    return _getElementFromDict(key="timeSeriesDatasetsPath", _getter=_getPaths)



#OBTIENE LOS  DATOS DEL ARCHIVO DE EXCEL
def getNonTimeSeriesDatasetsPath():
    return _getElementFromDict(key="nonTimeSeriesDatasetsPath", _getter=_getPaths)

#OBTIENE LOS  DATOS DEL ARCHIVO DE EXCEL RESUULLTADO
def getClusteringResultsPath():
    return _getElementFromDict(key="clusteringResultsPath", _getter=_getPaths)


def getDycleeName():
    return _getElementFromDict(key="dyclee", _getter=_getAlgoNames)


