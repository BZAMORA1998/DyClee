import numpy as np
import os
from sys import exit
import csv

#COGE LOS DATOS DEL ARCHIVO CVD
def getDatasetsFromFolder(resourcesFolder):
    #INTENTE OBTENER UNA LISTA DE TODOS LOS ARCHIVOS DENTRO DE LA CARPETA ESPECIFICADA
    try:
        files = sorted(
            os.listdir(resourcesFolder))  # DEVUELVE UNA LISTA CLASIFICADA CON TODOS LOS NOMBRES DE ARCHIVOS DENTRO DE LA CARPETA ESPECIFICADA
    except BaseException as e:
        print("NO SE PUDO ABRIR LA CARPETA DE RECURSOS: " + str(e))
        exit()
    #PARA GUARDAR JUNTOS TODOS LOS CONJUNTOS DE DATOS BUSCADOS
    datasets = []
    #ITERAR SOBRE LA LISTA DE ARCHIVOS
    for fileFullName in files:
        filePath = resourcesFolder + fileFullName
        # OBTENER EL PARÁMETRO K DEL ENCABEZADO

        with open(filePath, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # OBTIENE LA PRIMERA LÍNEA
        k = header[0]  # 'header' returns ['k',]
        skip_header = 1
        # PARA VER SI 'K' ESTÁ ESPECIFICADO EN EL ENCABEZADO O NO
        if len(k) != 1:
            k = '1'
            # SI NO SE ESPECIFICA 'K', ¡NO OMITA LA PRIMERA FILA!
            skip_header = 0
        # CONSIGUE LA DATA
        ndarray = np.genfromtxt(filePath, delimiter=",", skip_header=skip_header)
        fileNameWithoutExtension = fileFullName.split(".")[0]
        # append info to datasets
        datasets.append({
            'k': int(k),
            'name': fileNameWithoutExtension,
            'dataset': ndarray
        })
    return datasets
