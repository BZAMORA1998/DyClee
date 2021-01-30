#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime

from utils.helpers.custom_math_fxs import manhattanDistance
from utils.micro_clusters.CF import CF
import numpy as np


class MicroCluster:
    
    def __init__(self, hyperboxSizePerFeature, currTimestamp, point):
        self.currTimestamp = currTimestamp # timestamp object
        self.hyperboxSizePerFeature = hyperboxSizePerFeature
        self.CF = self.initializeCF(point)
        self.label = -1 #"unclass"
        self.previousCentroid = []
    


    def __repr__(self):
        return 'Micro Cluster'



    # initializes CF  
    def initializeCF(self, point):  
       # we assume point is a list of features
       LS = point
       # this vector will only have point elements squared
       SS = [a*b for a,b in zip(point, point)]
       now = self.currTimestamp.timestamp
       # CF creation
       cf = CF(n=1, LS=LS, SS=SS, tl=now, ts=now)
       return cf


    
    # Retunrs verdadero si el uc es alcanzable desde un elemento dado
    def isReachableFrom(self, point):
        myCentroid = self.getCentroid()
        maxDiff = float("-inf")
        featureIndex = 0
        # for each feature #   para cada característica
        for i in range(len(point)):
            # diferencia entre la característica del elemento y el centroide del clúster para esa característica
            diff = abs(point[i] - myCentroid[i])
            if diff > maxDiff:
                maxDiff = diff
                featureIndex = i
        # si para la característica de diferencia máxima el elemento no coincide con el clúster, devuelve falso
        if maxDiff >= (self.hyperboxSizePerFeature[i] / 2):
            return False
        # el elemento se ajusta al clúster u
        return True

    # devuelve el centroide del clúster u
    def getCentroid(self):
        centroid = []
        # for each feature
        for i in range(len(self.CF.LS)):
          centroid.append(self.CF.LS[i] / self.CF.n)
        return centroid
    
    
    
    # includes an element into the u cluster
    # updates CF vector
    def addElement(self, point, lambd):
        decayComponent = self.decayComponent(lambd)
        #Concatena los nuevos numeros de registro con lo que ya tenia
        self.updateN(decayComponent, point)#Actualiza el numero de elementos
        self.updateLS(decayComponent, point)#Actualiza la suma lineal
        self.updateSS(decayComponent, point)#Actualiza la suma cuadratica
        self.updateTl()



    def decayComponent(self, lambd):
        dt = self.currTimestamp.timestamp - self.CF.tl
        return 2 ** (-lambd * dt)


        
    def updateTl(self):
        self.CF.tl = self.currTimestamp.timestamp
        
        
        
    def updateN(self, decayComponent, point=None):
        N = self.CF.n * decayComponent
        if point is not None:
            N += 1
        self.CF.n = N

        
    
    def updateLS(self, decayComponent, point=None):
        # forget
        for i in range(len(self.CF.LS)):
            self.CF.LS[i] = self.CF.LS[i] * decayComponent
        # add element
        if point is not None:
            for i in range(len(point)):
                self.CF.LS[i] = self.CF.LS[i] + point[i]
            
            
    
    def updateSS(self, decayComponent, point=None):
        # forget
        for i in range(len(self.CF.SS)):
            self.CF.SS[i] = self.CF.SS[i] * decayComponent
        # add element
        if point is not None:
            for i in range(len(point)):
                self.CF.SS[i] = self.CF.SS[i] + (point[i] **2)

        

    def getD(self):
      V = np.prod(self.hyperboxSizePerFeature)
      return self.CF.n / V

    

    def hasUnclassLabel(self):
      return (self.label is -1)
    

    
    # retunrs true if the microCluster is directly connected to another microCluster
    #devuelve verdadero si el microCluster está conectado  directamente a otro microCluster
    def isDirectlyConnectedWith(self, microCluster, uncommonDimensions):
      featuresCount = len(self.CF.LS)
      currentUncommonDimensions = 0
      myCentroid = self.getCentroid()
      microClusterCentroid = microCluster.getCentroid()
      # for each feature
      for i in range(featuresCount):
          # difference between the u cluster centroids for that feature
          #diferencia entre los centroides del clúster u para esa característica
          aux = abs(myCentroid[i] - microClusterCentroid[i])
          # if for a given feature the element doesn't match the cluster, return false
          #si para una característica determinada el elemento no coincide con el clúster, devuelve falso
          if aux >= self.hyperboxSizePerFeature[i]:
              currentUncommonDimensions += 1
      return currentUncommonDimensions <= uncommonDimensions
        


    def applyDecayComponent(self, lambd):
        decayComponent = self.decayComponent(lambd)
        self.updateN(decayComponent)
        self.updateLS(decayComponent)
        self.updateSS(decayComponent)
        


    def distanceTo(self, microCluster):
        return manhattanDistance(self.getCentroid(), microCluster.getCentroid())


        
        
        
        
        
        
        
        
        
        