# S1
from utils.helpers.custom_math_fxs import manhattanDistance, stddev
from utils.micro_clusters.micro_cluster import MicroCluster
from utils.timestamp import Timestamp
from math import log10
# S2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from utils.helpers.custom_printing_fxs import printInMagenta
import matplotlib
# set matplotlib backend to Qt5Agg to make figure window maximizer work
##configure el backend de matplotlib en Qt5Agg para hacer que el maximizador de ventana de figura funcione
##configure el backend de matplotlib en Qt5Agg para hacer que el maximizador de ventana de figura funcione
matplotlib.use('Qt5Agg')


class valorCluster:
    x = 0.0
    y = 0.0
    color = ""

class color:
    color=""

class cluster:
    x = 0.0
    y = 0.0
    xcluster = 0.0
    ycluster = 0.0

class Dyclee:
    def __init__(self, dataContext, relativeSize = 0.6, speed = float("inf"), uncommonDimensions = 0, lambd = 0, periodicRemovalAt = float("inf"),
                 periodicUpdateAt = float("inf"), timeWindow = 5, findNotDirectlyConnButCloseMicroClusters = True,
                 closenessThreshold = 1.5):
        # hyper parameters
        #hiperparámetros
        self.relativeSize = relativeSize
        self.uncommonDimensions = uncommonDimensions
        self.processingSpeed = speed
        self.lambd = lambd
        self.periodicUpdateAt = periodicUpdateAt
        self.timeWindow = timeWindow
        self.periodicRemovalAt = periodicRemovalAt
        self.findNotDirectlyConnButCloseMicroClusters = findNotDirectlyConnButCloseMicroClusters
        self.closenessThreshold = closenessThreshold
        self.dataContext = dataContext # must be a bounding box instance  #debe ser una instancia de cuadro delimitador
        # define hyperboxSizePerFeature
        #definir hyperboxSizePerFeature
        self.hyperboxSizePerFeature = self.getHyperboxSizePerFeature()
        # internal vis //vis interna
        self.aList = []
        self.oList = []
        self.processedElements = 0
        self.timestamp = 0
        self.currTimestamp = Timestamp() # initialized at 0  //inicializado en 0
        self.densityMean = 0
        self.densityMedian = 0

#RETORNA UN JJASON
    def getConfig(self):
        return {
            "relativeSize": self.relativeSize,
            "uncommonDimensions" : self.uncommonDimensions,
            "speed": self.processingSpeed,
            "lambd": self.lambd,
            "periodicUpdateAt": self.periodicUpdateAt,
            "timeWindow": self.timeWindow,
            "periodicRemovalAt": self.periodicRemovalAt,
            "closenessThreshold": self.closenessThreshold,
            "dataContext": self.dataContextAsStr(),
        }

#INVOCA LAS FUNCIONES Y LAS COMPARA Y TAMBIEN CREA EL CONTADOR DATACONTEXT Y MULTIPLICA EL RELATIVESIZE =0.06 CON EL AUXILIR QUE EL VALOR ES 4 Y EL VALOR QUE ARROJA ES 0.24 PARA ESTO SIRVE ESTA FUNCION
    def getHyperboxSizePerFeature(self):
        hyperboxSizePerFeature = []
        for context in self.dataContext:
            aux = context.maximun - context.minimun
            hyperboxSizePerFeature.append(self.relativeSize * abs(aux))
        return hyperboxSizePerFeature


#ESTA FUNCION NOS ARROJA DATOS PERO SI SE LA COMENTA DA ERROR EN EL CODIGO
    def dataContextAsStr(self):
        aux=""
        for context in self.dataContext:
          #print("hola",context)
          aux += str(context.minimun) + "<" + str(context.maximun) + " | "
          print("hello",aux)
        return aux


    # S1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!

#EN ESTA LINEA DE CODIGO SE INGRESAN LOS DATOS QUE VIENEN DEL ARCHIVO CSV Y LOS CONVIERTE EN UN NUEVO ELEMENTO
    # returns a list of floats given an iterable object
    #devuelve una lista de flotantes dado un objeto iterable
    def trainOnElement(self, newEl,ac):
        # get an object matching the desired format (to guarantee consistency)
        #obtener un objeto que coincida con el formato deseado (para garantizar la coherencia)
        #ESTE PUNTO ES DONDE SE GUARDAR LOS NUEVOS ELEMENTOS QUE SE INGRESAN Y EL PRIMER VALOR SALE X=0.92,X=0.98
        point = self.getListOfFloatsFromIterable(newEl)
       # print("bb",point)
        self.processedElements += 1
        #este elemento es un contador del 1 al 500
        # control the stream speed /controlar la velocidad del flujo
        # TODO: check if the "speed" param is ok ...

        #No aplica xq no se va a trabajar con marca de tiempo
        if self.timeToIncTimestamp():
           self.timestamp += 1
           self.currTimestamp.timestamp = self.timestamp
        # now, check what to do with the new point
        #ahora, compruebe qué hacer con el nuevo punto

        self.processPoint(point,ac)
       # print(ac,"hello",point)

        # No aplica xq no se va a trabajar con marca de tiempo
        #ESTA FUNCION PROCESA DATOS EN LA LINEA 145
        if self.timeToCheckMicroClustersTl():
        #ESTA FUNCION PROCESA DATOS EN LA LINEA 142
            self.checkMicroClustersTl()
        # periodic cluster removal //eliminación periódica de racimos
        # No aplica xq no se va a trabajar con marca de tiempo
        if self.timeToPerformPeriodicClusterRemoval():
            #EN ESTA LINEA DE CODIGO ES DONDE ENVIA LOS DATOS PROCESADOS AL MAIN
         self.performPeriodicClusterRemoval()



#ESTE METODO SI LO UTILIZA Y LO UTILIZA EN LA LINEA 102 PARA GUARADAR EL NUEVO ELEMENTO
    def getListOfFloatsFromIterable(self, newEl):
            point = []
            for value in newEl:
               point.append(float(value))
            return point

#LO COMENTE PORQUE NO LO UTILIZA
    def timeToIncTimestamp(self):
       return self.processedElements % self.processingSpeed == 0

#SI UTILIZA ESTE METODO EN LA LINEA 122
    def timeToPerformPeriodicClusterRemoval(self):
        return self.processedElements % (self.periodicRemovalAt * self.processingSpeed) == 0

#SI LO UTILIZA EN LA LINEA 119
    def timeToCheckMicroClustersTl(self):
       return self.processedElements % (self.periodicUpdateAt * self.processingSpeed) == 0

    global listaMicrocluster
    listaMicrocluster = list()
    con = 0
    def processPoint(self, point,ac):
        #SUPUESTO: el punto es una lista de flotantes
        # find reachable u clusters for the new element
        #encontrar clústeres u accesibles para el nuevo elemento
        reachableMicroClusters = self.findReachableMicroClusters(point)
       # print(reachableMicroClusters)
        #AQUI EN ESTE PRINT ME IMPRIME ESTO O SI NO ESTO[MICROCLUSTER]-----[]
        #Si no tiene un cluster alcanzable el mismo crea su propio micro clustering

        #Solo entraria si llega el primer registro de la base de datos
        if not reachableMicroClusters:
            # empty list -> create u cluster from element
            #lista vacía -> crear u cluster a partir del elemento
            # the microCluster will have the parametrized relative size, and the Timestamp object to being able to access the
            # current timestamp any atime
            #el microCluster tendrá el tamaño relativo parametrizado y el objeto Timestamp para poder acceder al
             # marca de tiempo actual en cualquier momento
            microCluster = MicroCluster(self.hyperboxSizePerFeature, self.currTimestamp, point)
            self.oList.append(microCluster)
            self.con=self.con+1;
            #print(self.con, "micro",microCluster.getCentroid(),point)
           # print(self.con,"micro",self.oList,microCluster)

           # print(self.con, ". Microcluster: ", microCluster.getCentroid())
           # print(self.con,". Microcluster: ",microCluster.getCentroid(), "valor: ",microCluster.CF.LS)
           # a=cluster()
            #a.x(point[0])
            #a.x(point[1])
            #a.xcluster(microCluster.CF.LS[0])
            #a.ycluster(microCluster.CF.LS[1])
            #listaMicrocluster.append(a)
#EN ESTE ES ES DONDE PASAN TODOS LOS MICROCLUSTER CREADOS
        else:
            # find closest reachable u cluster
            #encontrar el clúster u accesible más cercano
            closestMicroCluster = self.findClosestReachableMicroCluster(point, reachableMicroClusters)
            closestMicroCluster.addElement(point=point, lambd=self.lambd)
            self.con = self.con + 1;
            print(self.con,closestMicroCluster.CF.n,closestMicroCluster.CF.data)
            #print(self.con, "Microcluster: ",closestMicroCluster.getCentroid())

        # at this point, self self.aList and self.oList are updated
        #en este punto, self self.aList y self.oList se actualizan

    #def checkMicroClustersTl(self):
    #    microClusters = self.aList + self.oList
   #     for micCluster in microClusters:
  #          print("hello",microClusters)
 #           if (self.timestamp - micCluster.CF.tl) > self.timeWindow:
#                micCluster.applyDecayComponent(self.lambd)


    #def performPeriodicClusterRemoval(self):
        # if the density of an outlier micro cluster drops below the low density threshold, it is eliminated
        #si la densidad de un micro cluster atípico cae por debajo del umbral de baja densidad, se elimina
        # we will only keep the micro clusters that fulfil the density requirements
        #solo mantendremos los micro clústeres que cumplan con los requisitos de densidad
        #newOList = []
        #for oMicroCluster in self.oList:
           # if oMicroCluster.getD() >= self.getDensityThershold():
               # newOList.append(oMicroCluster)
          #      print("chula",newOList)
            # do not penalize emerging concepts! A micro cluster must not be 'dense' but, if it is growing, let it grow!
            ##¡No penalices conceptos emergentes! Un micro racimo no debe ser 'denso' pero, si está creciendo, ¡déjalo crecer!
         #   elif (self.timestamp - oMicroCluster.CF.tl) < self.timeWindow:
        #        print(oMicroCluster.CF.tl)
       #         newOList.append(oMicroCluster) # we keep the micro cluster!//Nosotras mantenemos el micro cluster!
               # print(newOList)

        # at this point micro clusters which are below the density requirement were discarded
        #en este punto, se descartaron los micro racimos que están por debajo del requisito de densidad
        #print(self.oList)
      #  self.oList = newOList
        #print(self.oList)

   # def getDensityThershold(self):
    #    dMean = self.calculateMeanFor(self.oList)

     #   return dMean


   # def calculateMeanAndSD(self, dataset):
    #def calculateMeanAndSD (self, conjunto de datos):
    #     n = len(dataset)
    #     # sample taken to get the ammount of features
    #     anElement = dataset[0]
    #     # for each feature
    #     for fIndex in range(len(anElement)):
    #         acPerFeature = 0
    #         fValuesList = []
    #         # for each element in dataset
    #         for i in range(n):
    #             el = dataset[i]
    #             fValue = el[fIndex]
    #             acPerFeature += fValue
    #             # to later obtain ssdev
    #             fValuesList.append(fValue)
    #         featureMean = acPerFeature / n
    #         self.meanList.append(featureMean)
    #         featureSD = stddev(data=fValuesList, mean=featureMean)
    #         self.SDList.append(featureSD)

    cu=0
    # devuelve una lista de clústeres u accesibles para un elemento dado
    #AQUI ES DONDE LOS DATOS SON INGRESADOS A OLIST DEBIDO QUE SON LOS VALORES QUE NO PERTENENCEN A LOS MICROCLUSTER
    def findReachableMicroClusters(self, point):
        reachableMicroClusters = self.getReachableMicroClustersFrom(self.aList, point)
        if not reachableMicroClusters:
            # empty list -> check oList //lista vacía -> comprobar oList
            self.cu=self.cu+1
            #print(self.cu,"datos",self.oList)
           # print(self.cu,point)
            reachableMicroClusters = self.getReachableMicroClustersFrom(self.oList, point)
            #print(self.cu,point)
            #print("lista",reachableMicroClusters)
        return reachableMicroClusters


    # modifies reareachableMicroClusters iterating over a given list of u clusters
    #modifica los MicroClusters alcanzables que iteran sobre una lista dada de u clústeres
    a=0
    def getReachableMicroClustersFrom(self, microClustersList, point):
        res = []
        for microCluster in microClustersList:
            # the microCluster has the parametrized relative size
            #El microCluster tiene el tamaño relativo parametrizado
            if microCluster.isReachableFrom(point):
                res.append(microCluster)
                self.a=self.a+1
                #print(microCluster.__init__())
               # print(self.a,microCluster.getCentroid())
        return res

    co=0
    # returns the closest microCluster for an element, given a set of reachable microClusters
    #devuelve el microCluster más cercano para un elemento, dado un conjunto de microClusters alcanzables
    #EN ESTA FUNCION ES DONDE YA NO APARECEN LOS 103 DATOS Y SOLO NOS QUEDAN LOS 397 DATOS OJO
    def findClosestReachableMicroCluster(self, point, reachableMicroClusters):
        self.co=self.co+1
        closestMicroCluster = None
        minDistance = float("inf")
        for microCluster in reachableMicroClusters:
            #ARROJA VARIABLES COMO (MICROCLUSTER)
            #print(microCluster,reachableMicroClusters)
            distance = manhattanDistance(point, microCluster.getCentroid())
           # print(self.co,microCluster.getCentroid())
            #print(point)
            #print(self.co,"PUNTO",point,"MICRO",microCluster.getCentroid())
           # print(microCluster.previousCentroid)
            if distance < minDistance:
                minDistance = distance
                closestMicroCluster = microCluster
                #print(point)
                #print(self.co,"mini",minDistance,"distancia",distance,"centroide",microCluster.getCentroid())
                #print(microCluster.__repr__())
        return closestMicroCluster

#print(microCluster.__dict__)ESTA FUNCION NOS MUESTRA TODOS ESOS DATOS
#microcluster.label nos arroja datos -1
    #microcluster CF.ss,tl,ts,sl arroja datos que no nos sirven para la ejecucion del programa
#microcluster CurrTimestap nos arroja <utils.timestamp.Timestamp object at 0x0170A4F0>
    # microcluster.hyperboxSizePerFeature arroja [0.24 0.24]
    # microcluster.label nos arroja datos -1
    # microcluster.label nos arroja datos -1
    # microcluster.label nos arroja datos -1
    # microcluster.label nos arroja datos -1
    # microcluster.label nos arroja datos -1

# S2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ESTA FUNCION ACTUALIZA LOS VALORES EN EL GRAFICO
    def getClusteringResult(self):
        # update density mean and median values with current ones
        # Actualizar los valores medios y medianos de densidad con los actuales
        self.calculateDensityMeanAndMedian()
       # print(self.calculateDensityMeanAndMedian())
        # rearrange lists according to microClusters density, considering density mean and median limits
        #reorganizar las listas de acuerdo con la densidad de microClusters,
        # considerando los límites de la media y la mediana de la densidad
        self.rearrangeLists()
        #print(self.rearrangeLists())
        # form final clusters //formar grupos finales
        self.formClusters()
        #print(self.formClusters())
        # concatenate them: get both active and outlier microClusters together
        #concatenarlos: juntar microclústeres activos y atípicos
        #AQUI EN ESTE MICROCLUSTER SE MUESTRAN LOS 103 DATOS QUE SE GUARDAN EN EL ARCHIVO CSV.
        microClusters = self.aList + self.oList
        #print(microClusters)
        # extract dense microClusters from active list
        #extraer microclústeres densos de la lista activa

        #DMC = self.findDenseMicroClusters()

        #DMC = self.findDenseMicroClusters ()
        #AQUI EN DMC SE GUARDAN SOLO LOS VALORES NORMALES Y SE DESCARTA LOS VALORES ATIPICOS OSEA SOLO 61 DATOS  Y EN EL OLIST SE GUARDAN LOS 41 VALORES RESTANTES
        DMC = self.aList
        #print(DMC)
        # plot current state and micro cluster evolution
        #trazar el estado actual y la evolución del micro cluster
        self.plotClusters(microClusters, DMC)
        # update prev state once the evolution was plotted
        #actualizar el estado anterior una vez que se trazó la evolución
        self.updateMicroClustersPrevCentroid(microClusters, DMC)
        # send updated microClusters lists to s1 (needs to be done at this point to make prev state last; labels will last too)
        #enviar listas de microClusters actualizadas a s1 (debe hacerse en este punto para que el estado anterior dure; las etiquetas también durarán)
        # TODO: store clustering result -> microClusters
        return microClusters


    def rearrangeLists(self,):
        newAList = []
        newOList = []
        concatenatedLists = self.aList + self.oList
        for microCluster in concatenatedLists:
            if self.isOutlier(microCluster):
                newOList.append(microCluster)
            else:
                # microCluster is dense or semi dense
                #MicroCluster es denso o semi denso
                newAList.append(microCluster)
        self.aList = newAList
        self.oList = newOList


    def calculateDensityMeanAndMedian(self):
        concatenatedLists = self.aList + self.oList

        #Calcula la densidad media
        self.densityMean = self.calculateMeanFor(concatenatedLists)

        #Calcula la densidad mediana
        self.densityMedian = self.calculateMedianFor(concatenatedLists)


    def resetLabelsAsUnclass(self, microClusters):
        for microCluster in microClusters:
            microCluster.label = -1


    def calculateMeanFor(self, microClusters):
        return np.mean([microCluster.getD() for microCluster in microClusters])


    def calculateMedianFor(self, microClusters):
        return np.median([microCluster.getD() for microCluster in microClusters])


    # returns true if a given u cluster is considered dense
    #devuelve verdadero si un clúster u dado se considera denso
    def isDense(self, microCluster):
        return (microCluster.getD() >= self.densityMean and microCluster.getD() >= self.densityMedian)


    # returns true if a given u cluster is considered semi dense
    #devuelve verdadero si un clúster u dado se considera semi denso
    def isSemiDense(self, microCluster):
        # xor
        return (microCluster.getD() >= self.densityMean) != (microCluster.getD() >= self.densityMedian)


    # returns true if a given u cluster is considered outlier
    #devuelve verdadero si un clúster u dado se considera atípico
    def isOutlier(self, microCluster):
        return (microCluster.getD() < self.densityMean and microCluster.getD() < self.densityMedian)


    # returns only dense u clusters from a set of u clusters
    #devuelve solo clústeres u densos de un conjunto de clústeres u
    def findDenseMicroClusters(self):
        # it's unnecessary to look for dense microClusters in the oList
        #No es necesario buscar microclústeres densos en oList
        return [microCluster for microCluster in self.aList if self.isDense(microCluster)]


    def getAvgDistToMicroClustersFor(self, microCluster, microClusters):
        sum = 0
        dists = []
        for mc in microClusters:
            dist = microCluster.distanceTo(mc)
            sum += dist
            dists.append(dist)
        return sum/len(microClusters), dists


    def findSimilarMicroClustersFor(self, microCluster, microClusters):
        directlyConn =  self.findDirectlyConnectedMicroClustersFor(microCluster, microClusters)
        if not self.findNotDirectlyConnButCloseMicroClusters:
            return directlyConn
        else:
            notDirectlyConnButClose = self.findCloseMicroClustersFor(microCluster, microClusters)
           # print("l",microCluster)
            return directlyConn + notDirectlyConnButClose

    cont = 0

    def findCloseMicroClustersFor(self, microCluster, microClusters):
        stddevProportion = self.closenessThreshold
        # for encompassing more micro clusters
        #para abarcar más micro clústeres
        avgDistToAllMicroClusters, distances = self.getAvgDistToMicroClustersFor(microCluster, microClusters)
        stdev = stddev(distances, avgDistToAllMicroClusters)
        limit = avgDistToAllMicroClusters - (stdev * stddevProportion)
        # the set of close micro clusters which will be used to expand a macro one
        #el conjunto de microclústeres cercanos que se utilizarán para expandir uno macro
        res = []
        self.cont = self.cont + 1

        for mc in microClusters:
            mcIsClose = microCluster.distanceTo(mc) < limit
           # print(microCluster.CF.data)
            if not microCluster.isDirectlyConnectedWith(mc, self.uncommonDimensions) and mcIsClose:
                res.append(mc)
            # TO DEBUG
            #DEPURAR
            else:

                #print("dist promedio", avgDistToAllMicroClusters)
                #print("stdev", stdev)
                #print("límite", limit)
                #print(self.cont,"microcluster",microCluster.getCentroid())
                #print("yo", microCluster.getCentroid(), "el", mc.getCentroid())
                print("\n")
        return res


    def findDirectlyConnectedMicroClustersFor(self, microCluster, microClusters):
        res = []
        for mc in microClusters:
            if microCluster.isDirectlyConnectedWith(mc, self.uncommonDimensions):
                res.append(mc)
        return res


    def formClusters(self):
        # init currentClusterId// init ID de clúster actual
        currentClusterId = 0
        # reset microClusters labels as -1
        ##restablecer las etiquetas de los microclústeres como -1
        self.resetLabelsAsUnclass(self.aList)
        self.resetLabelsAsUnclass(self.oList)
        # start clustering
        #empezar a agrupar
        alreadySeen = []
        for denseMicroCluster in self.aList:
            if denseMicroCluster not in alreadySeen:
                alreadySeen.append(denseMicroCluster)
                currentClusterId += 1
                denseMicroCluster.label = currentClusterId
                connectedMicroClusters = self.findSimilarMicroClustersFor(denseMicroCluster, self.aList)
                self.growCluster(currentClusterId, alreadySeen, connectedMicroClusters, self.aList)


    # for loop finished -> clusters were formed
    #para bucle terminado -> se formaron grupos
    def growCluster(self, currentClusterId, alreadySeen, connectedMicroClusters, microClusters):
        i = 0
        while i < len(connectedMicroClusters):
            conMicroCluster = connectedMicroClusters[i]
            if (conMicroCluster not in alreadySeen):
                conMicroCluster.label = currentClusterId
                alreadySeen.append(conMicroCluster)
                # FIXME: the following if is redundant bc DMC is equals to the aList
                if self.isDense(conMicroCluster) or self.isSemiDense(conMicroCluster):
                    newConnectedMicroClusters = self.findSimilarMicroClustersFor(conMicroCluster, microClusters)
                    for newNeighbour in newConnectedMicroClusters:
                        connectedMicroClusters.append(newNeighbour)
            i += 1


    def plotClusters(self, microClusters, DMC):
        if not self.plottableMicroClusters(microClusters):
            return
        # let's plot!
        #vamos a trazar!
        # creates a figure with one row and two columns
        # crea una figura con una fila y dos columnas
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        self.plotCurrentClustering(ax1, microClusters)
        self.plotMicroClustersEvolution(ax2, DMC)
        self.plotMicroClustersSize(ax3, microClusters)

        cont=0
        cont1=0
        for col in listColor:
            cont=cont+1
            for li in lista:
                if col.color == li.color:
                    cont1 =cont1+1
                    printInMagenta(str(cont1)+". Cluster "+str(cont)+" : X:"+str(li.x)+" || Y: "+str(li.y))

        # show both subplots
        #mostrar ambas subtramas
        f.canvas.manager.window.showMaximized()
        plt.show()


    def getMarkersSizeList(self, microClusters):
        res = []
        for microCluster in microClusters:
            if self.isOutlier(microCluster):
                # really small size -> comes out almost as a point
                #tamaño realmente pequeño -> sale casi como un punto
                res.append(5)
            elif self.isSemiDense(microCluster):
                # big marker
                #gran marcador
                res.append(20)
            elif self.isDense(microCluster):
                # medium size marker
                #marcador de tamaño mediano
                res.append(50)
        return res


    def plotCurrentClustering(self, ax1, microClusters):
        # first set markers size to represent different densities
        ##Primero establezca el tamaño de los marcadores para representar diferentes densidades
        s = self.getMarkersSizeList(microClusters)
        # then get a list with u cluster labels
        #luego obtén una lista con etiquetas de clúster u
        labels = [microCluster.label for microCluster in microClusters]
        # clusters will be a sequence of numbers (cluster number or -1) for each point in the dataset
        #los grupos serán una secuencia de números (número de grupo o -1) para cada punto del conjunto de datos
        labelsAsNpArray = np.array(labels)
        # get microClusters centroids
        #obtener centroides de microClusters
        centroids = [microCluster.getCentroid() for microCluster in microClusters]

        x, y = zip(*centroids)
        # show info to user
        #Mostrar información al usuario
        self.showClusteringInfo(labelsPerUCluster=labels, clusters=labelsAsNpArray, x=x, y=y)
        # scatter'
        #dispersión'
        ax1.scatter(x, y, c=labelsAsNpArray, cmap="nipy_spectral", marker='s', alpha=0.8, s=s)
        # add general style to subplot n°1
        #agregar estilo general a la subtrama n ° 1
        self.addStyleToSubplot(ax1,
                               title='ESTADOS ACTUALES\nCuadrado lrg = microcluster denso \ncuadrado medio = microcluster semidenso\nSml cuadrado = microcluster atípico')


    def plotMicroClustersEvolution(self, ax2, DMC):
        (DMCwPrevState, newDMC) = self.formMicroClustersEvolutionLists(DMC)
        for denseMicroClusterWPrevSt in DMCwPrevState:
            # an arrow will be drawn to represent the evolution in the centroid location for a dense micro cluster
            # se dibujará una flecha para representar la evolución en la ubicación del centroide para un micro cluster denso
            ax2.annotate("", xy=denseMicroClusterWPrevSt.previousCentroid, xytext=denseMicroClusterWPrevSt.getCentroid(),
                         arrowprops=dict(arrowstyle='<-'))
        # get newDMC centroids
        #obtener nuevos centroides DMC
        if len(newDMC) is not 0:
            centroids = [microCluster.getCentroid() for microCluster in newDMC]
            x, y = zip(*centroids)
            ax2.plot(x, y, ".", alpha=0.5, )

            cont1=0
            for x1 in x:
                printInMagenta("Densa Microcluster: Valor Y: " + x1.__repr__()+ " Valor: "+y[cont1].__repr__())
                cont1=cont1+1
        # add general style to subplot n°2
        ## agregar estilo general a la subtrama n ° 2
        self.addStyleToSubplot(ax2, title='EVOLUCIÓN DENSA DE MICRO CLÚSTERS\n"." significa que no hay cambio \n"->" implica evolución')

    global lista
    global listColor
    lista=list()
    listColor=list()

    def plotMicroClustersSize(self, ax3, microClusters):
        # choose palette
        ## elegir paleta
        ns = plt.get_cmap('nipy_spectral')
        # get labels
        ## obtener etiquetas
        labels = [microCluster.label for microCluster in microClusters]

        # skip repeated leabels
        # omitir etiquetas repetidas
        s = set(labels)
        # especify normalization to get the correct colors
        # especificar la normalización para obtener los colores correctos
        norm = clrs.Normalize(vmin=min(s), vmax=max(s))
        # for every micro cluster
        ## para cada micro cluster
        for microCluster in microClusters:

            # get coordinate x from microCluster centroid
            ## obtener la coordenada x del centroide del microCluster
            realX = microCluster.getCentroid()[0]
            # get coordinate y from microCluster centroid
            # obtener la coordenada y del centroide del microCluster
            realY = microCluster.getCentroid()[1]
            # x n y are the bottom left coordinates for the rectangle
            ##x n y son las coordenadas de la parte inferior izquierda del rectángulo
            # to obtain them we have to substract half the hyperbox size to both coordinates
            #para obtenerlos tenemos que restar la mitad del tamaño del hipercuadro a ambas coordenadas
            offsetX = microCluster.hyperboxSizePerFeature[0] / 2
            offsetY = microCluster.hyperboxSizePerFeature[1] / 2
            x = realX - offsetX
            y = realY - offsetY

            # the following are represented from the bottom left angle coordinates of the rectangle
            ## lo siguiente se representa a partir de las coordenadas del ángulo inferior izquierdo del rectángulo
            width = microCluster.hyperboxSizePerFeature[0]
            height = microCluster.hyperboxSizePerFeature[1]
            # get the color
            ## consigue el color
            c = ns(norm(microCluster.label))
            # make the rectangle //# haz el rectángulo
            rect = plt.Rectangle((x, y), width, height, color=c, alpha=0.5)
            ax3.add_patch(rect)



            # printInMagenta("Valor X: " + x.__repr__() + " Color: "+c.__repr__())
            # printInMagenta("Valor Y: " + y.__repr__()+ " Color: "+c.__repr__())
            a=valorCluster();
            a.x=float(realX.__repr__())
            a.y=float(realY.__repr__())
            a.color=c.__repr__()
            lista.append(a)

            if len(listColor)==0:
                b=color()
                b.color=c.__repr__()
                listColor.append(b)
            else :
                cont=0
                for col in listColor:
                    if col.color == c.__repr__():
                        cont=cont+1;

                if cont==0 :
                    b = color()
                    b.color = c.__repr__()
                    listColor.append(b)

            #printInMagenta("Valor X: "+str(a.x)+" || Valor Y:"+str(a.y) +" || Color: "+a.color)

            # plot the rectangle center (microCluster centroid)
            ## trazar el centro del rectángulo (centroide microCluster)
            ax3.plot(realX, realY, ".", color=c, alpha=0.3)
        self.addStyleToSubplot(ax3, title='MICRO CLUSTERS TAMAÑO REAL')


    def formMicroClustersEvolutionLists(self, DMC):
        DMCwPrevState = []
        newDMC = []
        for denseMicroCluster in DMC:
            if (len(denseMicroCluster.previousCentroid) is 0) or (denseMicroCluster.getCentroid() == denseMicroCluster.previousCentroid):
                # dense microCluster hasn't previous state --> is a new dense microCluster
                #microCluster denso no tiene estado anterior -> es un nuevo microCluster denso
                # dense microCluster prev state and current centroid match --> dense microCluster hasn't changed nor evolutioned; just mark its position
                #el estado anterior del microCluster denso y el centroide actual coinciden -> el microCluster denso no ha cambiado ni evolucionado; solo marca su posición
                newDMC.append(denseMicroCluster)
            else:
                # dense microCluster has previous state --> dense microCluster has evolutioned
                # microCluster denso tiene estado anterior -> microCluster denso ha evolucionado
                DMCwPrevState.append(denseMicroCluster)
        return (DMCwPrevState, newDMC)


    def updateMicroClustersPrevCentroid(self, microClusters, DMC):
        for microCluster in microClusters:
            if microCluster not in DMC:
                # microCluster prev state doesn't matter; if a dense microCluster ended up being an outlier, its position is no longer important
                ## El estado anterior del microCluster no importa; si un microCluster denso terminó siendo un valor atípico, su posición ya no es importante
                microCluster.previousCentroid = []
            else:
                # microCluster is dense; current state must be saved for viewing future evolution
                # microCluster es denso; el estado actual debe guardarse para ver la evolución futura
                microCluster.previousCentroid = microCluster.getCentroid()


    # Trabaja con graficos para visualizarlo al usuario
    def addStyleToSubplot(self, ax, title=''):
        # set title//establecer título
        ax.set_title(title)
        # set axes limits//establecer límites de ejes
        minAndMaxDeviations = [-2.5, 2.5]
        ax.set_xlim(minAndMaxDeviations)
        ax.set_ylim(minAndMaxDeviations)
        # set plot general characteristics
        #establecer las características generales de la parcela
        ax.set_xlabel("Característica 1")
        ax.set_ylabel("Característica 2")
        ax.grid(color='k', linestyle=':', linewidth=1)


    def showClusteringInfo(self, labelsPerUCluster, clusters, x, y):
        # show final clustering info
        ## mostrar información de agrupamiento final
        dic = self.clustersElCounter(labelsPerUCluster)
        dicLength = len(dic)
        if dicLength == 1:
            msg = "Solo hay 1 grupo final y"
        else:
            msg = "Hay " + len(dic).__repr__() + " grupos finales y "
        if -1 in labelsPerUCluster:
            msg += "uno de ellos representa valores atípicos (el negro)."
        else:
            msg += "sin valores atípicos."
        printInMagenta(msg + "\n")
        for key, value in dic.items():
            printInMagenta("- Cluster n°" + key.__repr__() + " -> " + value.__repr__() + " microClusters" + "\n")
        # show detailed info regarding lists of microClusters coordinates and labels
        ## mostrar información detallada sobre listas de coordenadas y etiquetas de microClusters

        printInMagenta("* microClusters etiquetas: " + '\n' + clusters.__repr__() + '\n')

        printInMagenta("* microClusters etiquetas: " + '\n' + clusters.__repr__() + '\n')
        printInMagenta("* microClusters 'x' coordenadas: " + '\n' + x.__repr__() + '\n')
        printInMagenta("* microClusters 'y' coordenadas: " + '\n' + y.__repr__())


    # returns a dictionary in which every position represents a cluster and every value is the amount of microClusters w that label
    ## devuelve un diccionario en el que cada posición representa un clúster y cada valor es la cantidad de microClusters con esa etiqueta
    def clustersElCounter(self, labelsPerUCluster):
        dicKeys = set(labelsPerUCluster)
        dic = {key: 0 for key in dicKeys}
        for c in labelsPerUCluster:
            dic[c] += 1
        return dic


    # returns True if microClusters are plottable (regarding amount of features)
    ## devuelve True si los microClusters son trazables (con respecto a la cantidad de características)
    def plottableMicroClusters(self, microClusters):
        cl=0
        for microCluster in microClusters:
            cl=cl+1
            for data in microCluster.CF.data:
                print(str(cl)+". Microcluster: "+str(microCluster.getCentroid())+" - Valor de data: "+str(len(microCluster.CF.data))+" - data: "+str(data))

        if len(microClusters) == 0:
            # there are't any u clusters to plot
            ## no hay clusters de u para trazar
            print("UNABLE TO PLOT CLUSTERS: there are no micro clusters")
            return False
        firstEl = microClusters[0]
        if len(firstEl.CF.LS) != 2:
            print("UNABLE TO PLOT CLUSTERS: it's not a 2D data set")
            return False
        # microClusters are plottable//# microClusters son trazables
        return True