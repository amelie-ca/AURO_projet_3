import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes, PlotRobotMap
from simulation import LMCreation, GenerateRobotPosition, GenerateRobotMeasurment

"""
Funtion for the initialisation of particles. The robot pose is initialise in 0,0,0, and the landmark position in initialise acording to a gaussian law
Input : 
    Part0 : The dict list for the particles 
    N : Number of particles
    
"""
def InitPF(N : int, Mamer : np.ndarray,  dispAmers : float) :
    Parts0 = [dict() for i in range(N)]

    for i in range(N) :
        amers = Mamer + np.linalg.cholesky(np.diag(dispAmers*np.ones(Mamer.shape[0])))@np.random.normal(size=(Mamer.shape[0],))
        Parts0[i] = {"RobPos" : [0,0,0], "Amers" : amers, "W" : 1/N}
    
    return Parts0

if __name__ == '__main__':
    #Donnees 
    Nbamer = 6
    distX = distY = 2
    xA0, yA0 = (1,1)
    dispAmers = 0.01
    tau = 0.3
    xR0, yR0 = (0,0)

    covDis = 0.0001
    covAng = np.pi/9500
    covDis0 = 0.000001
    covAng0 = np.pi/10000
    covDisMes = 0.01
    covAngMes = np.pi/20

    NbPart = 10
    
    Mamer, amers = LMCreation(Nbamer, distX, distY, xA0, yA0, dispAmers)
    U, Xreel, Nb1, Nb2, Nb3, PX0, Qw1, Qw2, Qw3 = GenerateRobotPosition(xR0, yR0, tau, covDis, covAng, covDis0, covAng0)
    N = Nb1+Nb2+Nb3+1
    Zr = GenerateRobotMeasurment (N, Nbamer, Xreel, amers, covAng, covDis)
    PlotRobotMap(Xreel, amers, "Test")
    plt.show()

    #Filtrage 
    #Initialisation 
    Part = [[dict() for x in range(NbPart)] for y in range(N)]
    Part[0] = InitPF(NbPart, Mamer, dispAmers)
    
    #