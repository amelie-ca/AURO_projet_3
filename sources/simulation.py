import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes, PlotRobotMap

"""
Function for the generation of landmark position. The landmarks are disposed in 2 lines, the landmark position is generated acording to a gaussian law, with the same varience on both axis
Inuput : 
    nbamer, int : number of landmark
    distX, float : the distence in X between 2 landmaks on the same line
    distY, float : the distence in Y between the 2 lines of landmarks
    xA0, float : the position of the first landmark in X
    yA0, float : the position of the first landmark in Y
    dispA, float : the dispertion of the landmark around the ideal position, this dispertion is used for X and Y axis
Output : 
    amers : ideal landmarks position 
    amersB : real landmarks position, with noise
"""
def LMCreation(nbamer : int, distX : float, distY : float, xA0 : float, yA0 : float, dispA : float) -> np.ndarray :
    print("... Creation de la Carte ...")
    amers = np.empty((2*nbamer,))
    amers[0:2] = (xA0, yA0)
    for i in range(2, nbamer, 2) :
        amers[i] = amers[i-2]+distX
        amers[i+1] = yA0
    for i in range(nbamer, 2*nbamer, 2) :
        amers[i] = amers[i-nbamer]
        amers[i+1] = yA0+distY
    sigma = np.diag(dispA*np.ones(2*nbamer))
    amersB = amers + np.linalg.cholesky(sigma)@np.random.normal(size=(2*nbamer,))
    return amers, amersB

"""
Function for the generation of the command vector and robot state during the simulation. The trajectory of the robot is a straight line, a U-turn, then a straight line again.
The robot dynimic is non linear, with gaussian noise added.
Input : 
    xRO, float : initial robot position on the X axis 
    yR0, float : initial robot position on the Y axis
    pasTau, float : difference in translation between 2 consecutiv robot state
    covPos, float : dispersion of robot position in X and Y (the same value is used) around the ideal position
    covAng, float : dispersion of the robot orientation around the ideal value 
    covPos0, float : same as covPos, for the initial position 
    covAng0, float : same as covAng, for the initial orientation 
Output : 
    U, ndarray : command vector
    RobPos ndarray : real robot pose, with noise
    N1, int : instant of the end of the straight line  
    N2, int : instant of the end of the U-turn 
    N3, int : instant of the end of the second straight line 
    sigma0, ndarray : initial variance matrix 
    sigma1, ndarray : variance matrix between instant 1 and N1
    sigma2, ndarray : variance matrix between instant N1 and N2
    sigma3, ndarray : variance matrix between instant N2 and N3
"""    
def GenerateRobotPosition(xR0 : float, yR0 : float, pasTau : float, covPos : float, covAng : float, covPos0 : float, covAng0 : float ) -> np.ndarray :
    N1 = 70
    N2 = 45
    N3 = 70
    N = N1+N2+N3
    #Creation des vecteurs et matrices
    U = np.empty((2,N))
    RobPos = np.empty((3,N+1))
    sigma0 = np.diag([covPos0, covPos0, covAng0])
    sigma1 = np.diag([covPos, covPos, 0.00000000000000000000001])
    sigma2 = np.diag([0.2*covPos, 0.2*covPos, covAng])
    sigma3 = sigma1
    RobPos[:,0] = [xR0, yR0, 0] + np.linalg.cholesky(sigma0)@np.random.normal(size=(3,))
    
    #Vecteur de commande 
    for k in range(N1):
        U[:,k] = [pasTau, 0]
    for k in range(N1, N1+N2) :
        U[:,k] = [pasTau*0.7, np.pi/N2]
    for k in range(N2+N1, N) :
        U[:,k] = [pasTau, 0]
    
    #Calcul de la trajectoire du robot 
    for k in range(N1) :
        RobPos[:,k+1] = [RobPos[0, k]+math.cos(RobPos[2,k])*U[0, k], RobPos[1, k]+math.sin(RobPos[2,k])*U[0, k], RobPos[2,k]+U[1,k]]
        RobPos[:,k+1] = RobPos[:,k+1] + np.linalg.cholesky(sigma1)@np.random.normal(size=(3,))
    for k in range(N1, N1+N2) :
        RobPos[:,k+1] = [RobPos[0, k]+math.cos(RobPos[2,k])*U[0, k], RobPos[1, k]+math.sin(RobPos[2,k])*U[0, k], RobPos[2,k]+U[1,k]]
        RobPos[:,k+1] = RobPos[:,k+1] + np.linalg.cholesky(sigma2)@np.random.normal(size=(3,))
    for k in range(N2+N1, N) :
        RobPos[:,k+1] = [RobPos[0, k]+math.cos(RobPos[2,k])*U[0, k], RobPos[1, k]+math.sin(RobPos[2,k])*U[0, k], RobPos[2,k]+U[1,k]]
        RobPos[:,k+1] = RobPos[:,k+1] + np.linalg.cholesky(sigma3)@np.random.normal(size=(3,))
    return U, RobPos, N1, N2, N3, sigma0, sigma1, sigma2, sigma3

"""
Function for the generation of robot measurment. The observation model is non linear. 
Input : 
    NbInst, int : the number of instant in the simulation 
    Nbamer, int : the number of landmarks
    RobPos, ndarray : the robot real pose 
    amers, ndarray : the position of landmarks in the map
    covAng, float : the dispersion of noise for the bearing 
    covDis, float : the dispersion of noise for the range
Output :
    Mes : 2D array of dicitionnary with the measurment for each instant and each landmark
"""
def GenerateRobotMeasurment (NbInst : int, Nbamer : int, RobPos : np.ndarray, amers : np.ndarray, covAng : float, covDist : float) :
    Mes = [[dict() for x in range(Nbamer)] for y in range(NbInst)]
    for y in range(NbInst) :
        for x in range(0, 2*Nbamer, 2) : 
            #Calcul range et bearing 
            ran = math.sqrt((amers[x]-RobPos[0, y])**2+(amers[x+1]-RobPos[1, y])**2) + covDist/2*np.random.normal()
            bear = math.atan2(amers[x+1]-RobPos[1, y], amers[x]-RobPos[0, y])-RobPos[2, y] + covAng/2*np.random.normal()
            if (abs(bear)>math.pi/2 or ran > 2) :
                ran = np.nan
                bear = np.nan
            #Ajout dans la structure 
            Mes[y][int(x/2)] = {"amer" : x/2, "range" : ran, "bearing" : bear}
    return Mes