import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes, PlotRobotMap

def AmerCreation(nbamer : int, distX : int, distY : int, xA0 : int, yA0 : int) -> np.ndarray :
    print("... Creation de la Carte ...")
    amers = np.empty((2*nbamer,))
    amers[0:2] = (xA0, yA0)
    for i in range(2, nbamer, 2) :
        amers[i] = amers[i-2]+distX
        amers[i+1] = yA0
    for i in range(nbamer, 2*nbamer, 2) :
        amers[i] = amers[i-nbamer]
        amers[i+1] = yA0+distY
    return amers

def GenerateRobotPosition(xR0 : int, yR0 : int, pasTau : float, covPos : float, covAng : float ) -> np.ndarray :
    N1 = 50
    N2 = 45
    N3 = 50
    N = N1+N2+N3
    #Creation des vecteurs et matrices
    U = np.empty((2,N))
    RobPos = np.empty((3,N+1))
    RobPos[:,0] = [xR0, yR0, 0]

    #Vecteur de commande 
    for k in range(N1):
        U[:,k] = [pasTau, 0]
    for k in range(N1, N1+N2) :
        U[:,k] = [pasTau*0.7, np.pi/N2]
    for k in range(N2+N1, N) :
        U[:,k] = [pasTau, 0]
    
    #Calcul de la trajectoire du robot 
    for k in range(N) :
        RobPos[:,k+1] = [RobPos[0, k]+math.cos(RobPos[2,k])*U[0, k], RobPos[1, k]+math.sin(RobPos[2,k])*U[0, k], RobPos[2,k]+U[1,k]]
        print(RobPos[:,k])
    return U, RobPos, N+1 

if __name__ == '__main__':

    amers = AmerCreation(8, 1, 2, 1, 1)
    U, Xreel, Nb = GenerateRobotPosition(0, 0, 0.1, 0.01, 0.01)
    PlotRobotMap(Xreel, amers, 'test', 1)
    plt.show()