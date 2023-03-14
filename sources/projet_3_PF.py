import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes, PlotRobotMap

def AmerCreation(nbamer : int, distX : float, distY : float, xA0 : float, yA0 : float, dispA : float) -> np.ndarray :
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

if __name__ == '__main__':