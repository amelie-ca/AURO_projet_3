import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes, PlotRobotMap

def AmerCreation(nbamer : int, distX : float, distY : float, xA0 : float, yA0 : float) -> np.ndarray :
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

#Pour les mesures : dictlist = [[dict() for x in range(n)] for y in range(6)] permet de creer un dictionnaire 2D, le faire pour les X amers et à chaque amer mettre mesure en angle, distance et cov de bruit associee a chaque elem

if __name__ == '__main__':
    #Donnees 
    Nbamer = 8
    distX = distY = 2
    xA0, yA0 = (1,1)
    dispAmers = 0.01
    tau = 0.1
    xR0, yR0 = (0,0)
    covDis = 0.0001
    covAng = np.pi/9500
    covDis0 = 0.000001
    covAng0 = np.pi/10000



    amers = AmerCreation(Nbamer, distX, distY, xA0, yA0)
    U, Xreel, Nb1, Nb2, Nb3, PX0, PX1, PX2, PX3 = GenerateRobotPosition(xR0, yR0, tau, covDis, covAng, covDis0, covAng0)
    PlotRobotMap(Xreel, amers, 'test', (1,1,1))
    print("\n--- Fermez la figure pour terminer ---\n")
    plt.show()