"""
Code realise pour l'etape 1 du projet bloc 3 SLAM du M2 AURO
Auteur : Carrierou A
But : 
    Creation et affichage d'une carte de N amers
    Modele de robot lineaire 
    SLAM base Kalman 
    Affichage au cours de la simulation
"""
import matplotlib.pyplot as plt
import numpy as np
import math

"""
Fonction de creation de la carte. 
Inuput : nombre amers, distance en x entre deux amers, distance en y entre les deux lignes d'amer, position premer amer (xA0, yA0)
Output : ndarray de dim (2*nbamer, 1) contenant Xamer0, Yamer0, Xamer1, Yamer1, ...
"""
def AmerCreation(nbamer : int, distX : int, distY : int, xA0 : int, yA0 : int, dispAmers : int) -> np.ndarray :
    print("... Creation de la Carte ...")
    amers = np.empty((2*nbamer,))
    amers[0:2] = (xA0, yA0)
    for i in range(2, nbamer, 2) :
        amers[i] = amers[i-2]+distX
        amers[i+1] = yA0
    for i in range(nbamer, 2*nbamer, 2) :
        amers[i] = amers[i-nbamer]
        amers[i+1] = yA0+distY

    #Ajout de bruit dans la position des amers 
    sigma = np.zeros((2*nbamer, 2*nbamer))
    for i in range(2*nbamer) :
        sigma[i,i] = dispAmers
    amersBruit = amers + np.linalg.cholesky(sigma)@np.random.normal(size=(2*nbamer,))
    print("--- Carte cree ---")
    return amers, amersBruit

"""
Fonction pour la generation de la trajectoire de la commande du robot 
Input xR0, yR0 : position initiale du robot, amers : position des amers non bruitee, pas : pas de translation de la commande, covPos, covAng : covarience du bruit pour la position et l'orientation du robot
Output : U : matrice avec la commande du robot pour chaque instant, xR matrice avec la pose du robot pour chaque instant
"""
def GenerateRobotPosition(xR0 : int, yR0 : int, amers : np.ndarray, pas : float, covPos : float, covAng : float ) -> np.ndarray :
    #Recuperation des donnees pour la generation de la commande 
    depX = (amers[2]-amers[0]) * (nbamer/2-1) + amers[0] + xR0 + 1
    depY = (amers[nbamer+1]-amers[1])/2 + amers[1] + yR0

    # Boucle de generation de la commande 
    print("... Calcul de la commande ...")
    elemX = depX/pas
    elemY = depY/pas
    k = 0 
    U = np.empty((3,int(2*elemX+elemY+2)))
    while k < elemX :
        U[:,k] = (pas, 0, 0)
        k += 1
    U[:,k] = (0, 0, np.pi/2)
    k += 1
    while k < elemY+elemX+1 :
        U[:,k] = (0, pas, 0)
        k += 1
    U[:,k] = (0, 0, np.pi/2)
    k += 1
    while k < 2*elemX+elemY+2 :
        U[:,k] = (-pas, 0, 0)
        k += 1
    print("--- Commande calculee ---")

    # Boucle de generation de la trajectoire 
    print("\n... Calcul de la trajectoire ...")
    xR = np.empty((3, U.shape[1]+1))
    xRB = np.empty((3, U.shape[1]+1))
    xR[:,0] = (xR0, yR0, 0)
    k=0
    while k < U.shape[1] :
        xR[:,k+1] = xR[:,k]+U[:,k]
        k += 1
    #Ajout du bruit dans la trajcetoire
    sigma = np.zeros((3, 3))
    sigma[0,0] = covPos
    sigma[1,1] = covPos
    sigma[2,2] = covAng

    k = 0
    while k < xR.shape[1] :
        xRB[:,k] = xR[:,k]+np.linalg.cholesky(sigma)@np.random.normal(size=(3,))
        k += 1
    print("--- Trajectoire calculee ---")       
    return U, xR, xRB

""""
Fonction de generation des mesures du robot par un champ de vision circulaire 
Input Rpose : la pose reelle du robot, amers : la position reelle des amers, distVizu : le rayon du cercle du champ de vision du robot
Output : Mes : le vecteur des mesures recues par le robot pour chaque instant
"""
def RobotVizu(Rpose : np.ndarray, amers : np.ndarray, distVizu : float, covBruit : float) -> np.ndarray:
    print("\n... Calcul des mesures ...")
    Nbamer = amers.shape[0]
    Nbinst = Rpose.shape[1]
    Mes = np.empty((Nbamer, Nbinst))

    for k in range(1, Nbinst) :
        for n in range(0,Nbamer,2) :
            Xrel = amers[n] - Rpose[0,k]
            Yrel = amers[n+1] - Rpose[1,k]
            if math.sqrt(Xrel**2+Yrel**2) <= distVizu :
                #Pour un scalaire, chol(X) = sqrt(X)
                Mes[n, k] = Xrel + math.sqrt(covBruit) * np.random.normal()
                Mes[n+1, k] = Yrel + math.sqrt(covBruit) * np.random.normal()
            else :
                Mes[n, k] = np.nan
                Mes[n+1, k] = np.nan
    print("--- Mesures calculees ---") 
    return Mes

"""
Fonction pour l'affichage de la carte avec les amers
Input : l'etat contenant la position des amers
Output : None 
"""
def MapPlot(amers : np.ndarray) :
    fig, ax = plt.subplots()
    N = amers.shape[0]
    x = np.empty((int(N/2),))
    y = np.empty((int(N/2),))
    for i in range(0, N, 2):
        x[int(i/2)] = amers[i]
        y[int(i/2)] = amers[i+1]

    ax.scatter(x, y, marker="X", color="darkcyan")
    ax.set(xlim=(-1, 10), xticks=np.arange(-1, 11), ylim=(-1, 5), yticks=np.arange(-1, 6))
    plt.grid()
    plt.draw()
    return fig, ax

"""
Fonction pour l'affichage de la carte avec les amers et la trajectoire du robot 
Input : etat : pose du robot pour toute la simulation, amers : position des amers 
Output : None
"""
def PlotRobotMap(etat : np.ndarray, amers : np.ndarray) :
    fig, ax = MapPlot(amers)
    N = etat.shape[1]
    x = np.empty((N,))
    y = np.empty((N,))
    for i in range(N) :
        x[i] = etat[0,i]
        y[i] = etat[1,i]
    ax.plot(x, y, color="seagreen", marker="1")
    ax.set_title('Carte avec les amers et la trajectoire reelle du robot', fontsize=14)
    print("\n--- Fermez la figure pour continuer ---\n")
    plt.draw()
    plt.show()


if __name__ == '__main__':
    #Donnee
    nbamer = 8
    distX = distY = 2
    xA0, yA0 = (1,1)
    dispAmers = 0.01
    pas = 0.1
    xR0, yR0 = (0,0)
    distVizu = 2
    covB = 0.01

    #Simumation de l'environnement et du deplacement
    amers, amersB = AmerCreation(nbamer, distX, distY, xA0, yA0, dispAmers)
    U, RobPose, RobPoseB = GenerateRobotPosition(xR0, yR0, amers, pas, 0.0005, 0.00001)
    PlotRobotMap(RobPoseB, amersB)
    Z = RobotVizu(RobPoseB, amersB, distVizu, covB)

    #Filtrage