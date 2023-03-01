"""
Code realise pour l'etape 1 du projet bloc 3 SLAM du M2 AURO
Auteur : Carrierou A
But : 
    Creation et affichage d'une carte de N amers
    Modele de robot lineaire 
    SLAM base Kalman 
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes

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
    xRB[:,0] = (xR0, yR0, 0)
    sigma = np.diag([covPos, covPos, covAng])
    k=0
    while k < U.shape[1] :
        xR[:,k+1] = xRB[:,k]+U[:,k]
        xRB[:,k+1] = xR[:,k+1]+np.linalg.cholesky(sigma)@np.random.normal(size=(3,))
        k += 1
    xRB[:,k] = xR[:,k]+np.linalg.cholesky(sigma)@np.random.normal(size=(3,))
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
    Mes = np.empty((Nbamer, Nbinst-1))

    for k in range(1, Nbinst) :
        for n in range(0,Nbamer,2) :
            Xrel = amers[n] - Rpose[0,k]
            Yrel = amers[n+1] - Rpose[1,k]
            if math.sqrt(Xrel**2+Yrel**2) <= distVizu :
                #Pour un scalaire, chol(X) = sqrt(X)
                Mes[n, k-1] = Xrel + math.sqrt(covBruit) * np.random.normal()
                Mes[n+1, k-1] = Yrel + math.sqrt(covBruit) * np.random.normal()
            else :
                Mes[n, k-1] = np.nan
                Mes[n+1, k-1] = np.nan
    print("--- Mesures calculees ---") 
    return Mes

"""
Fonction de tri des mesures, permet d'enlever les elements NaN du vecteur de mesure
"""
def MeasSelect(Mes : np.ndarray, NbEtat : int, dispAmers) : 
    #indZ initial : 0 -> indice du dernier element ajoute dans Zuse, H init : 3*3 
    indZ = 0
    H = np.zeros((2,NbEtat))
    Zuse = np.ndarray((2,))
    for k in range(0, Mes.size, 2) :
        if not np.isnan(Mes[k]) :
            if indZ == 0:
                H[0,0] = -1
                H[0, k+3] = 1
                H[1, 1] = -1
                H[1, k+4] = 1
                Zuse[0] = Mes[k]
                Zuse[1] = Mes[k+1]
            else :
                H = np.append(H, np.zeros((2,NbEtat)), axis=0)
                H[indZ,0] = -1
                H[indZ, k+3] = 1
                H[indZ+1, 1] = -1
                H[indZ+1, k+4] = 1
                Zuse = np.append(Zuse, [[Mes[k]],[Mes[k+1]]])
            indZ +=2
    Rv = np.diag(dispAmers*np.ones(Zuse.shape[0]))
    return Zuse, H, Rv



if __name__ == '__main__':
    #Donnees
    nbamer = 8
    distX = distY = 2
    xA0, yA0 = (1,1)
    dispAmers = 0.01
    pas = 0.1
    xR0, yR0 = (0,0)
    covDis = 0.00025
    covAng = 0.01
    distVizu = 2
    covB = 0.01

    #Simumation de l'environnement et du deplacement
    amers, amersB = AmerCreation(nbamer, distX, distY, xA0, yA0, dispAmers)
    U, RobPose, RobPoseB = GenerateRobotPosition(xR0, yR0, amers, pas, covDis, covAng)
    Z = RobotVizu(RobPoseB, amersB, distVizu, covB)

    #Filtrage
    #Initialisation
    print("Filtrage - Initialisation")
    NbInst = U.shape[1]+1
    Xest = np.empty((3+nbamer*2, NbInst))
    Pest = np.empty((3+nbamer*2,3+nbamer*2,NbInst))
    Xpred = np.empty((3+nbamer*2, NbInst))
    Ppred = np.empty((3+nbamer*2,3+nbamer*2,NbInst))
    Qw = np.diag(np.append([covDis, covDis, covAng], 0.000001*np.ones(2*nbamer)))
    B = np.append(np.identity(3), np.zeros((2*nbamer, 3)), axis=0)

    Xest[:,0] = np.append([xR0, yR0, 0], amers)
    Pest[:,:,0] = np.diag(np.append([0, 0, 0], dispAmers*np.ones(2*nbamer)))

    for k in range(1, NbInst) :
        print('Kalman : prediction, iteration %d' %(k))
        #Prediction 
        Xpred[:,k] = Xest[:,k-1]+B@U[:,k-1]
        Ppred[:,:,k] = Pest[:,:,k-1] + Qw #A = identite, pas besoin de le mettre
        print('Kalman : mise a jour, iteration %d' %(k))
        #Mise a jour 
        Zuse, H, Rv = MeasSelect(Z[:,k-1], Xpred.shape[0], dispAmers) #Decalage de 1 pour les mesures
        Zest = (H@Xpred[:,k]).T
        S = Rv + H@Ppred[:,:,k]@H.T
        K = Ppred[:,:,k]@H.T@np.linalg.inv(S)
        Xest[:,k] = Xpred[:,k] + K@(Zuse-Zest)
        Pest[:,:,k] = Ppred[:,:,k] - K@H@Ppred[:,:,k]
    
    #Affichage des resultats 
    PlotRes(RobPoseB, amersB, Xest, Pest, NbInst)