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

"""
Fonction de creation de la carte. 
Inuput : nombre amers, distance en x entre deux amers, distance en y entre les deux lignes d'amer, position premer amer (xA0, yA0)
Output : ndarray de dim (2*nbamer, 1) contenant Xamer0, Yamer0, Xamer1, Yamer1, ...
"""
def AmerCreation (nbamer : int, distX : int, distY : int, xA0 : int, yA0 : int, dispAmers : int) -> np.ndarray :
    print("... Creation de la Carte ...")
    amers = np.ndarray((2*nbamer,))
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
Input xR0, yR0 : position initiale du robot, amers : position des amers non bruitee, pas : pas de translation de la commande
Output : U : matrice avec la commande du robot pour chaque instant, xR matrice avec la pose du robot pour chaque instant
"""


def GenerateRobotPosition(xR0 : int, yR0 : int, amers : np.ndarray, pas : float ) -> np.ndarray :
    #Recuperation des donnees pour la generation de la commande 
    print(amers[2], amers[0], nbamer-1)
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
    U[:,k] = (0, 0, np.pi)
    k += 1
    while k < elemY+elemX+1 :
        U[:,k] = (0, pas, 0)
        k += 1
    U[:,k] = (0, 0, np.pi)
    k += 1
    while k < 2*elemX+elemY+2 :
        U[:,k] = (-pas, 0, 0)
        k += 1
    print("--- Commande calculee ---")

    # Boucle de generation de la trajectoire 
    print("... Calcul de la trajectoire ...")
    xR = np.empty((3, U.shape[1]+1))
    xR[:,0] = (xR0, yR0, 0)
    k=0
    while k < U.shape[1] :
        xR[:,k+1] = xR[:,k]+U[:,k]
        k += 1
    print("--- Trajectoire calculee ---")       
    return U, xR


"""
Fonction pour l'affichage de la carte 
Input : l'etat contenant la position des amers
Output : None 
"""
def MapPlot (etat : np.ndarray):
    fig, ax = plt.subplots()
    N = etat.shape[0]
    x = np.ndarray((int(N/2),))
    y = np.ndarray((int(N/2),))
    for i in range(0, N, 2):
        x[int(i/2)] = etat[i]
        y[int(i/2)] = etat[i+1]

    ax.scatter(x, y, marker="X")
    ax.set(xlim=(-1, 10), xticks=np.arange(-1, 11), ylim=(-1, 5), yticks=np.arange(-1, 6))
    ax.set_title('Carte avec les amers')
    plt.grid()
    print("\n--- Fermez la figure pour continuer ---\n")
    plt.show()

if __name__ == '__main__':
    #Donnee
    nbamer = 8
    distX = distY = 2
    xA0, yA0 = (1,1)
    dispAmers = 0.01
    pas = 0.1
    xR0, yR0 = (0,0)

    #Simumation de l'environnement et du deplacement
    amers, amersB = AmerCreation(nbamer, distX, distY, xA0, yA0, dispAmers)
    MapPlot(amersB)
    U, RobPose = GenerateRobotPosition(xR0, yR0, amers, pas)

    #Filtrage