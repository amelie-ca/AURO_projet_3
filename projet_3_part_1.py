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
def AmerCreation (nbamer : int, distX : int, distY : int, xA0 : int, yA0 : int) -> np.ndarray :
    print("... Creation de la Carte ...")
    amers = np.ndarray((2*nbamer,))
    amers[0:2] = (xA0, yA0)
    for i in range(2, nbamer, 2) :
        amers[i] = amers[i-2]+distX
        amers[i+1] = yA0
    for i in range(nbamer, 2*nbamer, 2) :
        amers[i] = amers[i-nbamer]
        amers[i+1] = yA0+distY
    print("--- Carte cree ---")
    return amers

def MapPlot (etat : np.ndarray):
    fig, ax = plt.subplots()
    N = etat.shape[0]
    x = np.ndarray((int(N/2),))
    y = np.ndarray((int(N/2),))
    for i in range(0, N, 2):
        x[int(i/2)] = etat[i]
        y[int(i/2)] = etat[i+1]

    ax.scatter(x, y, marker="+")
    ax.set(xlim=(-1, 10), xticks=np.arange(-1, 11), ylim=(-1, 5), yticks=np.arange(-1, 6))

    ax.set_title('Carte avec les amers')
    plt.show()

if __name__ == '__main__':
    nbamer = 8
    distX = distY = 2
    xA0, yA0 = (1,2)
    amers = AmerCreation(nbamer, distX, distY, xA0, yA0)
    MapPlot(amers)