import matplotlib.pyplot as plt
import numpy as np

"""
Fonction pour l'affichage de la carte avec les amers
Input : l'etat contenant la position des amers
Output : None 
"""
def MapPlot(amers : np.ndarray, posPlot) :
    ax = plt.subplot(posPlot[0], posPlot[1], posPlot[2])
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
    return ax

"""
Fonction pour l'affichage de la carte avec les amers et la trajectoire du robot 
Input : etat : pose du robot pour toute la simulation, amers : position des amers 
Output : None
"""
def PlotRobotMap(etat : np.ndarray, amers : np.ndarray, title : str, posPlot) :
    ax = MapPlot(amers, posPlot)
    N = etat.shape[1]
    x = np.empty((N,))
    y = np.empty((N,))
    for i in range(N) :
        x[i] = etat[0,i]
        y[i] = etat[1,i]
    ax.plot(x, y, color="seagreen", marker="1")
    ax.set_title(title, fontsize=14)
    plt.draw()
    
"""
Fonction qui permet d'afficher 6 cartes differents : une carte de la situation reelle, 
une carte de la situation totale estimee et 4 cartes pour des situation estimee intermediaires
Inupt etatReel : le vecteur d'etat reel du robot, amersReel : le vecteur de la position reelle des amers, etatEst : le vecteur d'etats estimes robot et amers, NbInst : le nombre d'iteration du filtre
"""
def PlotRes(etatReel : np.ndarray, amersReel : np.ndarray, etatEst : np.ndarray, NbInst : int) : 
    N = int(NbInst/5)
    PlotRobotMap(etatReel, amersReel, 'Realite terain', (2,3,1))
    PlotRobotMap(etatEst[:3, :], etatEst[3:, NbInst-1], "Estimation finale", (2,3,4))
    PlotRobotMap(etatEst[:3, :N-1], etatEst[3:, N-1], "Estimation a l'iteration "+str(N), (2,3,2))
    PlotRobotMap(etatEst[:3, :2*N-1], etatEst[3:, 2*N-1], "Estimation a l'iteration "+str(2*N), (2,3,3))
    PlotRobotMap(etatEst[:3, :3*N-1], etatEst[3:, 3*N-1], "Estimation a l'iteration "+str(3*N), (2,3,5))
    PlotRobotMap(etatEst[:3, :4*N-1], etatEst[3:, 4*N-1], "Estimation a l'iteration "+str(4*N), (2,3,6))
    print("\n--- Fermez la figure pour terminer ---\n")
    plt.show()
