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
"""
Function for the creation of new particles acording to robots dynamics 
Input : 
    N, int : number of particles 
    PartsM, dictlist : particles for the previous iteration
    U, ndarray : commend vector for the generation of new particles
    Qw : variance of dynamic noise 
Output : 
    PartsF : new particles generated
"""
def Propag(N : int, NbLM : int, PartsM, U : np.ndarray, Qw : np.ndarray, Z, covA : float, covD : float) :
    PartsF = [dict() for i in range(N)]
    X = np.empty(3)
    SumW = 0
    S = 0
    
    for k in range(N) :
        theta = PartsM[k]["RobPos"][2]
        X[0] = PartsM[k]["RobPos"][0] + U[0]*math.cos(theta)
        X[1] = PartsM[k]["RobPos"][1] +U[0]*math.sin(theta)
        X[2] = PartsM[k]["RobPos"][2] +U[1]
        X = X + np.linalg.cholesky(Qw)@np.random.normal(size=(3,))
        PartsF[k] = {"RobPos" : X, "Amers" : PartsM[k]["Amers"], "W" : 0}
        Zest = DetMes(PartsF[k], NbLM)
        W = CalclW(Z, Zest, PartsM[k]["W"], NbLM, covA, covD)  
        SumW += W
        PartsF[k]["W"] = W
        print(PartsF[k]["W"])
    print('\n')
    for k in range(N) : 
        PartsF[k]["W"] = PartsF[k]["W"]/SumW
        S+=PartsF[k]["W"] 
        print(PartsF[k]["W"])
    print('\n')
    print(S)
    print('-----------------------')
        


    return PartsF

def DetMes(Part, NbLM : int) : 
    Zest = [dict() for x in range(NbLM)]
    for i in range(0, 2*NbLM, 2) : 
            rang = math.sqrt((Part['Amers'][i]-Part['RobPos'][0])**2+(Part['Amers'][i+1]-Part['RobPos'][1])**2)
            bear = math.atan2(Part['Amers'][i+1]-Part['RobPos'][1], Part['Amers'][i]-Part['RobPos'][0])-Part['RobPos'][2] 
            Zest[int(i/2)] = {'amer' : i, 'range' : rang, 'bearing' : bear}
    return Zest

def CalclW(Zr, Zest, Wm, NbLM, covA, covD) :
    W = 0 
    Rv = np.diag([covD, covA])
    Mes = False
    for i in range(NbLM) :
        if not np.isnan(Zr[i]["range"]) :
            ErrRang = Zr[i]["range"] - Zest[i]["range"]
            ErrBear = Zr[i]["bearing"] - Zest[i]["bearing"]
            Err = np.array((ErrRang, ErrBear))
            W += -1/2 * (Err@np.linalg.inv(Rv)@Err.T)
            Mes = True
    if Mes : 
        W = W*math.log(Wm)
        W = math.exp(W)
    else : 
        W = Wm
    return W
        
def EspPart(Parts, NbPart, Nbamer) :
    X = np.empty((NbPart, int(3+2*Nbamer)))
    Xest = np.zeros(int(3+2*Nbamer))
    Pest = 0
    for i in range(NbPart) : 
        X[i,:3] = Parts[i]["RobPos"]
        X[i, 3:] = Parts[i]["Amers"]
        Xest = Xest + Parts[i]["W"]*X[i]
    for i in range(NbPart) :
        Pest =  Pest + (X[i]-Xest)@(X[i]-Xest).T
    return Xest, Pest


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
    PlotRobotMap(Xreel, amers, "Trajectoire")
    print("--- Carte affichee, fermez la fenetre pour continuer ---")
    plt.show()

    #Filtrage 
    #Initialisation 
    Part = [[dict() for x in range(NbPart)] for y in range(N)]
    Part[0] = InitPF(NbPart, Mamer, dispAmers)
    Neff = 0 
    for i in range(NbPart) :
        Neff = Neff + Part[0][i]["W"]**2
    Neff = 1/Neff
    print("Neff initial : ", Neff)
   
    for k in range(1, N) :
        print('-----------------------')
        print("Iteration : ", k, '\n')
        #Propagation 
        if k<Nb1 :
            Qw = Qw1
        elif k<Nb2+Nb1 :
            Qw = Qw2
        else :
            Qw = Qw3
        Part[k] = Propag(NbPart, Nbamer, Part[k-1], U[:,k-1], Qw, Zr[k-1], covAngMes, covDisMes)
        #Estim et cov 
        Xest, Pest = EspPart(Part[k], NbPart, Nbamer)
        Neff = 0 
        for i in range(NbPart) :
            Neff = Neff + Part[k][i]["W"]**2
        Neff = 1/Neff
        print('Estimations : ', Xest, '\t', Pest, '\t', Neff)