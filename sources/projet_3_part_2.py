import matplotlib.pyplot as plt
import numpy as np
import math
from affichage import PlotRes

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

