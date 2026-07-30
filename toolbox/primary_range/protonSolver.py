import numpy as np
import pandas as pd
import scipy.io
import scipy as sc
from scipy.special import expi
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt

def protonSolve(pe):
    #Constants
    me = 9.10938356e-31 #electron mass
    e = 1.60217662e-19 #electron charge
    eo = 8.854187817e-12 #vacuum permittivity
    I = 75. * e #ionization mean
    n = 3.343e29 #electron density(electrons / cc)
    mp = 1.6726219e-27 #particlemass - in this case proton
    z = 1 #charge in multiples of electron
    c = 299792458 #speed of  light

    ke_p = pe * 1e6 * e #proton energy(converted from MeV ie.150MeV)
    gam_p = (mp * c**2 + ke_p) / (mp * c**2) #relativstic gamma
    bet_p = np.sqrt(1 - (1 / gam_p)**2) #relativistic beta

    vo = bet_p * c #initial particle velocity

    #Collect together unweldy terms!

    fc = (e ** 2) / (4 * np.pi * eo)
    A = ((4 * np.pi * n * (z ** 2)) / me) * (fc ** 2)
    B = (2 * me) / I
    As = A / mp #ignore minus

    mat_data = scipy.io.loadmat("toolbox/primary_range/EiInvTotalFull.mat")
    EiInv = mat_data['EiInv']
    EiInv = np.array(EiInv).flatten()

    #START OF ANALYTICAL SOLUTION

    d = np.arange(0, 0.4, 0.0001)  #d is depth in matter in units metres, millimetres here

    R = expi(np.log(B ** 2 * vo ** 4)) / (2 * B ** 2 * As) #calculate max range

    v = np.zeros(d.size)
    for i in range(0, d.size):

        disE = R - d[i]
        TablePlace = round(2 * B ** 2 * As * disE) #Find position in look up table

        if TablePlace > 0:
            inv = EiInv[TablePlace]
            v[i] = ((1 / B ** 2) * np.exp(inv)) ** 0.25
        else:
            v[i] = 0 #speed is zero beyond end of range



    #now we make a gamma cubed function

    gamma = 1 / np.sqrt(1 - (v / c) ** 2)
    gamma3 = gamma ** 3
    g3 = CubicSpline(d, gamma3)



    #transform x' -> x
    def spline_function(x):
        return g3(x)  # Evaluate the spline at x
    Rn, _ = quad(spline_function, 0, R)

    return Rn*100




selected_energies = pd.read_csv('data/proton_energies.txt', header=None).to_numpy().reshape(-1)
Rs =[]
for ep in selected_energies:
    print(f'Starting the computation of Ep = {ep}')
    Rn = protonSolve(ep)
    print(f'just solved for {ep}')
    Rs.append(Rn)

data = {'Ep': selected_energies, 'Range': Rs}
df = pd.DataFrame(data=data)
df.to_csv('data/proton_range.csv')
