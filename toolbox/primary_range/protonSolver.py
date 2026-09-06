"""Proton depth-energy solver."""
import numpy as np
import pandas as pd
import scipy.io
import scipy as sc
from scipy.special import expi
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt

def protonSolve(pe):

    me = 9.10938356e-31
    e = 1.60217662e-19
    eo = 8.854187817e-12
    I = 75. * e
    n = 3.343e29
    mp = 1.6726219e-27
    z = 1
    c = 299792458

    ke_p = pe * 1e6 * e
    gam_p = (mp * c**2 + ke_p) / (mp * c**2)
    bet_p = np.sqrt(1 - (1 / gam_p)**2)

    vo = bet_p * c

    fc = (e ** 2) / (4 * np.pi * eo)
    A = ((4 * np.pi * n * (z ** 2)) / me) * (fc ** 2)
    B = (2 * me) / I
    As = A / mp

    mat_data = scipy.io.loadmat("toolbox/primary_range/EiInvTotalFull.mat")
    EiInv = mat_data['EiInv']
    EiInv = np.array(EiInv).flatten()

    d = np.arange(0, 0.4, 0.0001)

    R = expi(np.log(B ** 2 * vo ** 4)) / (2 * B ** 2 * As)

    v = np.zeros(d.size)
    for i in range(0, d.size):

        disE = R - d[i]
        TablePlace = round(2 * B ** 2 * As * disE)

        if TablePlace > 0:
            inv = EiInv[TablePlace]
            v[i] = ((1 / B ** 2) * np.exp(inv)) ** 0.25
        else:
            v[i] = 0

    gamma = 1 / np.sqrt(1 - (v / c) ** 2)
    gamma3 = gamma ** 3
    g3 = CubicSpline(d, gamma3)

    def spline_function(x):
        return g3(x)
    Rn, _ = quad(spline_function, 0, R)

    return Rn*100


if __name__ == '__main__':
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
