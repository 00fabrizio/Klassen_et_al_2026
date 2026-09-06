"""Reader for the raw FLUKA .lis output."""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def load_radial_data(species, energies):
    MC = []
    ERR = []
    for energy in energies:
        num_e = 250
        num_r = 55
        num_z = 45

        r = np.round(np.linspace(0.1, 5.5, num_r), 3)
        z = np.round(np.linspace(0, 44.0, num_z), 3) + 1

        r_cutoff = 55
        r = r[:r_cutoff]

        filename = f'data/{species}/{energy}/New_ring_{species}_22_tab.lis'
        get_rows = pd.read_csv(filename, names=['header'])

        mc, err = np.zeros((num_z, num_r, num_e)), np.zeros((num_z, num_r, num_e))
        get_rows = pd.read_csv(filename, names=['header'])
        rows = get_rows[get_rows['header'].str.contains('Detector')].index.to_numpy()[:-25].reshape((num_r, num_z))
        rows = rows[:r_cutoff, :]
        rows = rows + 2 

        for j in range(r_cutoff):
            for i in range(num_z):
                start = rows[j, i]
                block = get_rows.iloc[start:start+num_e]['header']
                df_block = block.str.split(expand=True)
                mc[i, j, :] = df_block[2].astype(float).to_numpy()
                err[i, j, :] = df_block[3].astype(float).to_numpy()

        en_low = df_block[0].astype(float).to_numpy()
        en_upp = df_block[1].astype(float).to_numpy()

        mc = np.flip(mc, axis=0)*en_low
        err = np.flip(err, axis=0)
        mc  = mc[:, :r_cutoff, :]
        err = err[:, :r_cutoff, :]*mc

        MC.append(mc)
        ERR.append(err)
    MC = np.stack(MC, axis=0)
    ERR = np.stack(ERR, axis=0)
    return z, r, en_low, en_upp, MC, ERR
