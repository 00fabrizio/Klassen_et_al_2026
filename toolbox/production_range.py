import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "primary_range"

def build_xt_table(species: str, Ep_list, En_list_gev):
    """
    Returns
    -------
    XT : (nEp, nEn) array
        XT[i, j] = xt(Ep_list[i], En_list_gev[j]) in cm
        using your *original* mapping code (np.interp in En inside each file),
        but evaluated only at En_list_gev and only once.
    """
    Ep_list = np.asarray(Ep_list, dtype=float)
    En_list_gev = np.asarray(En_list_gev, dtype=float)
    En_mev = En_list_gev * 1000.0

    XT = np.empty((Ep_list.size, En_list_gev.size), dtype=np.float32)

    for i, Ep in enumerate(Ep_list):
        if species == "proton":
            df = pd.read_csv(BASE / "protonSolver" / f"protonSolver_{Ep}MeV")
            EnergyMEV = df["EnergyMEV"].to_numpy(dtype=float)
            xt = df["xt"].to_numpy(dtype=float) * 100.0  # -> cm
        elif species == "carbon":
            df = pd.read_csv(BASE / "carbonSolver" / f"carbonSolver_{Ep}MeV")
            EnergyMEV = df["EnergyMEV"].to_numpy(dtype=float) / 6.0
            xt = df["xt"].to_numpy(dtype=float) * 100.0
        else:
            raise ValueError(species)

        mask = EnergyMEV > 0
        EnergyMEV = EnergyMEV[mask]
        xt = xt[mask]

        order = np.argsort(EnergyMEV)
        EnergyMEV = EnergyMEV[order]
        xt = xt[order]

        # only evaluate at your En points (still uses 1D interpolation internally, but only once)
        XT[i, :] = np.interp(En_mev, EnergyMEV, xt).astype(np.float32)

    return XT
