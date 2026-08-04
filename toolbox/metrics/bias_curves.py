"""Volume-weighted bias curves, evaluated in parallel over primary energies.

The bias metric (paper Eq. 10) contracts z, rho and E_n away into two scalars
per primary energy, so each energy is an independent job. The model evaluation
is ~99 % of the cost and is single-threaded, so spreading the energies over
processes is close to a linear speedup.
"""
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

COLS = ['A1', 'gamma1', 'Sigma_t', 'Sigma', 'd1', 'a', 'w_c',
        'A2', 'gamma2', 'd2', 'kappa_ev', 'Epk',
        'A3', 'gamma3', 'A4', 'gamma4', 'kappa_f', 'kappa_s', 'E_th', 'n']

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = os.path.join(_ROOT, 'npy_data', '.bias_cache')


def _one_energy(job):
    """Bias of one primary energy. Runs in a worker process."""
    species, i, Ep, nr, vals = job

    import numpy as np
    from toolbox.production_range import build_xt_table
    from toolbox.metrics.kerma_coeffs import k_coeff_pGy_cm2_from_GeV
    from model import Model

    root = _ROOT
    z = np.load(f'{root}/npy_data/z.npy')
    rho = np.load(f'{root}/npy_data/rho.npy')[:nr]
    en_low = np.load(f'{root}/npy_data/en_low.npy')
    dE = np.load(f'{root}/npy_data/en_upp.npy') - en_low
    kc = k_coeff_pGy_cm2_from_GeV(en_low)
    nz, nE = len(z), len(en_low)

    mc = np.asarray(np.load(f'{root}/npy_data/{species}_mc.npy', mmap_mode='r')[i][:, :nr, :])
    XT = build_xt_table(species, np.array([Ep]), en_low)
    M = Model(species=species, XT=XT,
              iEp=np.repeat(np.arange(1), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in
                    np.meshgrid([Ep], z, rho, en_low, indexing='ij')]
    pr = M.spectral_energy_fluence_prefac2((Z, R, En, EP), *vals).reshape(nz, nr, nE)

    phim, phip = np.sum(mc * dE, axis=2), np.sum(pr * dE, axis=2)
    km, kp = np.sum(mc * dE * kc, axis=2), np.sum(pr * dE * kc, axis=2)
    return (np.sum((phip - phim) * rho) / np.sum(phim * rho) * 100,
            np.sum((kp - km) * rho) / np.sum(km * rho) * 100)


def bias_curves(species, param_file=None, nr=55, workers=None, cache=True):
    """Return (Ep, delta_Phi, delta_K) in percent for every primary energy.

    Results are cached under npy_data/.bias_cache, keyed by the parameter
    values and nr, so re-running with unchanged parameters is instant.
    """
    param_file = param_file or f'{_ROOT}/fitting_params/{species}_prefac2_twogroup.csv'
    par = pd.read_csv(param_file, index_col=0).loc['opt params']
    vals = [float(par[c]) for c in COLS]
    Ep_all = np.load(f'{_ROOT}/npy_data/{species}_energies.npy')

    key = hashlib.md5(
        f'{species}|{nr}|{"|".join(repr(v) for v in vals)}'.encode()).hexdigest()[:16]
    path = os.path.join(_CACHE, f'{species}_{key}.npz')
    if cache and os.path.exists(path):
        d = np.load(path)
        return d['Ep'], d['dphi'], d['dk']

    jobs = [(species, i, float(E), nr, vals) for i, E in enumerate(Ep_all)]
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
        out = list(ex.map(_one_energy, jobs, chunksize=1))
    dphi = np.array([o[0] for o in out])
    dk = np.array([o[1] for o in out])

    if cache:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez(path, Ep=Ep_all, dphi=dphi, dk=dk)
    return Ep_all, dphi, dk
