"""Volume-weighted bias curves (paper Eq. 10), in parallel over primary energies.

The metric contracts z, rho and E_n away into two scalars per primary energy, so
each energy is an independent job. The model evaluation is ~99 % of the cost and
is single-threaded, so spreading energies over processes is close to a linear
speedup.

Eq. 10 is defined on the DIFFERENTIAL fluence. The npy cache holds
E dPhi/dE, so phi = mc / en_low and

    Phi = sum phi dE            weight dE / en
    K   = sum k_phi phi dE      weight dE kc / en

Using dE and dE kc instead carries a spurious factor of E and shifts the band
shares badly (evaporation 8.6 % -> 42.1 % of kerma).

Note that this is NOT the fitting objective, which is unweighted least squares on
the energy fluence (~88 % cascade). Eq. 10 grades particle fluence (~13 % slow)
and kerma (~0 % slow). The two do not rank parameter sets the same way.
"""
import glob
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# positional order of both production parameter files
COLS = ['A1', 'gamma1', 'Sigma', 'ang', 'd1', 'a', 'w_c',
        'A2', 'gamma2', 'd2', 'kappa_ev', 'Epk',
        'A3', 'gamma3', 'A4', 'gamma4', 'kappa_slow', 'E_th', 'n']

DEFAULT_PARAMS = {
    'proton': 'proton_coupled_single.csv',
    'carbon': 'carbon_coupled_theta_10E.csv',
}

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = os.path.join(_ROOT, 'npy_data', '.bias_cache')


def _load(species, param_file=None):
    """Return (values, entry_point_name). Slot 3 fixes which entry point."""
    f = param_file or os.path.join(_ROOT, 'fitting_params',
                                   DEFAULT_PARAMS[species])
    par = pd.read_csv(f, index_col=0).loc['opt params']
    cols = list(par.index)
    entry = ('spectral_energy_fluence_theta' if 'theta_bar' in cols
             else 'spectral_energy_fluence')
    return [float(v) for v in par], entry


def _one_energy(job):
    """Bias of one primary energy. Runs in a worker process."""
    species, i, Ep, nr, vals, entry = job

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

    mc = np.asarray(np.load(f'{root}/npy_data/{species}_mc.npy',
                            mmap_mode='r')[i][:, :nr, :])
    M = Model(species=species,
              XT=build_xt_table(species, np.array([Ep]), en_low),
              iEp=np.repeat(np.arange(1), nz * nr * nE),
              iEn=np.tile(np.arange(nE), nz * nr))
    EP, Z, R, En = [a.ravel() for a in
                    np.meshgrid([Ep], z, rho, en_low, indexing='ij')]
    pr = getattr(M, entry)((Z, R, En, EP), *vals).reshape(nz, nr, nE)

    out = []
    for w in (dE / en_low, dE * kc / en_low):          # Eq. 10 Phi, Eq. 10 K
        m = np.sum(mc * w, axis=2)
        q = np.sum(pr * w, axis=2)
        out.append(np.sum((q - m) * rho) / np.sum(m * rho) * 100)
    return tuple(out)


def bias_curves(species, param_file=None, nr=55, workers=None, cache=True):
    """Return (Ep, delta_Phi, delta_K) in percent for every primary energy.

    Results are cached under npy_data/.bias_cache, keyed by the parameter
    values, nr, AND the identity of the kernel tables. The tables are part of
    the model: rebuilding one changes the curves at unchanged parameters, so
    keying on parameters alone would serve pre-rebuild results silently.
    """
    vals, entry = _load(species, param_file)
    Ep_all = np.load(f'{_ROOT}/npy_data/{species}_energies.npy')

    tabs = []
    for f in sorted(glob.glob(os.path.join(_ROOT, 'toolbox',
                                           'diffusion_integrals', '*.dat'))):
        st = os.stat(f)
        tabs.append(f'{os.path.basename(f)}:{st.st_size}:{st.st_mtime_ns}')

    key = hashlib.md5(
        f'{species}|{nr}|{entry}|{"|".join(repr(v) for v in vals)}|'
        f'{"|".join(tabs)}'.encode()).hexdigest()[:16]
    path = os.path.join(_CACHE, f'{species}_{key}.npz')
    if cache and os.path.exists(path):
        d = np.load(path)
        return d['Ep'], d['dphi'], d['dk']

    jobs = [(species, i, float(E), nr, vals, entry)
            for i, E in enumerate(Ep_all)]
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
        out = list(ex.map(_one_energy, jobs, chunksize=1))
    dphi = np.array([o[0] for o in out])
    dk = np.array([o[1] for o in out])

    if cache:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez(path, Ep=Ep_all, dphi=dphi, dk=dk)
    return Ep_all, dphi, dk


if __name__ == '__main__':
    for sp in ('proton', 'carbon'):
        Ep, dphi, dk = bias_curves(sp)
        for lab, v in (('dPhi', dphi), ('dK', dk)):
            print(f'{sp:7s} {lab}: mean|.| {np.abs(v).mean():5.2f} %   '
                  f'median {np.median(np.abs(v)):5.2f} %   '
                  f'max {np.abs(v).max():5.2f} % at {Ep[np.abs(v).argmax()]:.0f}   '
                  f'outside +-40 %: {np.sum(np.abs(v) > 40)}/{len(v)}')
