"""Shared MC | AM | |Delta| map layout."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from toolbox.figures import style

WIDTH = 15.5
PANEL_ASPECT = 1.08
SPECIES_GAP = 0.15
CBAR_RATIO = 0.05
MARGINS = dict(left=0.055, right=0.895, bottom=0.11, top=0.90)


def figure_height(nrows):
    avail_w = WIDTH * (MARGINS['right'] - MARGINS['left'])
    panel_w = avail_w / (2 + SPECIES_GAP) / (3 + CBAR_RATIO)
    return nrows * panel_w * PANEL_ASPECT / (MARGINS['top'] - MARGINS['bottom'])


def mirror(rho, A):
    if np.isclose(rho[0], 0.0):
        return (np.concatenate([-rho[:0:-1], rho]),
                np.concatenate([A[:, 1:][:, ::-1], A], axis=1))
    return (np.concatenate([-rho[::-1], rho]),
            np.concatenate([A[:, ::-1], A], axis=1))


def _ticks_white(ax):
    ax.tick_params(direction='in', which='both', top=True, right=True,
                   colors='white', labelcolor='black')
    for s in ax.spines.values():
        s.set_edgecolor('white')


def draw(blocks, z, rho, quantity=None, exp_per_row=True, rho_tick_step=2):
    style.apply()
    nrows = len(blocks[0]['rows'])
    fig = plt.figure(figsize=(WIDTH, figure_height(nrows)))
    gs_outer = gridspec.GridSpec(1, 2, wspace=SPECIES_GAP)
    cols = [r'$\mathrm{MC}$', r'$\mathrm{AM}$', r'$|\Delta|$']

    first_ax, cbar_ax = {}, {}
    for si, blk in enumerate(blocks):
        gs = gridspec.GridSpecFromSubplotSpec(
            nrows, 4, subplot_spec=gs_outer[si], wspace=0.0, hspace=0.0,
            width_ratios=[1, 1, 1, CBAR_RATIO])

        block_max = max(float(np.nanmax([mc, am]))
                        for _, mc, am in blk['rows'])
        block_exp = int(np.floor(np.log10(block_max))) if block_max > 0 else 0

        cbar_ax[si] = []
        for i, (row_label, mc2d, am2d) in enumerate(blk['rows']):
            rho_m, mc = mirror(rho, mc2d)
            _, am = mirror(rho, am2d)
            dif = np.abs(am - mc)
            vmax = float(np.nanmax([mc, am, dif]))
            norm = Normalize(0, vmax if vmax > 0 else 1)
            ext = [rho_m.min(), rho_m.max(), z.min(), z.max()]

            for j, dat in enumerate((mc, am, dif)):
                ax = fig.add_subplot(gs[i, j])
                if i == 0 and j == 0:
                    first_ax[si] = ax
                im = ax.imshow(dat, origin='lower', aspect='auto',
                               extent=ext, norm=norm)
                _ticks_white(ax)

                half = np.arange(0, np.floor(rho_m.max()) + 1, rho_tick_step)
                xt = np.concatenate([-half[:0:-1], half])
                ax.set_xticks(xt)

                ax.set_xticklabels([f'{abs(int(round(t)))}' for t in xt])
                ax.set_yticks(np.arange(np.ceil(z.min() / 10) * 10,
                                        np.floor(z.max() / 10) * 10 + 1, 10))
                if j == 0 and si == 0:
                    ax.set_ylabel(r'$z$ (cm)')
                else:
                    ax.tick_params(labelleft=False)
                if j == 0:
                    ax.text(0.05, 0.95, row_label, color='white',
                            transform=ax.transAxes, ha='left', va='top',
                            fontsize=style.annot_size())
                if i == nrows - 1:
                    ax.set_xlabel(r'$\rho$ (cm)')
                else:
                    ax.tick_params(labelbottom=False)
                if i == 0:
                    ax.set_title(cols[j])

            cax = fig.add_subplot(gs[i, 3])
            cbar_ax[si].append(cax)
            cb = fig.colorbar(im, cax=cax)
            e = block_exp if not exp_per_row else (
                int(np.floor(np.log10(vmax))) if vmax > 0 else 0)
            cb.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos, ee=e: f'{x / 10 ** ee:g}'))

            t = cb.get_ticks()
            lo, hi = norm.vmin, norm.vmax
            cb.set_ticks(t[(t > lo + 0.04 * (hi - lo)) &
                           (t < hi - 0.04 * (hi - lo))])
            cb.ax.tick_params(colors='white', labelcolor='black',
                              direction='in',
                              labelsize=style.SIZES['tick_labelsize'] - 4)
            for s in cb.ax.spines.values():
                s.set_edgecolor('white')
            cb.ax.yaxis.get_offset_text().set_visible(False)

            if exp_per_row and e != 0:

                cb.ax.set_ylabel(rf'$\times 10^{{{e}}}$', rotation=90,
                                 labelpad=2,
                                 fontsize=style.SIZES['tick_labelsize'] - 4)
            elif not exp_per_row and i == 0 and block_exp != 0:
                cb.ax.set_title(rf'$\times 10^{{{block_exp}}}$', pad=4,
                                fontsize=style.SIZES['tick_labelsize'] - 2)

    fig.subplots_adjust(**MARGINS)
    fig.canvas.draw()

    for si, blk in enumerate(blocks):
        left = first_ax[si].get_position().x0
        right = cbar_ax[si][0].get_position().x0
        top = first_ax[si].get_position().y1
        fig.text(0.5 * (left + right), top + 0.42 / fig.get_figheight(),
                 blk['header'], ha='center', va='bottom',
                 fontsize=plt.rcParams['axes.titlesize'])

    if quantity is not None:
        p_top = cbar_ax[1][0].get_position()
        p_bot = cbar_ax[1][-1].get_position()

        dx = 0.058 if exp_per_row else 0.034
        fig.text(p_top.x1 + dx, 0.5 * (p_top.y1 + p_bot.y0), quantity,
                 rotation=90, ha='left', va='center',
                 fontsize=plt.rcParams['axes.labelsize'])
    return fig


def save(fig, name, out_dir):
    return style.save(fig, name, out_dir)
