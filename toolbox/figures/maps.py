"""Shared MC | AM | |Delta| map layout for figures 4, 5 and 6.

The three figures are the same object with a different row axis: figure 4 puts
four NEUTRON energies at one primary energy, figures 5 and 6 put three PRIMARY
energies of one integrated quantity. They were three separate pieces of drawing
code with three sets of hardcoded font sizes; this module is the one layout, and
the generators supply only data and labels.

Geometry
--------
Panel size is derived, not guessed. The canvas width is fixed at the width the
other manuscript figures use; the panel aspect PANEL_ASPECT then FIXES the
canvas height for a given row count:

    block   = avail_w / (2 + SPECIES_GAP)          two species blocks + gap
    panel_w = block / (3 + CBAR_RATIO)             MC | AM | |D| + colourbar
    H       = nrows panel_w PANEL_ASPECT / (top - bottom)

so every figure built here has panels of the same shape whatever its row count.

Colour norms are per ROW and shared by that row's three panels, so MC, AM and
|Delta| are directly comparable; |Delta| is on the same scale as the two it is
the difference of, which is the point of the column.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from toolbox.figures import style

WIDTH = 15.5              # canvas width, as figures 2, 3 and 7
PANEL_ASPECT = 1.08       # panel height / panel width -- slightly taller than wide
SPECIES_GAP = 0.15        # gap between the species blocks, as a GridSpec wspace
CBAR_RATIO = 0.05         # colourbar width, as a width_ratio against one panel
MARGINS = dict(left=0.055, right=0.895, bottom=0.11, top=0.90)


def figure_height(nrows):
    avail_w = WIDTH * (MARGINS['right'] - MARGINS['left'])
    panel_w = avail_w / (2 + SPECIES_GAP) / (3 + CBAR_RATIO)
    return nrows * panel_w * PANEL_ASPECT / (MARGINS['top'] - MARGINS['bottom'])


def mirror(rho, A):
    """(nz, nr) -> (nz, 2 nr - 1), rho reflected about the beam axis."""
    if np.isclose(rho[0], 0.0):
        return (np.concatenate([-rho[:0:-1], rho]),
                np.concatenate([A[:, 1:][:, ::-1], A], axis=1))
    return (np.concatenate([-rho[::-1], rho]),
            np.concatenate([A[:, ::-1], A], axis=1))


def _ticks_white(ax):
    """Ticks and spines in white: they sit ON the image, not beside it."""
    ax.tick_params(direction='in', which='both', top=True, right=True,
                   colors='white', labelcolor='black')
    for s in ax.spines.values():
        s.set_edgecolor('white')


def draw(blocks, z, rho, quantity=None, exp_per_row=True, rho_tick_step=2):
    """Build the figure.

    blocks   two dicts, proton then carbon, each
             {'header': str, 'rows': [(row_label, mc2d, am2d), ...]}
             with the 2d arrays shaped (nz, nr), unmirrored.
    quantity a rotated axis label for the colourbars, drawn once to the right of
             the carbon block. None to omit it.
    exp_per_row  True  -- each row's colourbar carries its own x10^e, as its
                 own right-hand label. Figure 4 needs this: its rows span three
                 decades. False -- one exponent per species block, as a small
                 title over the top colourbar, which is what figures 5 and 6
                 want and what keeps the colourbars uncluttered.
    """
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

        # one exponent for the whole block, when the rows share a scale
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

            # NOT sharex/sharey: shared axes also share the tick formatter, so
            # the last set_xticklabels in the row would win and the per-panel
            # blanking below would collapse onto one pattern. All three panels
            # are drawn with the same extent, so their limits match anyway.
            for j, dat in enumerate((mc, am, dif)):
                ax = fig.add_subplot(gs[i, j])
                if i == 0 and j == 0:
                    first_ax[si] = ax
                im = ax.imshow(dat, origin='lower', aspect='auto',
                               extent=ext, norm=norm)
                _ticks_white(ax)
                # Ticks run the full mirrored range, out to the last whole
                # centimetre inside rho_max -- the panels show all 5.5 cm and
                # the axis has to say so. The panels touch, though, so a label
                # on an edge SHARED with the next panel collides with that
                # panel's; those are blanked, and only the outer edges of the
                # block keep theirs. Blanking the label rather than dropping the
                # tick keeps the tick marks even where the number goes.
                xt = np.arange(np.ceil(rho_m.min()),
                               np.floor(rho_m.max()) + 1, rho_tick_step)
                edge = 0.08 * (rho_m.max() - rho_m.min())
                ax.set_xticks(xt)
                # rho is mirrored, so both halves are labelled with |rho|
                ax.set_xticklabels(
                    ['' if (j > 0 and t < rho_m.min() + edge)
                     or (j < 2 and t > rho_m.max() - edge)
                     else f'{abs(int(round(t)))}' for t in xt])
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
            # drop the 0 tick: it sits on the row boundary and collides with the
            # top tick of the colourbar below
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
            # 10^0 is not a factor; printing it is noise
            if exp_per_row and e != 0:
                # to the RIGHT of the bar, not above it: rows are hspace=0, so a
                # title would land on the panel above
                cb.ax.set_ylabel(rf'$\times 10^{{{e}}}$', rotation=90,
                                 labelpad=2,
                                 fontsize=style.SIZES['tick_labelsize'] - 4)
            elif not exp_per_row and i == 0 and block_exp != 0:
                cb.ax.set_title(rf'$\times 10^{{{block_exp}}}$', pad=4,
                                fontsize=style.SIZES['tick_labelsize'] - 2)

    fig.subplots_adjust(**MARGINS)
    fig.canvas.draw()

    # Species headers, centred over their block: from the left edge of its MC
    # panel to the right edge of its |Delta| panel, so the colourbar column does
    # not pull the header off centre.
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
        # per-row exponents already occupy the strip right of the colourbars,
        # so the quantity label has to clear them as well as the tick labels
        dx = 0.058 if exp_per_row else 0.034
        fig.text(p_top.x1 + dx, 0.5 * (p_top.y1 + p_bot.y0), quantity,
                 rotation=90, ha='left', va='center',
                 fontsize=plt.rcParams['axes.labelsize'])
    return fig


def save(fig, name, out_dir):
    """Save at the canvas size, as style.save does -- see the note there."""
    return style.save(fig, name, out_dir)
