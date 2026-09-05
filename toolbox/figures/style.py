"""Shared plotting style for the manuscript figures.

Every figure generator calls `apply()` first and takes its sizes from
`plt.rcParams` rather than hardcoding them, so a change here moves all figures
together. Sizes are those of the Phi/K comparison figures, which are the
reference for label and tick size.

Colours are per REGIME and fixed, so a regime keeps its colour across figures.
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

# regime colours -- keep these stable across every figure
CAS = '#4C72B0'      # blue
EV  = '#C44E52'      # red
SLOW = '#55A868'     # green (epithermal + thermal share a kernel)
EXTRA = ['#8172B2', '#CCB974', '#64B5CD']

# species colours, for figures that compare proton against carbon
PROTON = CAS      # blue
CARBON = EV       # red

SIZES = dict(
    axes_titlesize=23,
    axes_labelsize=21,
    legend_fontsize=21,
    tick_labelsize=19,
)


def has_times():
    return any('Times New Roman' == f.name for f in font_manager.fontManager.ttflist)


def apply():
    """Times New Roman serif, stix math, the reference sizes, inward ticks."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman',
                                  'Liberation Serif', 'DejaVu Serif']
    plt.rcParams.update({
        'mathtext.fontset': 'stix',
        'axes.titlesize': SIZES['axes_titlesize'],
        'axes.labelsize': SIZES['axes_labelsize'],
        'legend.fontsize': SIZES['legend_fontsize'],
        'xtick.labelsize': SIZES['tick_labelsize'],
        'ytick.labelsize': SIZES['tick_labelsize'],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.major.pad': 8,
        'ytick.major.pad': 6,
        'axes.prop_cycle': plt.cycler('color', [CAS, EV, SLOW] + EXTRA),
        'axes.linewidth': 0.9,
        'lines.linewidth': 2.2,
    })


# the E_0 = ... annotation size, as used by the Phi/K figures
def annot_size():
    return plt.rcParams['xtick.labelsize']


# ---------------------------------------------------------------- layout
# Two-panel figures (2 and 7) share one geometry so they sit the same on the
# page: same canvas, same central gap, same margins, and one legend in a band
# across the top rather than a legend inside each panel.
FIGSIZE_2P = (15.5, 6)
GAP = 0.20          # central gap, as a width_ratio against two panels of 1.0
MARGINS = dict(left=0.07, right=0.98, bottom=0.16, top=0.82)


def two_panel():
    """Return (fig, ax_left, ax_right) with the shared two-panel geometry."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=FIGSIZE_2P)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, GAP, 1.0], wspace=0.0)
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])


def top_legend(fig, ax, ncol):
    """One legend in the band above both panels."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(**MARGINS)
