"""Shared plotting style for the manuscript figures."""
import matplotlib.pyplot as plt
from matplotlib import font_manager


CAS = '#4C72B0'
EV  = '#C44E52'
SLOW = '#55A868'
EXTRA = ['#8172B2', '#CCB974', '#64B5CD']


PROTON = CAS
CARBON = EV

SIZES = dict(
    axes_titlesize=23,
    axes_labelsize=21,
    legend_fontsize=21,
    tick_labelsize=19,
)


def has_times():
    return any('Times New Roman' == f.name for f in font_manager.fontManager.ttflist)


def apply():
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


def annot_size():
    return plt.rcParams['xtick.labelsize']


FIGSIZE_2P = (15.5, 6)
GAP = 0.20
MARGINS = dict(left=0.07, right=0.98, bottom=0.16, top=0.82)


def two_panel():
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    fig = plt.figure(figsize=FIGSIZE_2P)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, GAP, 1.0], wspace=0.0)
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])


def top_legend(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(**MARGINS)


def boxed_legend(ax, loc='best', **kw):
    leg = ax.legend(loc=loc, frameon=True, fancybox=False, framealpha=1.0,
                    facecolor='white', edgecolor='black',
                    borderpad=0.5, labelspacing=0.4, handlelength=1.6,
                    handletextpad=0.6, **kw)
    leg.get_frame().set_linewidth(plt.rcParams['axes.linewidth'])
    return leg


def save(fig, name, out_dir):
    import os
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'),
                    dpi=150 if ext == 'png' else None)
