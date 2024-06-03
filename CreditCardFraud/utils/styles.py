from cycler import cycler
import matplotlib as mpl

cmap_petroff = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

style_dict = {
    "axes.grid": True,
    #"axes.prop_cycle": cycler("color", cmap_petroff),
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros",
    "mathtext.bf": "TeX Gyre Heros:bold",
    "mathtext.sf": "TeX Gyre Heros",
    "mathtext.it": "TeX Gyre Heros:italic",
    "mathtext.tt": "TeX Gyre Heros",
    "mathtext.cal": "TeX Gyre Heros",
    "mathtext.default": "regular",
    "figure.figsize": (10.0, 10.0),
    "font.size": 20,
    "savefig.bbox" : "tight",
    "axes.labelsize": "12",
    "axes.unicode_minus": False,
    "xtick.labelsize": "12",
    "ytick.labelsize": "12",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "grid.color": "#b0b0b0",
    "axes.linewidth": 2,
    "savefig.transparent": False,
    "xaxis.labellocation": "center",
    "yaxis.labellocation": "center",
}

# Filter extra (labellocation) items if needed
style_dict = {k: v for k, v in style_dict.items() if k in mpl.rcParams}
