import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return (i / inch for i in tupl[0])
    return tuple(i / inch for i in tupl)


### colors ###
# color palette for the local neighbourhood classes
cln = [
    "#E5E5E5",
    "#E69F00",
    "#0072B2",
    "#009E73",
    "#F0E442",
    "#CC79A7",
    "#56B4E9",
    "#000000",
    "#7F7F7F",
    "#e41a1c",
    "#984ea3",
]

# color map for the cell types in simulation outcomes
cellcolor_map = {
    "Vessel": "grey",
    "Tumour": "#010E71",
    "Macrophage": "#D55E00",
    "Necrotic": "#BCBD22",
}

params = {
    "text.usetex": True,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "font.family": "lmodern",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.borderpad": 0.6,
}

plt.rcParams.update(params)
