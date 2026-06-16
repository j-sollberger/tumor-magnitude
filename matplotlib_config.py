import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

doc_textwidth = 5.126  # inches
doc_fontsize = 10


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
    "font.size": doc_fontsize,
    "axes.labelsize": doc_fontsize,
    "axes.titlesize": doc_fontsize,
    "figure.titlesize": doc_fontsize,
    "font.family": "lmodern",
    "xtick.labelsize": doc_fontsize - 2,
    "ytick.labelsize": doc_fontsize - 2,
    "legend.fontsize": doc_fontsize - 2,
    "legend.borderpad": 0.6,
}

plt.rcParams.update(params)
