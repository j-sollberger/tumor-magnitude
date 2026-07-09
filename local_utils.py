import math
import os
import random

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from tqdm import tqdm

from data_preparation_precomputation import *
from matplotlib_config import *


def prep_data(
    data_directory: str,
    local_magnitude_directory: str,
    N: int,
    compute_features: bool,
    *,
    magnitudes: bool = None,
    cellcounts: bool = None,
    distances: bool = None,
    scale: float = None,
) -> tuple:
    """Prepares an information dataframe with one row per simulation outcome and if desired a list of corresponding feature vectors.

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        local_magnitude_directory (str): where local magnitude data is stored
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
        compute_features (bool): determines whether feature vectors should be computed
        magnitudes (bool, optional): whether (simple) magnitudes should be used as features. Defaults to None.
        cellcounts (bool, optional): whether cellcounts should be used as features. Defaults to None.
        distances (bool, optional): whether average pairwise distances within each cell type should be used as features. Defaults to None.
        scale (float, optional): scaling factor for magnitude. Defaults to None.

    Raises:
        ValueError: if compute_features is True but one of magnitudes, cellcounts, distances is not passed, or if magnitudes is True and scale is not passed

    Returns:
        tuple: (all_info,) if compute_features is False and (all_info, list_of_vectors) if compute_features is True, where all_info is a pd.Dataframe with columns 'filenr', 'index1', 'index2', 'chi', 'c12' and with one row for each local neighbourhood for each simulation output, and where list_of_vectors contains a feature vector for each simulation output (row).
    """
    # create a data frame for all the information
    filenrs = collect_files(data_directory, range(1, 1621))
    paramdf = pd.read_csv(
        os.path.join(os.path.dirname(data_directory), "params_2ParamSweep.csv"),
        index_col=1,
    )
    bin_counts = pd.DataFrame(
        0, columns=[i / 10 for i in range(1, 10)], index=[i / 2 for i in range(1, 10)]
    )

    all_info = pd.DataFrame(
        {
            "filenr": [filenrs[k // (N * N)] for k in range(N * N * len(filenrs))],
            "index1": [(k // N) % N for k in range(N * N * len(filenrs))],
            "index2": [k % N for k in range(N * N * len(filenrs))],
        }
    )

    # add parameter info and count the number of simulation outcomes per parameter combination
    for filenr in filenrs:
        chi = paramdf.loc[filenr, "chi_macrophageToCSF"]
        c12 = paramdf.loc[filenr, "halfMaximalExtravasationCsf1Conc"]
        all_info.loc[all_info["filenr"] == filenr, "chi"] = chi
        all_info.loc[all_info["filenr"] == filenr, "c12"] = c12
        bin_counts.loc[chi, c12] += 1

    return_tuple = (all_info,)

    if compute_features:
        # ensure that required input is given:
        if (
            magnitudes is None
            or cellcounts is None
            or distances is None
            or (magnitudes is True and scale is None)
        ):
            raise ValueError("not all necessary input provided")
        # load pre-saved magnitudes / cellcounts if applicable
        if magnitudes:
            md = read_local_magnitudes(local_magnitude_directory, scale, N)
        if cellcounts:
            cd = read_local_cellcounts(local_magnitude_directory, N)
        if distances:
            dd = read_local_average_pairwise_distances(local_magnitude_directory, N)

        # calculate all local magnitude incl.-excl. difference vectors + all combination magnitudes
        list_of_vectors = []
        for k in tqdm(range(len(filenrs))):
            filenr = filenrs[k]

            for l in range(N * N):
                index = N * N * k + l
                i = all_info.loc[index, "index1"]
                j = all_info.loc[index, "index2"]

                vector = []

                if magnitudes:
                    vector = vector + [
                        md.loc[k, ",".join([str(i), str(j)] + t)]
                        for t in [["T"], ["M"], ["N"]]
                    ]
                if cellcounts:
                    vector = vector + [
                        cd.loc[k, ",".join([str(i), str(j)] + t)]
                        for t in [["T"], ["M"], ["N"]]
                    ]
                if distances:
                    vector = vector + [
                        dd.loc[k, ",".join([str(i), str(j)] + t)]
                        for t in [["T"], ["M"], ["N"]]
                    ]

                list_of_vectors.append(vector)

        return_tuple = return_tuple + (list_of_vectors,)

    return return_tuple


def plot_local_signatures(
    indices: list, labels: list, plot_title: str, colors: list
) -> None:
    """Given labels for NxN local balls, plots them as colored squares, each corresponding to the "local magnitude type" a local ball.

    Args:
        indices (list): of the form [[0,0],[0,1],...], that gives the order in which the labels are given
        labels (list): assigns to each local ball a "local magnitude type"
        plot_title (str): title of plot
        colors (list): list of colors
    """
    n = max(labels) + 1
    show_labels = np.array(
        [[indices[i][0], indices[i][1], labels[i]] for i in range(len(indices))]
    )
    for c in range(n):
        d = show_labels[show_labels[:, 2] == c]
        plt.scatter(d[:, 0], d[:, 1], s=17500 / (len(labels)), c=colors[c], marker="s")
    plt.title(plot_title)
    return


def plot_across_schemes_prep(info_frame: pd.DataFrame, seed: int) -> list:
    """Preparation to plot the local magnitude signature-assignments of one pseudo-randomly selected simulation outcome per parameter scheme. Randomly selects one outcome per each of the 9x9 schemes and gives the corresponding file number.

    Args:
        info_frame (pd.DataFrame): has columns "filenr", "chi" and "c12"
        seed (int): Seed for the pseudo random selection of simulation outcomes for display of results

    Returns:
        list: contains one filenr for each parameter scheme, in the form [[(chi=4.5,c12=0.1),...,(chi=4.5,c12=0.9)],...,[(chi=0.5,c12=0.1),...,(chi=0.5,c12=0.9)]]
    """
    random.seed(seed)
    display_filenrs = []
    for chi in [i / 2 for i in range(1, 10)]:
        sublist = []
        for c12 in [i / 10 for i in range(1, 10)]:
            subframe = info_frame.loc[info_frame["chi"] == chi]
            subframe = subframe.loc[subframe["c12"] == c12]
            subframe = subframe.reset_index(drop=True)
            bin_count = len(set(subframe["filenr"]))
            a = random.randint(0, 100) % bin_count
            sublist.append(subframe.loc[a, "filenr"])
        display_filenrs.append(sublist)
    display_filenrs.reverse()

    return display_filenrs


def plot_across_mxm_schemes(
    result_directory: str,
    result_name: str,
    all_info: pd.DataFrame,
    colors: list,
    N: int,
    seed: int,
    m: int = 9,
) -> None:
    """Plots the local magnitude signature-assignments of one pseudo-randomly (pre-)selected simulation outcome per parameter scheme, where a mxm (m=5 or m=9) subset of parameter schemes is used.

    Args:
        result_directory (str): where the plot should be saved to
        result_name (str): under what name the plot should be saved to
        all_info (pd.DataFrame): Contains the precomputed labels (signature assignments). Has columns 'filenr', 'index1', 'index2', 'chi', 'c12', 'label'
        colors (list): list of colors corresponding to the local signatures
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
        seed (int): determines pseudo-randomly chosen simulation outcome (per parameter scheme) whose corresponding results will be displayed
        m (int): supported values are only m=5 or m=9. Results in 5x5 or 9x9 displayed parameter schemes
    """
    if m != 5 and m != 9:
        print("value for m not supported")
        return
    info_frame = all_info.loc[
        [k * (N * N) for k in range(int(all_info.shape[0] / (N * N)))],
        ["filenr", "chi", "c12"],
    ]
    display_filenrs = plot_across_schemes_prep(info_frame, seed)
    indices1 = all_info.loc[0 : N * N, "index1"]
    indices2 = all_info.loc[0 : N * N, "index2"]
    # plot them
    fig, axs = plt.subplots(
        m,
        m,
        sharex=True,
        figsize=(
            (doc_textwidth, doc_textwidth)
            if m == 9
            else (0.48 * doc_textwidth, 0.48 * doc_textwidth)
        ),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    for i in range(9):
        for j in range(9):
            if m == 9 or (i % 2 == 0 and j % 2 == 0):
                display_labels = list(
                    all_info.loc[all_info["filenr"] == display_filenrs[i][j]]["label"]
                )

                # Visualize
                ax = axs[i // (8 // (m - 1)), j // (8 // (m - 1))]
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                n = max(display_labels) + 1
                show_labels = np.array(
                    [
                        [indices1[i], indices2[i], display_labels[i]]
                        for i in range(len(display_labels))
                    ]
                )
                for c in range(n):
                    d = show_labels[show_labels[:, 2] == c]
                    ax.scatter(
                        d[:, 0],
                        d[:, 1],
                        s=(doc_textwidth * N) ** 2 * 2.94 * (m - 2) / (1000 * m),
                        c=colors[c],
                        marker="s",
                    )

    gs = axs[0, 0].get_gridspec()
    ax_dummy = fig.add_subplot(gs[:, :], zorder=-1)
    ax_dummy.tick_params(length=0)
    ax_dummy.set_xticks(np.linspace(1 / (2 * m), 1 - 1 / (2 * m), m))
    ax_dummy.set_xticklabels([i / 10 for i in range(1, 10, 8 // (m - 1))], size=7)
    ax_dummy.set_yticks(np.linspace(1 / (2 * m), 1 - 1 / (2 * m), m))
    ax_dummy.set_yticklabels([i / 2 for i in range(1, 10, 8 // (m - 1))], size=7)
    ax_dummy.spines[:].set_visible(False)
    ax_dummy.set_xlabel(r"$c_{1/2}$", labelpad=8, ha="center", va="center")
    ax_dummy.set_ylabel(
        r"$\chi_c^m$", rotation=0, labelpad=10, ha="center", va="center"
    )

    plt.savefig(
        os.path.join(result_directory, result_name), format="pdf", bbox_inches="tight"
    )
    plt.close()
    return


def plot_across_schemes_display_sim_outcomes(
    data_directory: str, figure_directory: str, file_name: str, seed: int
) -> None:
    """Plots one pseudo-randomly chosen simulation output per parameter scheme.

    Args:
        data_directory (str): where the data is stored
        figure_directory (str): where the resulting figure should be saved
        file_name (str): under what name the figure should be saved
        seed (int): for the pseudo-random selection of displayed simulation outcomes
    """
    # check if result directory exists / create it
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    filenrs = collect_files(data_directory, range(1, 1621))
    paramdf = pd.read_csv(
        os.path.join(os.path.dirname(data_directory), "params_2ParamSweep.csv")
    )
    info_frame = paramdf.loc[paramdf["ID"].isin(filenrs)]
    info_frame = info_frame.reset_index(drop=True)
    info_frame = info_frame.rename(
        columns={
            "ID": "filenr",
            "chi_macrophageToCSF": "chi",
            "halfMaximalExtravasationCsf1Conc": "c12",
        }
    )
    display_filenrs = plot_across_schemes_prep(info_frame, seed)
    fig, axs = plt.subplots(
        9,
        9,
        sharex=True,
        figsize=(doc_textwidth, doc_textwidth),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    for i in range(9):
        for j in range(9):
            data = pd.read_csv(
                os.path.join(
                    data_directory,
                    f"ID-{display_filenrs[i][j]}_time-500_From2ParamSweep_Data.csv",
                )
            )
            ax = axs[i, j]
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            for label in cellcolor_map:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_box_aspect(1)
                mask = data["celltypes"] == label
                ax.scatter(
                    data.loc[mask, "points_x"],
                    data.loc[mask, "points_y"],
                    facecolors=data.loc[mask, "celltypes"].map(cellcolor_map),
                    label=label,
                    marker="o" if label != "Vessel" else "x",
                    s=0.25 if label != "Vessel" else 1,
                    edgecolors="none" if label != "Vessel" else None,
                    linewidths=0.5 if label == "Vessel" else None,
                )

    handles = axs[8, 1].get_legend_handles_labels()[0]
    fakehandle = Line2D([], [], linestyle="none")
    legend = fig.legend(
        [fakehandle] + handles,
        ["Cell Types: "] + list(cellcolor_map.keys()),
        loc="upper center",
        ncols=5,
        bbox_to_anchor=(0.5, 0.94),
        handlelength=0.1,
        borderpad=0.5,
        frameon=False,
    )
    for legobj in legend.legend_handles[1:5]:
        legobj.set_sizes([20])
    legend.legend_handles[1].set_linewidth([2])

    gs = axs[0, 0].get_gridspec()
    ax_dummy = fig.add_subplot(gs[:, :], zorder=-1)
    ax_dummy.tick_params(length=0)
    ax_dummy.set_xticks(np.linspace(1 / 18, 17 / 18, 9))
    ax_dummy.set_xticklabels([i / 10 for i in range(1, 10)], size=7)
    ax_dummy.set_yticks(np.linspace(1 / 18, 17 / 18, 9))
    ax_dummy.set_yticklabels([i / 2 for i in range(1, 10)], size=7)
    ax_dummy.spines[:].set_visible(False)
    ax_dummy.set_xlabel(r"$c_{1/2}$", labelpad=8, ha="center", va="center")
    ax_dummy.set_ylabel(
        r"$\chi_c^m$", rotation=0, labelpad=10, ha="center", va="center"
    )

    plt.savefig(
        os.path.join(figure_directory, file_name), format="pdf", bbox_inches="tight"
    )


def plot_averages_representatives(
    data_directory: str,
    result_directory: str,
    result_name: str,
    all_info: pd.DataFrame,
    list_of_vectors: list,
    number_of_signatures: int,
    colors: list,
    N: int,
    std: bool,
    *,
    permutation: list = None,
    ylabel: str = r"Magnitude $|\cdot|$",
) -> None:
    """Plots the average local magnitude features of each local signature + the closest representative local neighbourhood.

    Args:
        data_directory (str): where the data (simulation outcomes) are stored
        result_directory (str): where the plot should be saved to
        result_name (str): under what name the plot should be saved
        all_info (pd.DataFrame): contains the precomputed labels (signature assignments) of all simulations. Has columns 'filenr', 'index1', 'index2', 'chi', 'c12', 'label'
        list_of_vectors (list): contains the precomputed local features that were clustered to obtain the labels in all_info (same order)
        number_of_signatures (int): number of local signatures
        colors (list): list of colors corresponding to the labels (local signatures)
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
        std (bool): determines if for the averages the standard deviations should be plotted or not
        permutation (list, optional): permutes the order in which the local signatures are displayed. Defaults to None.
        ylabel (str, optional): typically corresponds to the the used method for features. Defaults to r'Magnitude $|\cdot|$'.
    """
    cln_positions = {c: i for i, c in enumerate(cln)}
    perm = [sorted(colors, key=lambda c: cln_positions[c]).index(c) for c in colors]
    if permutation is not None:
        perm = [perm[permutation[i]] for i in range(len(perm))]
    averages = []
    std = []
    representatives = []
    # calculate the averages
    for i in range(number_of_signatures):
        subframe = all_info.loc[all_info["label"] == i]
        indices = list(subframe.index)
        averages.append(
            [
                sum([list_of_vectors[j][k] for j in indices]) / len(indices)
                for k in range(len(list_of_vectors[0]))
            ]
        )
        std.append(
            [
                np.std([list_of_vectors[j][k] for j in indices])
                for k in range(len(list_of_vectors[0]))
            ]
        )
        # find representative
        distances = [
            np.linalg.norm(np.array(averages[i]) - np.array(list_of_vectors[j]))
            for j in indices
        ]
        representatives.append(indices[np.argmin(distances)])

    # plot the averages
    magnitudes_of_centroids = {
        "Tumour": [
            [averages[perm.index(i)][0] + 0.0001 for i in range(number_of_signatures)],
            [std[perm.index(i)][0] for i in range(number_of_signatures)],
        ],
        "Macrophage": [
            [averages[perm.index(i)][1] + 0.0001 for i in range(number_of_signatures)],
            [std[perm.index(i)][1] for i in range(number_of_signatures)],
        ],
        "Necrotic": [
            [averages[perm.index(i)][2] + 0.0001 for i in range(number_of_signatures)],
            [std[perm.index(i)][2] for i in range(number_of_signatures)],
        ],
    }

    x = np.arange(number_of_signatures)
    width = 0.25
    multiplier = 0

    fig = plt.figure(
        constrained_layout=True, figsize=(doc_textwidth, doc_textwidth * 1.05)
    )
    gs = GridSpec(7, number_of_signatures, figure=fig)
    ax = fig.add_subplot(gs[0:4, :])

    for attribute, measurement in magnitudes_of_centroids.items():
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            measurement[0],
            0.2,
            label=attribute,
            color=cellcolor_map[attribute],
            alpha=0.8,
        )
        if std:
            ax.errorbar(
                x=x + offset,
                y=measurement[0],
                yerr=measurement[1],
                fmt="none",
                color="r",
                elinewidth=1,
                capsize=2,
            )
        else:
            ax.bar_label(
                rects,
                labels=[r"$|$" + attribute[0] + r"$|$"] * number_of_signatures,
                padding=3,
            )
        multiplier += 1

    ax.set_ylabel(ylabel)
    ax.set_title(rf"\textbf{{Average  {ylabel}s by Local Neighbourhood Class}}")
    ax.set_xticks(x + width, [r"$\bullet$"] * number_of_signatures)
    ax.tick_params(axis="x", pad=0)
    for i in range(number_of_signatures):
        label = ax.get_xticklabels()[i]
        label.set_fontsize(doc_fontsize * 2.5)
        label.set_color(colors[perm.index(i)])
    ax.legend(loc="upper left", ncols=3)

    for k in range(number_of_signatures):
        ax = fig.add_subplot(gs[6, perm[k]])
        filenr = all_info.loc[representatives[k], "filenr"]
        i = all_info.loc[representatives[k], "index1"]
        j = all_info.loc[representatives[k], "index2"]
        data = pd.read_csv(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            ),
            index_col=0,
        )
        d = 25 / float(N)
        x0 = d + 2 * i * d
        y0 = d + 2 * j * d
        ball = give_ball(data, x0, y0, math.sqrt(2) * 25 / float(N))
        for label in ["Tumour", "Macrophage", "Necrotic"]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            mask = ball["celltypes"] == label
            ax.scatter(
                ball.loc[mask, "points_x"],
                ball.loc[mask, "points_y"],
                c=ball.loc[mask, "celltypes"].map(cellcolor_map),
                label=label,
                marker=".",
                s=10,
            )
        ax.set_facecolor("lightgray")
        circle = plt.Circle(
            (x0, y0), math.sqrt(2) * 25 / float(N), color="white", fill=True, zorder=0
        )
        ax.add_patch(circle)
        ax.set_xlabel(
            r"$\bullet$", color=colors[k], fontsize=doc_fontsize * 2.5, labelpad=0
        )
        ax = fig.add_subplot(gs[5, perm[k]])
        for label in cellcolor_map:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            mask = data["celltypes"] == label
            ax.scatter(
                data.loc[mask, "points_x"],
                data.loc[mask, "points_y"],
                c=data.loc[mask, "celltypes"].map(cellcolor_map),
                label=label,
                marker=".",
                s=1,
                alpha=0.3,
            )
            mask = ball["celltypes"] == label
            ax.scatter(
                ball.loc[mask, "points_x"],
                ball.loc[mask, "points_y"],
                c=ball.loc[mask, "celltypes"].map(cellcolor_map),
                label=label,
                marker=".",
                s=1,
            )
            circle = plt.Circle(
                (x0, y0), math.sqrt(2) * 25 / float(N), color="red", fill=False
            )
            ax.add_patch(circle)

    ax_group = fig.add_subplot(gs[5:7, :])
    ax_group.axis("off")
    ax_group.set_title(r"\textbf{Typical Representatives}")

    ax_group = fig.add_subplot(gs[4, :])
    ax_group.axis("off")

    ax = fig.add_subplot(gs[6, :])
    ax.set_xlabel("Local Magnitude Neighbourhood Classes", labelpad=20)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(
        os.path.join(result_directory, result_name), format="pdf", bbox_inches="tight"
    )

    plt.close()
    return
