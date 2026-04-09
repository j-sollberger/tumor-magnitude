import os
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from data_preparation_precomputation import *
from matplotlib_config import *

### Methods ###


def give_vector_all_comb_magnitudes(
    magnitude_directory: str, scales: list, combs: list
) -> list:
    """Gives the feature vector that consists of all magnitudes of celltype combinations listed in comb and for all scaling factors listed in scales.

    Args:
        magnitude_directory (str): where the precomputed magnitudes are stored
        scales (list): scaling factors for magnitudes
        combs (list): for each combination of celltypes in this list, the magnitude of the union of the respective point (cell) clouds is taken at all the scales in scales

    Returns:
        list: feature vector
    """
    # load magnitudes
    magnitudes = [
        pd.read_csv(os.path.join(magnitude_directory, f"magnitudesScale{s}.csv"))
        for s in scales
    ]

    # create vectors
    vectors = []
    for k in range(magnitudes[0].shape[0]):
        vector = []
        for md in magnitudes:
            for t in combs:
                vector.append(md.loc[k, ",".join([t[i] for i in range(len(t))])])
        vectors.append(vector)

    return vectors


def give_vector_magnitude_differences(
    magnitude_directory: str, scales: list, combs: list
) -> list:
    """Gives the feature vector that consists of all possible magnitude inclusion-exclusion-type differences of cell-type combinations listed in comb and for all scaling factors listed in scales. Example: for the combination ['T','M','N'], the three differences |T,M|+|N|-|T,M,N|, |T,N|+|M|-|T,M,N| and |M,N|+|T|-|T,M,N| are entries of the feature vector for as many scales as are listed in the input. For single cell-types ['S'] in comb, simply its magnitude |S| is added to the feature vector (at all scales in scales).

    Args:
        magnitude_directory (str): where precomputed magnitudes are stored
        scales (list): scaling factors for magnitudes
        combs (list): for each combination of celltypes in this list, all respective inclusion-exclusion-type differences of the magnitude of respective (unions of) point (cell) clouds are added to the feature vector for all the scales in scales

    Returns:
        list: feature vector
    """
    # load magnitudes
    magnitudes = [
        pd.read_csv(os.path.join(magnitude_directory, f"magnitudesScale{s}.csv"))
        for s in scales
    ]

    # create vectors
    vectors = []
    for fileindex in range(magnitudes[0].shape[0]):
        vector = []
        for md in magnitudes:
            for t in combs:
                vector = (
                    vector
                    + [
                        md.loc[fileindex, ",".join([t[i] for i in subset])]
                        + md.loc[
                            fileindex,
                            ",".join([t[i] for i in range(len(t)) if i not in subset]),
                        ]
                        - md.loc[fileindex, ",".join(t)]
                        for k in range(1, len(t))
                        for subset in combinations(range((len(t) + 1) // 2), k)
                    ]
                    if len(t) > 1
                    else [md.loc[fileindex, t[0]]]
                )
        vectors.append(vector)

    return vectors


### Pipeline ###


def align_labels(classification: list, gt: list) -> list:
    """assigns each cluster (0, 1 and 2) from the given classification to one of the 3 'E's (elimination: 0, equilibrium: 1, escape: 2) such that the classification matches the ground truth the closest.

    Args:
        classification (list): assigned cluster (0, 1 or 2) for each parameter scheme in lexicographic order
        gt (list): contains some ground truth. That is, for every parameter scheme the value 0 for elimination, the value 1 for equilibrium or the value 2 for escape

    Returns:
        list: assignment (0 = elimination, 1 = equilibrium, 2 = escape) for each parameter scheme in lexicographic order
    """
    # link classification to ground truth
    # possible matchings:
    options = [list(p) for p in permutations(range(3))]
    # find best one
    concordance = [
        sum(
            [options[k][classification[i]] == gt[i] for i in range(len(classification))]
        )
        for k in range(len(options))
    ]
    link = options[np.argmax(np.array(concordance))]
    # modify classification accordingly
    classification_aligned = [
        link[classification[i]] for i in range(len(classification))
    ]
    return classification_aligned


def return_classification(
    vector: list, number_of_clusters: int, paramdf_reduced: pd.DataFrame
) -> tuple:
    """Returns a classification of parameter schemes based on a k-means clustering of all simulations.

    Args:
        vector (list): contains a vector for each simulation, these are clustered
        number_of_clusters (int): gives the k parameter in kmeans clustering
        paramdf (pd.DataFrame): dataframe with columns 'filenr', 'chi_macrophageToCSF' and '	halfMaximalExtravasationCsf1Conc'

    Returns:
        tuple: classification (list), labels (Series), diff_assigned (list), numb_of_sim (list)
    """

    # cluster
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(vector)
    labels = kmeans.labels_

    # read parameter info
    chi_macrophageToCSF = paramdf_reduced["chi_macrophageToCSF"].to_numpy()
    halfMaximalExtravasationCsf1Conc = paramdf_reduced[
        "halfMaximalExtravasationCsf1Conc"
    ].to_numpy()

    # classify
    classification = []
    diff_assigned = []
    numb_of_sim = []

    for chi in [i / 2 for i in range(1, 10)]:

        for Csf1 in [i / 10 for i in range(1, 10)]:

            indices = np.where(
                (chi_macrophageToCSF == chi)
                & (halfMaximalExtravasationCsf1Conc == Csf1)
            )

            numb_of_sim.append(np.array(indices).size)
            if np.array(indices).size == 0:
                continue

            z = np.bincount(labels[indices]).argmax()
            diff_assigned.append(
                len(labels[indices]) - len(np.where(labels[indices] == z)[0])
            )
            classification.append(z)

    return classification, labels, diff_assigned, numb_of_sim


def classify_schemes(vector: list, paramdf_reduced: pd.DataFrame, gt: list) -> tuple:
    """Classifying parameter schemes into the three E's given a list of vectors, one for each simulation, and returning a score.

    Args:
        vector (list): contains a vector for each simulation, these are clustered
        paramdf (pd.DataFrame): dataframe with columns 'filenr', 'chi_macrophageToCSF' and '	halfMaximalExtravasationCsf1Conc'
        gt (list): contains ground truth. That is, for every parameter scheme in lexicographic order a list like [0,0,-1] judging the match of the parameter scheme to elimination, equilibrium, escape (in that order). 1 means: clear good match (e.g. this scheme is clearly elimination), -1 means: clear bad match (e.g. this scheme is definitely not escape), 0 means: neither obviously good, nor bad (e.g. a good part of the simulations in this scheme could be assigned to equilibrium, but certainly not all of them)

    Returns:
        tuple: classification (list), labels (Series), diff_assigned (list), numb_of_sim (list), score (float)
    """
    classification, labels, diff_assigned, numb_of_sim = return_classification(
        vector, 3, paramdf_reduced
    )

    classification_aligned = align_labels(classification, gt)

    return classification_aligned, labels, diff_assigned, numb_of_sim


### Plotting functions ###


def plot_classification_with_purities(
    result_directory: str,
    classification: list,
    purities: list,
    plot_title: str,
    save_title: str,
) -> None:
    """Plots a given classification of parameter schemes with the corresponding purities.

    Args:
        result_directory (str): where the plots should be saved to
        classification (list): given classification of parameter schemes
        purities (list): given classification purity for each parameter scheme (value between 0.5 and 1)
        plot_title (str): title of the plot
        save_title (str): under which name the plot should be saved
    """
    # predefine colors and parameter schemes
    mapping = {0: ("blue", "o"), 1: ("yellow", "o"), 2: ("red", "o")}
    schemes = np.array(
        [
            [c12, chi]
            for chi in [i / 2 for i in range(1, 10)]
            for c12 in [i / 10 for i in range(1, 10)]
        ]
    ).T

    # visualize the classification
    show_labels = np.array([schemes[0], schemes[1], classification]).T
    show_purity = np.array([schemes[0], schemes[1], purities]).T

    label_map = {0: "Elimination", 1: "Equilibrium", 2: "Escape"}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=False, layout="constrained", figsize=(15 / 2.54, 7.5 / 2.54)
    )

    for c in np.unique(show_labels[:, 2]):
        d = show_labels[show_labels[:, 2] == c]
        ax1.scatter(
            d[:, 0],
            d[:, 1],
            s=135,
            c=mapping[c][0],
            marker=mapping[c][1],
            label=label_map[c],
        )
    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        ncols=1,
        labelspacing=1.5,
        handlelength=1,
        fontsize=8,
    )
    ax1.set_box_aspect(1)  # makes it square
    ax1.set_xlabel("$c_{1/2}$")
    ax1.set_ylabel(r"$\chi_c^m$", rotation=0, labelpad=10)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_major_locator(MultipleLocator(0.2, 0.1))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_major_locator(MultipleLocator(1, 0.5))
    ax1.set_xmargin(0.07)
    ax1.set_ymargin(0.07)
    ax1.set_title("(a) Classification")

    im = ax2.scatter(
        show_purity[:, 0],
        show_purity[:, 1],
        s=135,
        c=show_purity[:, 2],
        marker="o",
        vmin=0,
        vmax=1,
    )
    ax2.set_box_aspect(1)  # makes it square
    ax2.set_xlabel("$c_{1/2}$")
    ax2.set_ylabel(r"$\chi_c^m$", rotation=0, labelpad=10)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.xaxis.set_major_locator(MultipleLocator(0.2, 0.1))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(MultipleLocator(1, 0.5))
    ax2.set_xmargin(0.07)
    ax2.set_ymargin(0.07)
    ax2.set_title("(b) Purities")
    fig.colorbar(im, ax=ax2, shrink=0.65)

    fig.suptitle(plot_title, y=0.95, fontsize=11)

    plt.savefig(
        os.path.join(result_directory, save_title), format="pdf", bbox_inches="tight"
    )
    return


def reduced_plot_classification_with_purities(
    result_directory: str,
    classification: list,
    purities: list,
    plot_title: str,
    save_title: str,
) -> None:
    """Plots a given classification of parameter schemes with the corresponding purities. Reduced version (less labels, less space)

    Args:
        result_directory (str): where the plots should be saved to
        classification (list): given classification of parameter schemes
        purities (list): given classification purity for each parameter scheme (value between 0.5 and 1)
        plot_title (str): title of the plot
        save_title (str): under which name the plot should be saved
    """
    # predefine colors and parameter schemes
    mapping = {0: ("blue", "o"), 1: ("yellow", "o"), 2: ("red", "o"), 3: ("green", "o")}
    schemes = np.array(
        [
            [Csf1, chi]
            for chi in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
            for Csf1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ]
    ).T

    # visualize the classification
    show_labels = np.array([schemes[0], schemes[1], classification]).T
    show_purity = np.array([schemes[0], schemes[1], purities]).T

    label_map = {0: "Elim.", 1: "Equi.", 2: "Esc."}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=False, layout="constrained", figsize=(2.4, 1.4)
    )

    for c in np.unique(show_labels[:, 2]):
        d = show_labels[show_labels[:, 2] == c]
        ax1.scatter(
            d[:, 0],
            d[:, 1],
            s=60,
            c=mapping[c][0],
            marker=mapping[c][1],
            label=label_map[c],
        )
    ax1.set_box_aspect(1)  # makes it square
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xmargin(0.07)
    ax1.set_ymargin(0.07)

    ax2.scatter(
        show_purity[:, 0],
        show_purity[:, 1],
        s=60,
        c=show_purity[:, 2],
        marker="o",
        vmin=0,
        vmax=1,
    )
    ax2.set_box_aspect(1)  # makes it square
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xmargin(0.07)
    ax2.set_ymargin(0.07)

    fig.suptitle(plot_title, fontsize="x-small")  # y=0.95

    plt.savefig(
        os.path.join(result_directory, "reduced-" + save_title),
        format="pdf",
        bbox_inches="tight",
    )
    return
