from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from local_utils import *
from matplotlib_config import *

BASE_DIR = Path(__file__).resolve().parent

### manual settings ###
seed = 73
N = 8
RESULT_DIR = os.path.join(BASE_DIR, "Results/local")
DATA_DIR = os.path.join(BASE_DIR, "Data/17082022_all2Params_t500")
LOC_MAG_DIR = os.path.join(BASE_DIR, "Data/savedLocalMagnitudes")

## set experiments ##
experiments = [
    {
        "active": True,  # just magnitudes - not normalised - 5 signatures
        "scale": 0.35,
        "num_signatures": 5,
        "norm": False,
        "magnitudes": True,
        "cellcounts": False,
        "distances": False,
        "order": [0, 3, 1, 5, 4],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # just magnitudes - not normalised - 6 signatures
        "scale": 0.35,
        "num_signatures": 6,
        "norm": False,
        "magnitudes": True,
        "cellcounts": False,
        "distances": False,
        "order": [0, 3, 1, 5, 2, 4, 6],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # just magnitudes - not normalised - 7 signatures
        "scale": 0.35,
        "num_signatures": 7,
        "norm": False,
        "magnitudes": True,
        "cellcounts": False,
        "distances": False,
        "order": [0, 3, 1, 5, 2, 4, 6],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # just magnitudes - not normalised - 8 signatures
        "scale": 0.35,
        "num_signatures": 8,
        "norm": False,
        "magnitudes": True,
        "cellcounts": False,
        "distances": False,
        "order": [0, 3, 1, 5, 6, 4, 2, 7],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # just magnitudes - not normalised - 9 signatures
        "scale": 0.35,
        "num_signatures": 9,
        "norm": False,
        "magnitudes": True,
        "cellcounts": False,
        "distances": False,
        "order": [0, 6, 5, 8, 4, 3, 2, 7, 1],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts - normalised - 5 signatures
        "num_signatures": 5,
        "norm": True,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "order": [0, 3, 4, 2, 1],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts - not normalised - 5 signatures
        "num_signatures": 5,
        "norm": False,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "order": [0, 4, 3, 1, 2],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts - normalised - 6 signatures
        "num_signatures": 6,
        "norm": True,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "order": [0, 2, 4, 5, 1, 3],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts - not normalised - 6 signatures
        "num_signatures": 6,
        "norm": False,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "order": [0, 4, 3, 5, 2, 1],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts - normalised - 7 signatures
        "num_signatures": 7,
        "norm": True,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "order": [0, 2, 4, 5, 1, 3, 6],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts - not normalised - 7 signatures
        "num_signatures": 7,
        "norm": False,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "order": [0, 4, 2, 6, 5, 1, 3],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts+distances - normalised - 6 signatures
        "num_signatures": 6,
        "norm": True,
        "magnitudes": False,
        "cellcounts": True,
        "distances": True,
        "order": [0, 1, 2, 4, 5, 3, 6],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts+distances - not normalised - 6 signatures
        "num_signatures": 6,
        "norm": False,
        "magnitudes": False,
        "cellcounts": True,
        "distances": True,
        "order": [0, 4, 3, 2, 1, 5, 6],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts+distances - normalised - 7 signatures
        "num_signatures": 7,
        "norm": True,
        "magnitudes": False,
        "cellcounts": True,
        "distances": True,
        "order": [0, 1, 2, 4, 5, 3, 6],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
    {
        "active": True,  # cellcounts+distances - not normalised - 7 signatures
        "num_signatures": 7,
        "norm": False,
        "magnitudes": False,
        "cellcounts": True,
        "distances": True,
        "order": [0, 4, 3, 6, 1, 5, 2],
        "plot_5x5": True,
        "plot_9x9": True,
        "plot_averages": True,
        "plot_scores": True,
    },
]
experiments_scores = [
    {
        "active": True,  # just magnitudes - not normalised
        "scale": 0.35,
        "numbers_signatures": list(range(2, 11)),
        "norm": False,
        "magnitudes": True,
        "cellcounts": False,
        "distances": False,
        "title": r"\textbf{Simple Magnitudes}",
    },
    {
        "active": True,  # cellcounts - not normalised
        "numbers_signatures": list(range(2, 11)),
        "norm": False,
        "magnitudes": False,
        "cellcounts": True,
        "distances": False,
        "title": r"\textbf{Simple Cell Counts}",
    },
    {
        "active": True,  # cellcounts + distances - normalised
        "numbers_signatures": list(range(2, 11)),
        "norm": True,
        "magnitudes": False,
        "cellcounts": True,
        "distances": True,
        "title": r"\textbf{Counts + Distances}",
    },
]

### set directory ###
for item in experiments + experiments_scores:
    a = "magnitudes_" if item["magnitudes"] else ""
    b = "cellcounts_" if item["cellcounts"] else ""
    c = "distances_" if item["distances"] else ""
    d = "normalised" if item["norm"] else "not-normalised"
    item["result_directory"] = os.path.join(RESULT_DIR, a + b + c, d)


### functions for experiments ###
def run_experiment(
    data_directory: str,
    local_magnitude_directory: str,
    N: int,
    num_signatures: int,
    norm: bool,
    magnitudes: bool,
    cellcounts: bool,
    distances: bool,
    colors: list,
    plot_5x5: bool,
    plot_9x9: bool,
    plot_averages: bool,
    result_directory: str,
    *,
    scale: float = None,
    seed: int = None,
) -> None:
    """Runs the experiment specified by inputs, i.e. it classifies NxN neighbourhoods in all simulation outcomes into a given number of local signatures according to certain features (magnitude, cellcounts, distances), and it plots the results in the desired ways (for 5x5 or 9x9 pseudo-randomly chosen simulation outcomes, or the average features per simulation outcome with typical representatives).

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        local_magnitude_directory (str): where precomputed local magnitudes / local cellcounts / distances are stored
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
        num_signatures (int): number of local signatures
        norm (bool): whether feature vectors are normalised prior to clustering
        magnitudes (bool): whether (simple) magnitudes should be used as features
        cellcounts (bool): whether cellcounts should be used as features
        distances (bool): whether average pairwise distances within each cell type should be used as features
        colors (list): list of colors (will be applied in that order to features)
        plot_5x5 (bool): whether results should be plotted across 5x5 parameter schemes (pseudo randomly chosen simulation instances)
        plot_9x9 (bool): whether results should be plotted across 9x9 parameter schemes (pseudo randomly chosen simulation instances)
        plot_averages (bool): whether the average features of each local signature together with a typical representative should be plotted
        result_directory (str): where the resulting plots should be saved to
        scale (float, optional): scaling factor for magnitude (if magnitude features are used). Defaults to None.
        seed (int, optional): for pseudo random selection of simulation outcomes to be displayed. Defaults to None.

    Raises:
        ValueError: input scale is needed if magnitudes is True, and input seed is needed if plot_5x5 or plot_9x9 are true
    """
    # check if all necessary input is given
    if (
        (plot_5x5 and seed is None)
        or (plot_9x9 and seed is None)
        or (magnitudes and scale is None)
    ):
        raise ValueError("not all necessary input provided")

    # check if result directory exists / create it
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    if os.path.exists(
        os.path.join(result_directory, f"all_info_{num_signatures}signatures")
    ) and os.path.exists(os.path.join(result_directory, "vectors")):
        picklefile = open(
            os.path.join(result_directory, f"all_info_{num_signatures}signatures"), "rb"
        )
        all_info = pickle.load(picklefile)
        picklefile.close()
        picklefile = open(os.path.join(result_directory, "vectors"), "rb")
        list_of_vectors = pickle.load(picklefile)
        picklefile.close()

    else:
        if os.path.exists(os.path.join(result_directory, "vectors")):
            picklefile = open(os.path.join(result_directory, "vectors"), "rb")
            list_of_vectors = pickle.load(picklefile)
            picklefile.close()
            all_info = prep_data(
                data_directory, local_magnitude_directory, N, compute_features=False
            )[0]

            # cluster using KMeans
            if norm:
                scaler = MinMaxScaler()
                scaler.fit(list_of_vectors)
                norm_list_of_vectors = scaler.transform(list_of_vectors)
                kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                    norm_list_of_vectors
                )
            else:
                kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                    list_of_vectors
                )

            labels = kmeans.labels_
            all_info["label"] = labels

            # store data
            file = open(
                os.path.join(result_directory, f"all_info_{num_signatures}signatures"),
                "wb",
            )
            pickle.dump(all_info, file)
            file.close()

        else:
            all_info, list_of_vectors = prep_data(
                data_directory,
                local_magnitude_directory,
                N,
                compute_features=True,
                magnitudes=magnitudes,
                cellcounts=cellcounts,
                distances=distances,
                scale=scale,
            )

            # cluster using KMeans
            if norm:
                scaler = MinMaxScaler()
                scaler.fit(list_of_vectors)
                norm_list_of_vectors = scaler.transform(list_of_vectors)
                kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                    norm_list_of_vectors
                )
            else:
                kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                    list_of_vectors
                )

            labels = kmeans.labels_
            all_info["label"] = labels

            # store data
            file = open(
                os.path.join(result_directory, f"all_info_{num_signatures}signatures"),
                "wb",
            )
            pickle.dump(all_info, file)
            file.close()
            file = open(os.path.join(result_directory, "vectors"), "wb")
            pickle.dump(list_of_vectors, file)
            file.close()

    if plot_5x5:
        plot_across_mxm_schemes(
            result_directory=result_directory,
            result_name=f"5x5-{num_signatures}signatures-s={scale}.pdf",
            all_info=all_info,
            colors=colors,
            N=N,
            seed=seed,
            m=5,
        )
    if plot_9x9:
        plot_across_mxm_schemes(
            result_directory=result_directory,
            result_name=f"9x9-{num_signatures}signatures-s={scale}.pdf",
            all_info=all_info,
            colors=colors,
            N=N,
            seed=seed,
            m=9,
        )
    if plot_averages:
        plot_averages_representatives(
            data_directory=data_directory,
            result_directory=result_directory,
            result_name=f"averages-{num_signatures}signatures-s={scale}.pdf",
            all_info=all_info,
            list_of_vectors=list_of_vectors,
            number_of_signatures=num_signatures,
            colors=colors,
            N=N,
            std=True,
            ylabel="Magnitude" if magnitudes else "Cellcount",
        )

    return


def run_cluster_quality_experiment(
    data_directory: str,
    local_magnitude_directory: str,
    N: int,
    numbers_signatures: list,
    norm: bool,
    magnitudes: bool,
    cellcounts: bool,
    distances: bool,
    result_directory: str,
    title: str,
    *,
    scale: float = None,
) -> None:
    """Generates plots (silhouette score, and inertia / WCSS for elbow method) to assess cluster quality depending on the number of signatures (i.e. k in k-means clustering) for the specified method (norm, magnitudes, cellcounts, distances and scale if applicable).

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        local_magnitude_directory (str): where precomputed local magnitudes / local cellcounts / distances are stored
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
        numbers_signatures (list): all values for the number of signatures for which the resulting silhouette score / WCSS should be plotted
        norm (bool): whether feature vectors are normalised prior to clustering
        magnitudes (bool): whether (simple) magnitudes should be used as features
        cellcounts (bool): whether cellcounts should be used as features
        distances (bool): whether average pairwise distances within each cell type should be used as features
        result_directory (str): where the resulting plots should be saved to
        title (str): plot title (e.g. the used method)
        scale (float, optional): scaling factor for magnitude (if magnitude features are used). Defaults to None.

    Raises:
        ValueError: input scale is needed if magnitudes is True
    """
    # check if all necessary input is given
    if magnitudes and scale is None:
        raise ValueError("not all necessary input provided")

    # check if result directory exists / create it
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # load precomputed scores
    if os.path.exists(os.path.join(result_directory, "silhouette")):
        picklefile = open(os.path.join(result_directory, "silhouette"), "rb")
        scores_s = pickle.load(picklefile)
        picklefile.close()
    else:
        scores_s = {}
    if os.path.exists(os.path.join(result_directory, "elbow_i")):
        picklefile = open(os.path.join(result_directory, "elbow_i"), "rb")
        inertias = pickle.load(picklefile)
        picklefile.close()
    else:
        inertias = {}

    # compute scores
    y_s = []
    y_e_i = []
    changes_s = False
    changes_e_i = False

    for num_signatures in numbers_signatures:
        if (
            num_signatures not in scores_s.keys()
            or num_signatures not in inertias.keys()
        ):
            if (
                os.path.exists(
                    os.path.join(
                        result_directory, f"all_info_{num_signatures}signatures"
                    )
                )
                and os.path.exists(os.path.join(result_directory, "vectors"))
                and num_signatures in inertias.keys()
            ):
                picklefile = open(
                    os.path.join(
                        result_directory, f"all_info_{num_signatures}signatures"
                    ),
                    "rb",
                )
                all_info = pickle.load(picklefile)
                picklefile.close()
                picklefile = open(os.path.join(result_directory, "vectors"), "rb")
                list_of_vectors = pickle.load(picklefile)
                picklefile.close()

            else:
                if os.path.exists(os.path.join(result_directory, "vectors")):
                    picklefile = open(os.path.join(result_directory, "vectors"), "rb")
                    list_of_vectors = pickle.load(picklefile)
                    picklefile.close()
                    all_info = prep_data(
                        data_directory,
                        local_magnitude_directory,
                        N,
                        compute_features=False,
                    )[0]

                    # cluster using KMeans
                    if norm:
                        scaler = MinMaxScaler()
                        scaler.fit(list_of_vectors)
                        norm_list_of_vectors = scaler.transform(list_of_vectors)
                        kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                            norm_list_of_vectors
                        )
                    else:
                        kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                            list_of_vectors
                        )

                    labels = kmeans.labels_
                    all_info["label"] = labels

                    # store data
                    file = open(
                        os.path.join(
                            result_directory, f"all_info_{num_signatures}signatures"
                        ),
                        "wb",
                    )
                    pickle.dump(all_info, file)
                    file.close()

                else:
                    all_info, list_of_vectors = prep_data(
                        data_directory,
                        local_magnitude_directory,
                        N,
                        compute_features=True,
                        magnitudes=magnitudes,
                        cellcounts=cellcounts,
                        distances=distances,
                        scale=scale,
                    )

                    # cluster using KMeans
                    if norm:
                        scaler = MinMaxScaler()
                        scaler.fit(list_of_vectors)
                        norm_list_of_vectors = scaler.transform(list_of_vectors)
                        kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                            norm_list_of_vectors
                        )
                    else:
                        kmeans = KMeans(n_clusters=num_signatures, random_state=0).fit(
                            list_of_vectors
                        )

                    labels = kmeans.labels_
                    all_info["label"] = labels

                    # store data
                    file = open(
                        os.path.join(
                            result_directory, f"all_info_{num_signatures}signatures"
                        ),
                        "wb",
                    )
                    pickle.dump(all_info, file)
                    file.close()
                    file = open(os.path.join(result_directory, "vectors"), "wb")
                    pickle.dump(list_of_vectors, file)
                    file.close()

            if norm:
                scaler = MinMaxScaler()
                scaler.fit(list_of_vectors)
                X = scaler.transform(list_of_vectors)
            else:
                X = list_of_vectors

            if num_signatures not in scores_s.keys():
                changes_s = True
                scores_s[num_signatures] = silhouette_score(X, list(all_info["label"]))
            if num_signatures not in inertias.keys():
                changes_e_i = True
                inertias[num_signatures] = kmeans.inertia_

        y_s.append(scores_s[num_signatures])
        y_e_i.append(inertias[num_signatures])

    # update the saved scores if needed:
    if changes_s:
        picklefile = open(os.path.join(result_directory, "silhouette"), "wb")
        pickle.dump(scores_s, picklefile)
        picklefile.close()
    if changes_e_i:
        picklefile = open(os.path.join(result_directory, "elbow_i"), "wb")
        pickle.dump(inertias, picklefile)
        picklefile.close()

    # plot all the scores
    fig, axes = plt.subplots(2, 1)
    fig.set_figwidth(0.33 * doc_textwidth)
    fig.set_figheight(0.33 * 2 * doc_textwidth)
    ax = axes[0]
    ax.plot(numbers_signatures, y_s, "-")
    ax.set_title(title, pad=8)
    if magnitudes:
        ax.set_ylabel("Silhouette Score")
    ax.set_xticks(numbers_signatures)
    if not distances:
        ax.set_yticks([])
    ax.set_ylim(0.58, 0.86)
    ax.yaxis.tick_right()
    ax = axes[1]
    ax.plot(numbers_signatures, y_e_i, "-")
    ax.set_xlabel(r"Number of Clusters $k$")
    if magnitudes:
        ax.set_ylabel("WCSS")
    ax.set_xticks(numbers_signatures)
    ax.set_yticks([])
    plt.savefig(
        os.path.join(
            result_directory,
            "compare_scores_"
            + str(numbers_signatures[0])
            + "-"
            + str(numbers_signatures[-1])
            + ".pdf",
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
    return


### run experiments ###
plot_across_schemes_display_sim_outcomes(
    data_directory=DATA_DIR,
    figure_directory=RESULT_DIR,
    file_name="plots-across-parameters.pdf",
    seed=seed,
)
for item in tqdm([exp for exp in experiments if exp["active"]]):
    # compute local magnitudes / cellcounts / distances if precomputed files do not exist
    if item["magnitudes"] and not os.path.exists(
        os.path.join(LOC_MAG_DIR, f"localMagnitudes_N={N}_s={item['scale']}")
    ):
        save_local_magnitudes(
            DATA_DIR,
            LOC_MAG_DIR,
            collect_files(DATA_DIR, range(1, 1621)),
            item["scale"],
            N,
        )
    if item["cellcounts"] and not os.path.exists(
        os.path.join(LOC_MAG_DIR, f"localCellcounts_N={N}")
    ):
        save_local_cellcounts(
            DATA_DIR, LOC_MAG_DIR, collect_files(DATA_DIR, range(1, 1621)), N
        )
    if item["distances"] and not os.path.exists(
        os.path.join(LOC_MAG_DIR, f"localAvrDistances_N={N}")
    ):
        save_local_average_pairwise_distances(
            DATA_DIR, LOC_MAG_DIR, LOC_MAG_DIR, collect_files(DATA_DIR, range(1, 1621))
        )

    # run experiment
    run_experiment(
        data_directory=DATA_DIR,
        local_magnitude_directory=LOC_MAG_DIR,
        N=N,
        seed=seed,
        scale=item["scale"] if "scale" in item else None,
        num_signatures=item["num_signatures"],
        norm=item["norm"],
        magnitudes=item["magnitudes"],
        cellcounts=item["cellcounts"],
        distances=item["distances"],
        colors=[cln[item["order"][i]] for i in range(item["num_signatures"])],
        plot_5x5=item["plot_5x5"],
        plot_9x9=item["plot_9x9"],
        plot_averages=item["plot_averages"],
        result_directory=item["result_directory"],
    )

for item in tqdm([exp for exp in experiments_scores if exp["active"]]):
    # compute local magnitudes / cellcounts / distances if precomputed files do not exist
    if item["magnitudes"] and not os.path.exists(
        os.path.join(LOC_MAG_DIR, f"localMagnitudes_N={N}_s={item['scale']}")
    ):
        save_local_magnitudes(
            DATA_DIR,
            LOC_MAG_DIR,
            collect_files(DATA_DIR, range(1, 1621)),
            item["scale"],
            N,
        )
    if item["cellcounts"] and not os.path.exists(
        os.path.join(LOC_MAG_DIR, f"localCellcounts_N={N}")
    ):
        save_local_cellcounts(
            DATA_DIR, LOC_MAG_DIR, collect_files(DATA_DIR, range(1, 1621)), N
        )
    if item["distances"] and not os.path.exists(
        os.path.join(LOC_MAG_DIR, f"localAvrDistances_N={N}")
    ):
        save_local_average_pairwise_distances(
            DATA_DIR, LOC_MAG_DIR, LOC_MAG_DIR, collect_files(DATA_DIR, range(1, 1621))
        )

    # plot for cluster assessment
    run_cluster_quality_experiment(
        data_directory=DATA_DIR,
        local_magnitude_directory=LOC_MAG_DIR,
        N=8,
        scale=item["scale"] if "scale" in item else None,
        numbers_signatures=item["numbers_signatures"],
        norm=item["norm"],
        magnitudes=item["magnitudes"],
        cellcounts=item["cellcounts"],
        distances=item["distances"],
        result_directory=item["result_directory"],
        title=item["title"],
    )
