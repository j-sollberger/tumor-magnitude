import csv
import itertools
import math
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def magnitude(x: list, y: list, s: float) -> float:
    """Computes the magnitude of a point set in the Euclidean plane (scaled by s), where the points have coordinates x and y.

    Args:
        x (list): x-coordinates
        y (list): y-coordinates
        s (float): scaling of the metric space

    Returns:
        float: Magnitude if it exists; 0 if the point set is empty; -1 if the similarity matrix is not invertible
    """
    n = len(x)
    if n == 0:
        return 0  # no cells of this type
    sim_mat = [
        [math.e ** (-s * math.hypot(x[i] - x[j], y[i] - y[j])) for j in range(n)]
        for i in range(n)
    ]
    try:
        inv = np.linalg.inv(sim_mat)
        return np.add.reduce([np.add.reduce(inv[i, :]) for i in range(n)])
    except:
        return -1.0  # inversion failed


def magnitude_celltypes(data: pd.DataFrame, celltypes: list, s: float) -> float:
    """Calculates the magnitude of the point set consisting of exactly the species listed in the list celltypes.

    Args:
        data (pd.DataFrame): dataframe with the first 3 columns 'points_x', 'points_y', 'celltypes'
        celltypes (list): list of strings out of 'T', 'M', 'N', 'V'
        s (float): scaling factor for magnitude

    Returns:
        float: Magnitude
    """
    conversion = {"T": "Tumour", "M": "Macrophage", "N": "Necrotic", "V": "Vessel"}
    celltypes = [conversion[A] for A in celltypes]

    x = list((data.loc[data["celltypes"].isin(celltypes), "points_x"]).values)
    y = list((data.loc[data["celltypes"].isin(celltypes), "points_y"]).values)
    return magnitude(x, y, s)


def collect_files(data_directory: str, potential_filenrs: list) -> list:
    """Looks for all "good" files in the given directory and filenr in potential_filenrs outputs a list of the filenames.

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        potential_filenrs (list): will check for files with names f'ID-{filenr}_time-500_From2ParamSweep_Data.csv' for filenr in potential_filenrs

    Returns:
        list: file numbers
    """
    filenrs = []
    # go through each data file
    for filenr in potential_filenrs:
        # check if corresponding file exists
        if os.path.exists(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            )
        ):
            filenrs.append(filenr)

    return filenrs


### GLOBAL ###
def save_magnitudes_to_csv(
    data_directory: str, saved_magnitudes_directory: str, filenrs: list, s: float
) -> None:
    """Calculates the magnitudes for all possible combinations of celltypes out of 'Tumour', 'Macrophage', 'Necrotic', 'Vessel', and saves them to a csv file with rows corresponding to the different data files and columns being labeled as 'T', 'M', ..., 'T,M', 'T,N', ... , 'T,M,N,V'.

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        saved_magnitudes_directory (str): where the magnitudes should be saved to
        filenrs (list): list of file numbers
        s (float): scaling factor for magnitude
    """
    # check if saved magnitudes directory exists / create it
    if not os.path.exists(saved_magnitudes_directory):
        os.makedirs(saved_magnitudes_directory)

    cells = ["Tumour", "Macrophage", "Necrotic", "Vessel"]
    power_set = []
    for r in range(1, len(cells) + 1):
        power_set.extend(itertools.combinations(cells, r))

    combinations = []
    for filenr in tqdm(filenrs, leave=True, position=0):
        data = pd.read_csv(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            ),
            index_col=0,
        )
        combinations.append(
            {
                ",".join([item[0] for item in celltypes]): magnitude_celltypes(
                    data, celltypes, s
                )
                for celltypes in power_set
            }
        )

    with open(
        os.path.join(saved_magnitudes_directory, f"magnitudesScale{s}.csv"),
        "w",
        newline="",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(combinations[0].keys()))
        writer.writeheader()
        writer.writerows(combinations)
    return


### LOCAL ###
def give_ball(data: pd.DataFrame, x0: float, y0: float, r: float) -> pd.DataFrame:
    """Given data points, returns all points within a closed ball with given center and radius.

    Args:
        data (pd.DataFrame): data frame containing given point cloud, with columns 'points_x' and 'points_y'
        x0 (float): x coordinate of center of the ball
        y0 (float): y coordinate of center of the ball
        r (float): radius of the ball

    Returns:
        pd.DataFrame: sub- data frame of data with only points within the closed ball.
    """
    return data.loc[
        [
            k
            for k in range(data.shape[0])
            if (x0 - data.loc[k, "points_x"]) ** 2 + (y0 - data.loc[k, "points_y"]) ** 2
            <= r**2
        ]
    ]


def generate_checkpoints(N: int) -> list:
    """Given a 50x50 square domain, generates an evenly spaced grid of NxN check-points such that the distance between two neighbouring grid points is double the distance between a boundary grid point and the boundary of the domain.

    Args:
        N (int): simulation domain will be covered with NxN local neighbourhoods (disks), whose centers are computed here

    Returns:
        list: x (=y) coordinates of the checkpoints
    """
    d = 25 / float(N)
    return [d + 2 * k * d for k in range(N)]


def local_ball_magnitudes(
    data: pd.DataFrame, x0: float, y0: float, r: float, s: float
) -> list:
    """Calculates all combinations magnitude of a given point cloud restricted to a closed ball with given center and radius.

    Args:
        data (pd.DataFrame): data frame that contains entire point cloud, with columns 'points_x' and 'points_y'
        x0 (float): x coordinate of center
        y0 (float): y coordinate of center
        r (float): radius of the ball
        s (float): scaling factor for magnitude

    Returns:
        list: magnitudes of all combinations of tumor cells, macrophages, and necrotic cells (order: |T|, |M|, |N|, |TM|, |TN|, |MN|, |TMN|)
    """
    # get ball
    ball = give_ball(data, x0, y0, r)
    # cell type combinations
    cells = ["T", "M", "N"]
    combtypes = []
    for r in range(1, len(cells) + 1):
        combtypes.extend(list(comb) for comb in itertools.combinations(cells, r))

    return [magnitude_celltypes(ball, t, s) for t in combtypes]


def save_local_magnitudes(
    data_directory: str,
    saved_magnitudes_directory: str,
    filenrs: list,
    s: float,
    N: int,
) -> None:
    """Saves local magnitudes across all simulation outcomes to a pickle file. Here for a given number of NxN checkpoints (and minimal closed disks that cover the entire domain) and a given scaling factor s. The local magnitudes are saved to a pickle file 'localMagnitudes_N={N}_s={s}' in the format of a pandas dataframe, with rows corresponding to simulation outcomes, and columns labeled in the format of '1,2,M,N' – one for each celltype combination for each local disk.

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        saved_magnitudes_directory (str): where local magnitudes should be saved to
        filenrs (list): list of filenrs for whose corresponding simulation outcomes the local magnitudes should be saved
        s (float): scaling factor for magnitude
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
    """
    # check if saved magnitudes directory exists / create it
    if not os.path.exists(saved_magnitudes_directory):
        os.makedirs(saved_magnitudes_directory)

    cells = ["Tumour", "Macrophage", "Necrotic"]
    power_set = []
    for r in range(1, len(cells) + 1):
        power_set.extend(itertools.combinations(cells, r))

    xy = generate_checkpoints(N)
    radius = math.sqrt(2) * 25 / float(N)

    frame = pd.DataFrame(
        columns=[
            ",".join([str(i), str(j)] + [item[0] for item in celltypes])
            for i in range(len(xy))
            for j in range(len(xy))
            for celltypes in power_set
        ]
    )

    for filenr in tqdm(filenrs):
        data = pd.read_csv(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            ),
            index_col=0,
        )
        frame.loc[frame.shape[0]] = np.array(
            [local_ball_magnitudes(data, x, y, radius, s) for x in xy for y in xy]
        ).flatten()

    # dump into pickle file
    file = open(
        os.path.join(saved_magnitudes_directory, f"localMagnitudes_N={N}_s={s}"), "wb"
    )
    pickle.dump(frame, file)
    file.close()

    return


def read_local_magnitudes(
    saved_magnitudes_directory: str, s: float, N: int
) -> pd.DataFrame:
    """Reads the saved local magnitudes from the corresponding pickle file with name 'localMagnitudes_N={N}_s={s}'.

    Args:
        saved_magnitudes_directory (str): where precomputed local magnitudes are saved
        s (float): scaling factor
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)

    Returns:
        pd.DataFrame: data frame with rows corresponding to simulation outcomes, and columns labeled in the format of '1,2,M,N' – one for each celltype combination for each local disk.
    """
    file = open(
        os.path.join(saved_magnitudes_directory, f"localMagnitudes_N={N}_s={s}"), "rb"
    )
    newframe = pickle.load(file)
    file.close()

    return newframe


def save_local_cellcounts(
    data_directory: str, saved_cellcounts_directory: str, filenrs: list, N: int
) -> None:
    """Saves local cellcounts across all simulation outcomes to a pickle file. Here for a given number of NxN checkpoints (and minimal closed disks that cover the entire domain). The local cellcounts are saved to a pickle file 'localCellcounts={N}' in the format of a pandas dataframe, with rows corresponding to simulation outcomes, and columns labeled in the format of '1,2,M,N' – one for each celltype combination for each local disk. This is the exactly the same format as for local magnitudes.

    Args:
        data_directory (str): directory to data files
        saved_magnitudes_directory (str): where local cellcounts should be saved to
        files (list): list of file names of which the local cellcounts should be saved
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
    """
    # check if saved cellcounts directory exists / create it
    if not os.path.exists(saved_cellcounts_directory):
        os.makedirs(saved_cellcounts_directory)

    cells = ["Tumour", "Macrophage", "Necrotic"]
    power_set = []
    for r in range(1, len(cells) + 1):
        power_set.extend(itertools.combinations(cells, r))

    xy = generate_checkpoints(N)
    radius = math.sqrt(2) * 25 / float(N)

    frame = pd.DataFrame(
        columns=[
            ",".join([str(i), str(j)] + [item[0] for item in celltypes])
            for i in range(len(xy))
            for j in range(len(xy))
            for celltypes in power_set
        ]
    )

    for filenr in tqdm(filenrs):
        data = pd.read_csv(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            ),
            index_col=0,
        )
        row = []
        for x in xy:
            for y in xy:
                ball = give_ball(data, x, y, radius)
                for celltype_combination in power_set:
                    row.append(
                        ball.loc[ball["celltypes"].isin(celltype_combination)].shape[0]
                    )
        frame.loc[frame.shape[0]] = row

    # dump into pickle file
    file = open(
        os.path.join(saved_cellcounts_directory, f"localCellcounts_N={N}"), "wb"
    )
    pickle.dump(frame, file)
    file.close()

    return


def read_local_cellcounts(saved_cellcounts_directory: str, N: int) -> pd.DataFrame:
    """Reads the saved local cellcounts from the corresponding pickle file with name 'localCellcounts_N={N}'.

    Args:
        saved_magnitudes_directory (str): where precomputed local cellcounts are saved
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)

    Returns:
        pd.DataFrame: data frame with rows corresponding to simulation outcomes, and columns labeled in the format of '1,2,M,N' – one for each celltype combination for each local disk.
    """

    file = open(
        os.path.join(saved_cellcounts_directory, f"localCellcounts_N={N}"), "rb"
    )
    newframe = pickle.load(file)
    file.close()

    return newframe


def save_distances(
    data_directory: str, saved_distances_directory: str, filenrs: list
) -> None:
    """For each filenr in a given list, this function computes all pairwise distances corresponding to the simulation outcome with the given filenr. Results are saved in the format of a matrix (nested list) to a pickle file named distances_filenr{filenr}.

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        saved_distances_directory (str): where pairwise distances should be saved to
        filenrs (list): list of all filenrs for whose corresponding simulation outcomes the pairwise distances should be computed and stored
    """
    # check if saved distances directory exists / create it
    if not os.path.exists(saved_distances_directory):
        os.makedirs(saved_distances_directory)

    for filenr in tqdm(filenrs):
        data = pd.read_csv(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            ),
            index_col=0,
        )
        data = data.loc[data["celltypes"].isin(["Tumour", "Macrophage", "Necrotic"])]

        n = data.shape[0]
        A = [[0 for _ in range(n)] for _ in range(n)]

        array = data.to_numpy()
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = np.linalg.norm(array.iloc[i, :2] - array.iloc[j, :2])
                A[i][j] = d
                A[j][i] = d

        # dump into pickle file
        file = open(
            os.path.join(saved_distances_directory, f"distances_filenr{filenr}"), "wb"
        )
        pickle.dump(A, file)
        file.close()

    return


def read_distances(saved_distances_directory: str, filenr: int) -> list:
    """Reads the saved pairwise distances for a given filenr from the corresponding pickle file with name 'distances_filenr{filenr}'.

    Args:
        saved_distances_directory (str): where precomputed pairwise distances are saved
        filenr (int): determines the simulation outcome for which the distances are read

    Returns:
        list: nxn matrix A, where n is the total number of tumour cells / macrophages / necrotic cells in the given simulation outcome. Entry A[i][j] is the distance between cell i and cell j, with the order of cells given by the original data file.
    """

    picklefile = open(
        os.path.join(saved_distances_directory, f"distances_filenr{filenr}"), "rb"
    )
    A = pickle.load(picklefile)
    picklefile.close()

    return A


def save_local_average_pairwise_distances(
    data_directory: str,
    saved_distances_directory: str,
    saved_average_distances_directory: str,
    N: int,
    filenrs: list,
) -> None:
    """Saves local average pairwise distances across all simulation outcomes to a pickle file. Here for a given number of NxN checkpoints (and minimal closed disks that cover the entire domain). The local average distances are saved to a pickle file 'localAvrDistances_N={N}' in the format of a pandas dataframe, with rows corresponding to simulation outcomes, and columns labeled in the format of '1,2,M,N' – with a single listed cell type corresponding to pairwise distances within this cell type, and a pair of two cell types corresponding to pairwise distances between these two cell types.

    Args:
        data_directory (str): where the data (simulation outcomes) is stored
        saved_distances_directory (str): where all pairwise distances are stored
        saved_average_distances_directory (str): where local average distances should be saved to
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
        filenrs (list): list of filenrs for whose corresponding local average distances should be saved
    """

    if not os.path.exists(
        os.path.join(saved_distances_directory, f"distances_filenr{filenr}")
    ):
        save_distances(data_directory, saved_distances_directory, filenrs)

    cells = ["Tumour", "Macrophage", "Necrotic"]
    power_set = []
    for r in range(1, len(cells)):
        power_set.extend(itertools.combinations(cells, r))

    xy = generate_checkpoints(N)
    radius = math.sqrt(2) * 25 / float(N)

    frame = pd.DataFrame(
        columns=[
            ",".join([str(i), str(j)] + [item[0] for item in celltypes])
            for i in range(len(xy))
            for j in range(len(xy))
            for celltypes in power_set
        ]
    )

    for filenr in tqdm(filenrs):
        A = read_distances(saved_distances_directory, filenr)
        data = pd.read_csv(
            os.path.join(
                data_directory, f"ID-{filenr}_time-500_From2ParamSweep_Data.csv"
            ),
            index_col=0,
        )
        data = data.loc[data["celltypes"].isin(["Tumour", "Macrophage", "Necrotic"])]
        data = data.reset_index(drop=True)
        row = []
        for x in xy:
            for y in xy:
                ball = give_ball(data, x, y, radius)
                tcells = list(ball.loc[ball["celltypes"] == "Tumour"].index)
                mcells = list(ball.loc[ball["celltypes"] == "Macrophage"].index)
                ncells = list(ball.loc[ball["celltypes"] == "Necrotic"].index)
                # tumour average
                row.append(
                    np.average(
                        [
                            A[tcells[a]][tcells[b]]
                            for a in range(len(tcells) - 1)
                            for b in range(a + 1, len(tcells))
                        ]
                    )
                    if len(tcells) > 1
                    else 0
                )
                # macrophage average
                row.append(
                    np.average(
                        [
                            A[mcells[a]][mcells[b]]
                            for a in range(len(mcells) - 1)
                            for b in range(a + 1, len(mcells))
                        ]
                    )
                    if len(mcells) > 1
                    else 0
                )
                # necrotic average
                row.append(
                    np.average(
                        [
                            A[ncells[a]][ncells[b]]
                            for a in range(len(ncells) - 1)
                            for b in range(a + 1, len(ncells))
                        ]
                    )
                    if len(ncells) > 1
                    else 0
                )
                # tumour-macrophage average
                row.append(
                    np.average(
                        [
                            A[tcells[a]][mcells[b]]
                            for a in range(len(tcells))
                            for b in range(len(mcells))
                        ]
                    )
                    if len(tcells) > 0 and len(mcells) > 0
                    else 0
                )
                # tumour-necrotic average
                row.append(
                    np.average(
                        [
                            A[tcells[a]][ncells[b]]
                            for a in range(len(tcells))
                            for b in range(len(ncells))
                        ]
                    )
                    if len(tcells) > 0 and len(ncells) > 0
                    else 0
                )
                # macrophage-necrotic average
                row.append(
                    np.average(
                        [
                            A[mcells[a]][ncells[b]]
                            for a in range(len(mcells))
                            for b in range(len(ncells))
                        ]
                    )
                    if len(mcells) > 0 and len(ncells) > 0
                    else 0
                )
        frame.loc[frame.shape[0]] = row

    # dump into pickle file
    file = open(
        os.path.join(saved_average_distances_directory, f"localAvrDistances_N={N}"),
        "wb",
    )
    pickle.dump(frame, file)
    file.close()

    return


def read_local_average_pairwise_distances(
    saved_average_distances_directory: str, N: int
) -> pd.DataFrame:
    """Reads the saved local average pairwise distances per cell type from the corresponding pickle file with name 'localAvrDistances_N={N}'.

    Args:
        saved_average_distances_directory (str): where the pd dataframe with the local average pairwise distances is stored
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)

    Returns:
        pd.DataFrame: data frame with rows corresponding to simulation outcomes, and columns labeled in the format of '1,2,M,N' – with a single listed cell type corresponding to pairwise distances within this cell type, and a pair of two cell types corresponding to pairwise distances between these two cell types.
    """

    file = open(
        os.path.join(saved_average_distances_directory, f"localAvrDistances_N={N}"),
        "rb",
    )
    newframe = pickle.load(file)
    file.close()

    return newframe
