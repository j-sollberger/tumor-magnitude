import numpy as np
import libpysal
from esda.moran import Moran
import pandas as pd
import os
import math
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from data_preparation_precomputation import collect_files, read_distances
from pathlib import Path
from tqdm import tqdm
import pickle
from global_utils import (
    classify_schemes,
    plot_classification_with_purities,
    give_vector_magnitude_differences,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "Data/17082022_all2Params_t500")
RESULT_DIR = os.path.join(BASE_DIR, "Results-morans-i")
LOC_MAG_DIR = os.path.join(BASE_DIR, "Data/savedLocalMagnitudes")
MAG_DIR = os.path.join(BASE_DIR, "Data/savedMagnitudes")


def build_knn_weights(A: list, k: int = 5) -> np.ndarray:
    """Build binary kNN matrix.

    Args:
        A(list): Pairwise distances.
        k (int, optional): Number of nearest neighbours considered (if A is large enough). Defaults to 5.

    Returns:
        np.ndarray: Binary k-nearest-neighbour matrix.
    """
    m = len(A)
    k = min(m - 1, k)
    nbrs = NearestNeighbors(metric="precomputed", n_neighbors=k + 1).fit(A)
    _, ind = nbrs.kneighbors(A)

    W = np.zeros((m, m), dtype=int)
    for i, neighbors in enumerate(ind):
        for j in neighbors[1:]:
            W[i, j] = 1
            W[j, i] = 1
    return W


def build_similarity_matrix(id: int) -> np.ndarray:
    """Build similarity matrix.

    Args:
        id (int): Id of simulation output.

    Returns:
        np.ndarray: Similarity matrix.
    """
    A = read_distances(LOC_MAG_DIR, id)
    n = len(A)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = math.e ** (-A[i][j])
            W[j, i] = W[i, j]
    return W


def calculate_morans(
    weight_type: str, labels: list, A: np.ndarray | list, pairs: list, k: int = 5
) -> dict:
    """Calculates the Moran's I for all pairs of cell types. Defaults to 0 if not both cell types are present.

    Args:
        weight_type (str): whether k-nearest-neighbour or similarity matrix weights should be used
        labels (list): contains for each cell the cell type 'T', 'M', 'N' or 'V'
        A (np.ndarray | list): contains either pairwise distances between cells (knn case) or is the precomputed similarity matrix itself (similarity case)
        pairs (list): contains all possible pairs of cell types
        k (int, optional): k for knn. Defaults to 5.

    Raises:
        ValueError: If weight_type is neither 'knn' nor 'similarity'.

    Returns:
        dict: Has keys (a,b) for each pair a,b of celltypes and values the corresponding Moran's I.
    """
    if weight_type not in ["knn", "similarity"]:
        raise ValueError("Argument weight_type has to be 'knn' or 'similarity'.")

    morans = {}
    uniques = np.unique(labels)
    for a, b in pairs:
        if (a not in uniques) or (b not in uniques):
            morans[(a, b)] = 0
        else:
            indices = [i for i, label in enumerate(labels) if label in (a, b)]
            filtered_labels = [labels[i] for i in indices]
            mapping = {a: 0, b: 1}
            binary_labels = [mapping[label] for label in filtered_labels]
            A_sub = A[np.ix_(indices, indices)]

            if weight_type == "knn":
                W = build_knn_weights(A_sub, k)
            else:
                W = A_sub

            mi = Moran(y=binary_labels, w=libpysal.weights.full2W(W), permutations=0)
            morans[(a, b)] = mi.I
    return morans


# check if result directory exists / create it
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ground truth used to align colors of classification
gt = [1,1,1,1,1,1,1,1,1,
      0,0,0,1,1,1,1,1,1,
      0,0,0,0,2,2,2,2,2,
      0,0,0,0,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2]

# prep
filenrs = collect_files(DATA_DIR, range(1, 1621))
paramdf = pd.read_csv(os.path.join(os.path.dirname(DATA_DIR), "params_2ParamSweep.csv"))
paramdf_reduced = paramdf.loc[paramdf["ID"].isin(filenrs)]
pairs = [(str(k), (l)) for k, l in combinations(["T", "M", "N", "V"], 2)]


### Retreive or compute features ###
features_similarity = []
features_knn = []

knn_exist = os.path.exists(os.path.join(RESULT_DIR, "simple-knn-features"))
similarity_exist = os.path.exists(
    os.path.join(RESULT_DIR, "simple-similarity-features")
)

if knn_exist:
    picklefile = open(os.path.join(RESULT_DIR, "simple-knn-features"), "rb")
    features_knn = pickle.load(picklefile)
    picklefile.close()

if similarity_exist:
    picklefile = open(os.path.join(RESULT_DIR, "simple-similarity-features"), "rb")
    features_similarity = pickle.load(picklefile)
    picklefile.close()

if not knn_exist or not similarity_exist:
    for id in tqdm(filenrs):
        data = pd.read_csv(
            os.path.join(
                "Data/17082022_all2Params_t500",
                f"ID-{id}_time-500_From2ParamSweep_Data.csv",
            ),
            index_col=0,
        )

        data = data.loc[data["celltypes"] != "Stroma"]
        data = data.loc[data["celltypes"] != "Vessel"]
        labels_long = data["celltypes"]
        labels = np.array([label[0] for label in labels_long])

        if not similarity_exist:
            W = build_similarity_matrix(id)

            morans = calculate_morans("similarity", labels, W, pairs)

            features_similarity.append(list(morans.values()))

        if not knn_exist:
            A = np.array(read_distances(LOC_MAG_DIR, id))

            morans = calculate_morans("knn", labels, A, pairs, k=5)

            features_knn.append(list(morans.values()))

    picklefile = open(os.path.join(RESULT_DIR, "simple-knn-features"), "wb")
    pickle.dump(features_knn, picklefile)
    picklefile.close()

    picklefile = open(os.path.join(RESULT_DIR, "simple-similarity-features"), "wb")
    pickle.dump(features_similarity, picklefile)
    picklefile.close()

### Classify and plot ###

classification, _, diff_assigned, numb_of_sim = classify_schemes(
    features_knn, paramdf_reduced, gt
)
purities = [
    (numb_of_sim[i] - diff_assigned[i]) / numb_of_sim[i]
    for i in range(len(numb_of_sim))
]
plot_classification_with_purities(
    RESULT_DIR,
    classification,
    purities,
    rf"\textbf{{Using Simple Moran's I Features with 5 Nearest Neighbour Weights}}",
    "simple-morans-i-knn.pdf",
)

classification, _, diff_assigned, numb_of_sim = classify_schemes(
    features_similarity, paramdf_reduced, gt
)
purities = [
    (numb_of_sim[i] - diff_assigned[i]) / numb_of_sim[i]
    for i in range(len(numb_of_sim))
]
plot_classification_with_purities(
    RESULT_DIR,
    classification,
    purities,
    rf"\textbf{{Using Simple Moran's I Features with Similarity Matrix}}",
    "simple-morans-i-similarity.pdf",
)


### Compute and print Pearson correlation with pairwise-only magnitude features ###

features_mag = give_vector_magnitude_differences(
    MAG_DIR, [0.35], [[a, b] for a, b in pairs]
)
print(
    "Pearson correlation between Moran's I with 5-nearest-neighbour weights and magnitude = "
    + str(np.corrcoef(features_mag, features_knn)[0, 1])
)
print(
    "Pearson correlation between Moran's I with similarity matrix weights and magnitude = "
    + str(np.corrcoef(features_mag, features_similarity)[0, 1])
)
