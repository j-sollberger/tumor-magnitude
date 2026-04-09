from matplotlib.backends.backend_pdf import PdfPages

from local_utils import *


def plot_all(
    result_directory: str,
    result_name: str,
    all_info: pd.DataFrame,
    colors: list,
    N: int,
) -> None:
    """Plot the (pre-computed) local magnitude signature assignments for all simulation outcomes.

    Args:
        result_directory (str): where the plots should be saved to
        result_name (str): under what name the plots should be saved
        all_info (pd.DataFrame): One row per simulation outcome. Has columns 'filenr', 'index1', 'index2', 'chi', 'c12', 'label'
        colors (list): list of colors that correspond to local signatures in that order
        N (int): simulation domain is covered with NxN local neighbourhoods (disks)
    """
    indices = [
        [all_info.loc[k, "index1"], all_info.loc[k, "index2"]] for k in range(N * N)
    ]
    with PdfPages(os.path.join(result_directory, result_name)) as pdf:
        for i in tqdm(range(all_info.shape[0] // (N * N))):
            plt.figure(layout="constrained", figsize=(3.25, 3.5))
            plot_local_signatures(
                indices,
                list(all_info.loc[i * N * N : (i + 1) * N * N, "label"]),
                f"filenr {all_info.loc[i*N*N,'filenr']}",
                colors,
            )
            pdf.savefig()
            plt.close()
    return


def plot_all_features(
    result_directory: str,
    result_name: str,
    all_info: pd.DataFrame,
    list_of_vectors: list,
    colors: list,
) -> None:
    """Creates a 3D plot of the 3 just-magnitude features of all local neighbourhoods, colored by their assigned local type.

    Args:
        result_directory (str): where the plot should be saved to
        result_name (str): under what name the plot should be saved to
        all_info (pd.DataFrame): Contains the precomputed labels (signature assignments). Has columns 'filenr', 'index1', 'index2', 'chi', 'c12', 'label'
        list_of_vectors (list): features corresponding to the local neighbourhoods in the same order as in all_info
        colors (list): list of colors corresponding to the local signatures
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    a = np.array(list_of_vectors).T
    ax.scatter(a[0], a[1], a[2], c=[colors[i] for i in list(all_info["label"])], s=0.1)
    ax.set_title("Local features with neighbourhood type")
    plt.savefig(
        os.path.join(result_directory, result_name), format="pdf", bbox_inches="tight"
    )
    plt.close()
    return
