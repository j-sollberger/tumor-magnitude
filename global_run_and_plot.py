import itertools
from pathlib import Path

from data_preparation_precomputation import collect_files
from global_utils import *

BASE_DIR = Path(__file__).resolve().parent


### manual settings ###

RESULT_DIR = os.path.join(BASE_DIR, "Results/global")
DATA_DIR = os.path.join(BASE_DIR, "Data/17082022_all2Params_t500")
MAG_DIR = os.path.join(BASE_DIR, "Data/savedMagnitudes")

# ground truth from Bull & Byrne: 0 for Equilibrium, 1 for Elimination, 2 for Escape
gt = [1,1,1,1,1,1,1,1,1,
      0,0,0,1,1,1,1,1,1,
      0,0,0,0,2,2,2,2,2,
      0,0,0,0,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2,
      0,0,0,2,2,2,2,2,2]

### list experiments ###

# get celltype combinations
cells = ["T", "M", "N", "V"]
comb = []
for r in range(1, len(cells) + 1):
    comb.extend(list(a) for a in itertools.combinations(cells, r))
combgeq2 = []
for r in range(2, len(cells) + 1):
    combgeq2.extend(list(a) for a in itertools.combinations(cells, r))

# experiments
experiments = (
    [
        {
            "active": True,  # all inclusion exclusion type differences (including just magnitudes |T|, ...)
            "scale": s,
            "combinations": [],
            "differences": comb,
            "plot_title": rf"\textbf{{Inclusion-Exclusion Differences at Scale {s}}}",
            "plot_title_reduced": rf"\textbf{{Inclusion-Exclusion Differences}}",
            "save_title": f"all-incl-excl-s={s}.pdf",
        }
        for s in [0.35]
    ]
    + [
        {
            "active": True,  # magnitudes of all possible combinations
            "scale": s,
            "combinations": comb,
            "differences": [],
            "plot_title": rf"\textbf{{All Combinations at Scale {s}}}",
            "plot_title_reduced": rf"\textbf{{All Combinations}}",
            "save_title": f"all-comb-s={s}.pdf",
        }
        for s in [0.35]
    ]
    + [
        {
            "active": True,  # all inclusion exclusion type differences and magnitudes of all possible combinations
            "scale": s,
            "combinations": comb,
            "differences": combgeq2,
            "plot_title": rf"\textbf{{All Combinations and Differences at Scale {s}}}",
            "plot_title_reduced": rf"\textbf{{All Combinations and Differences}}",
            "save_title": f"all-incl-excl-and-comb-s={s}.pdf",
        }
        for s in [0.35]
    ]
    + [
        {
            "active": True,  # just magnitudes |T|, |M|, |N|, |V|
            "scale": s,
            "combinations": [[cell] for cell in cells],
            "differences": [],
            "plot_title": rf"\textbf{{Just Magnitudes at Scale {s}}}",
            "plot_title_reduced": rf"\textbf{{Just Magnitudes}}",
            "save_title": f"just-magnitudes-s={s}.pdf",
        }
        for s in [0.35]
    ]
)

### function to run experiment ###


def run_experiment(
    magnitude_directory: str,
    result_directory: str,
    paramdf_reduced: pd.DataFrame,
    gt: list,
    scale: float,
    combinations: list,
    differences: list,
    plot_title: str,
    plot_title_reduced: str,
    save_title: str,
) -> None:
    """Runs the experiment specified by inputs, i.e. it classifies the parameter schemes into 3 long-term outcomes according to certain magnitude features (combinations and/or inclusion-exclusion type differences), and it plots the results in the desired way (colors aligned with a certain ground truth).

    Args:
        magnitude_directory (str): where the precomputed magnitudes are stored
        result_directory (str): where the resulting plots should be saved to
        paramdf_reduced (pd.DataFrame): dataframe with columns 'ID', 'chi_macrophageToCSF', 'halfMaximalExtravasationCsf1Conc', and reduced to only those rows whose ID exists in the data.
        gt (list): ground truth for alignment of labels / colors
        scale (float): scaling factor for magnitude
        combinations (list): celltype combinations whose magnitudes should be included in the feature vectors
        differences (list): celltype combinations for which the corresponding inclusion-exclusion type differences of magnitudes should be included in the feature vectors
        plot_title (str): title of the resulting plot
        plot_title_reduced (str): title of the resulting reduced plot
        save_title (str): under which name the plot should be saved
    """

    vectors = [
        a + b
        for a, b in zip(
            give_vector_all_comb_magnitudes(magnitude_directory, [scale], combinations),
            give_vector_magnitude_differences(
                magnitude_directory, [scale], differences
            ),
        )
    ]
    classification, _, diff_assigned, numb_of_sim = classify_schemes(
        vectors, paramdf_reduced, gt
    )
    purities = [
        (numb_of_sim[i] - diff_assigned[i]) / numb_of_sim[i]
        for i in range(len(numb_of_sim))
    ]

    # check if result directory exists / create it
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # visualize results
    plot_classification_with_purities(
        result_directory, classification, purities, plot_title, save_title
    )
    reduced_plot_classification_with_purities(
        result_directory, classification, purities, plot_title_reduced, save_title
    )

    return


### run experiments ###

filenrs = collect_files(DATA_DIR, range(1, 1621))
paramdf = pd.read_csv(os.path.join(os.path.dirname(DATA_DIR), "params_2ParamSweep.csv"))
paramdf_reduced = paramdf.loc[paramdf["ID"].isin(filenrs)]

for experiment in experiments:
    if experiment["active"]:
        # compute magnitudes if precomputed files do not exist
        if not os.path.exists(
            os.path.join(MAG_DIR, f"magnitudesScale{experiment['scale']}.csv")
        ):
            save_magnitudes_to_csv(DATA_DIR, MAG_DIR, filenrs, experiment["scale"])
        # run experiment
        run_experiment(
            magnitude_directory=MAG_DIR,
            result_directory=RESULT_DIR,
            paramdf_reduced=paramdf_reduced,
            gt=gt,
            scale=experiment["scale"],
            combinations=experiment["combinations"],
            differences=experiment["differences"],
            plot_title=experiment["plot_title"],
            plot_title_reduced=experiment["plot_title_reduced"],
            save_title=experiment["save_title"],
        )
