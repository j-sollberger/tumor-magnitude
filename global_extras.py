from global_utils import *

# This additional code is for the sake of using a more elaborate ground truth to quantify the quality of results for different methods. A score is assigned depending on how well the resulting classification agrees with the subjective ground truth.

# more elaborate custom ground truth
gt = [[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],
      [0,0,-1],[0,0,-1],[0,0,-1],[0,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],
      [1,-1,-1],[0,-1,0],[0,0,0],[0,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],[-1,0,0],
      [1,-1,-1],[1,-1,-1],[0,-1,0],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],
      [1,-1,-1],[1,-1,-1],[0,-1,0],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],
      [1,-1,-1],[1,-1,-1],[0,-1,0],[0,-1,0],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],
      [1,-1,-1],[1,-1,-1],[0,-1,0],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],
      [1,-1,-1],[1,-1,-1],[0,-1,0],[0,-1,0],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],
      [1,-1,-1],[1,-1,-1],[0,-1,0],[0,-1,0],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1]]

# adapt ground truth with harshness value
harshness = 10.2
gth = [
    [gt[i][j] * (gt[i][j] != -1) + (-harshness) * (gt[i][j] == -1) for j in range(3)]
    for i in range(81)
]


def align_labels_refined(classification: list, gt: list) -> list:
    """assigns each cluster (0, 1 and 2) from classification to one of the 3 'E's (elimination: 0, equilibrium: 1, escape: 2) such that the classification matches the ground truth the closest.

    Args:
        classification (list): cluster (0, 1 or 2) for each parameter scheme in lexicographic order
        gt (list): contains ground truth. That is, for every parameter scheme in lexicographic order a list like [0,0,-1] judging the match of the parameter scheme to elimination, equilibrium, escape (in that order). 1 means: clear good match (e.g. this scheme is clearly elimination), -1 means: clear bad match (e.g. this scheme is definitely not escape), 0 means: neither obviously good, nor bad (e.g. a good part of the simulations in this scheme could be assigned to equilibrium, but certainly not all of them)

    Returns:
        list: assignment (0 = elimination, 1 = equilibrium, 2 = escape) for each parameter scheme in lexicographic order
    """
    # link classification to ground truth
    # possible matchings:
    dict1 = {0: 0, 1: 1, 2: 2}
    dict2 = {0: 0, 1: 2, 2: 1}
    dict3 = {0: 2, 1: 1, 2: 0}
    dict4 = {0: 1, 1: 0, 2: 2}
    dict5 = {0: 1, 1: 2, 2: 0}
    dict6 = {0: 2, 1: 0, 2: 1}
    options = [dict1, dict2, dict3, dict4, dict5, dict6]
    # find best one
    concordance = [
        sum(
            [
                max(gt[i])
                * (options[k][classification[i]] == np.argmax(np.array(gt[i])))
                for i in range(len(classification))
            ]
        )
        for k in range(len(options))
    ]
    link = options[np.argmax(np.array(concordance))]
    # modify classification accordingly
    classification_aligned = [
        link[classification[i]] for i in range(len(classification))
    ]
    return classification_aligned


def score(
    classification: list,
    aligned: bool,
    differently_assigned: list,
    numb_of_sim: list,
    gt: list,
) -> float:
    """Calculates the score measured by ground truth for a given classification of parameter schemes.

    Args:
        classification (list): cluster (0, 1 or 2) for each parameter scheme in lexicographic order
        aligned (bool): put True only if the clusters are already aligned with the 3 'E's (elimination: 0, equilibrium: 1, escape: 2). If False, align_labels function will be called.
        differently_assigned (list): contains for each parameter scheme in lexicographic order
        numb_of_sim (list): contains for each parameter scheme in lexicographic order the number of simulations used the scheme
        gt (list): contains ground truth. That is, for every parameter scheme in lexicographic order a list like [0,0,-1] judging the match of the parameter scheme to elimination, equilibrium, escape (in that order). 1 means: clear good match (e.g. this scheme is clearly elimination), -1 means: clear bad match (e.g. this scheme is definitely not escape), 0 means: neither obviously good, nor bad (e.g. a good part of the simulations in this scheme could be assigned to equilibrium, but certainly not all of them)

    Returns:
        float: assigned score <= 1.
    """
    if aligned == False:
        classification = align_labels_refined(classification, gt)

    purities = [
        (numb_of_sim[i] - differently_assigned[i]) / numb_of_sim[i]
        for i in range(len(numb_of_sim))
    ]

    score_unnormed = sum(
        [
            sum([(classification[i] == e) * gt[i][e] for e in [0, 1, 2]]) * purities[i]
            for i in range(len(gt))
        ]
    )

    maximum = sum([max(gt[i]) for i in range(len(gt))])

    return max(0, score_unnormed / maximum)
