# Magnitude-based features for multispecies data
This repository contains code used in the preprint https://arxiv.org/abs/2606.11775 (only the code used on the synthetic data).


### Description
We use magnitude based features to analyze synthetic data of the tumour microenvironment in two different manners: (1) We apply magnitude __locally__ to create distinct local signatures that capture spatial heterogeneity, and (2) we apply magnitude __globally__ to classify underlying parameter schemes into three long-term tumour outcomes: elimination, equilibrium and escape. In the local application (1) we use simple magnitude feature vectors and further compare our method to simply summary statistics such as cell counts and average pairwise distances per cell type. In the global application (2) we use more intricate magnitude features and compare results with those from simpler magnitude features.


### Data
We use outputs from an agent based model that models the tumor microenvironment by Joshua A. Bull and Helen M. Byrne [[1]](#1). Specifically, we work with simulation outputs at a single late time point, obtained under variation of two parameters $\chi_c^m$ and $c_{1/2}$ that influence the behaviour of macrophages. The data is available [here](https://github.com/JABull1066/MacrophageSensitivityABM/releases/tag/2-param-data).


## Code Structure
- `data_preparation_precomputation.py` contains functions used to read existing file numbers in the given data, to compute magnitude, and to save and read precomputed magnitudes (both local and global) along with local cell counts and (local average) pairwise distances.
- `matplotlib_config.py`contains color palettes and shared configurations for plots.


### Local
- `local_utils.py` contains alone functions used in the local application. In particular, `prep_data` computes the desired feature vectors (depending on the specified method), and the remaining functions plot results.
- `local_run_and_plot.py` runs the local experiments specified at the top. That is, it clusters local neighbourhoods across all simulation outcomes into the specified number of local signatures, using the specified features, and it plots the results in the specified ways (using `run_experiment`). It further runs experiments to analyse cluster quality depending on the number of clusters and the used features (using `run_cluster_quality_experiment`).
- `local_extras.py` contains extra code connected to the local application that is not used to produce results relevant for the above preprint but that could be useful for further work.


### Global
- `global_utils.py` contains alone functions used in the global application. In particular, `give_vector_all_comb_magnitudes` and `give_vector_magnitude_differences` are used to compute feature vectors, `align_labels`, `return_classification` and `classify_schemes` are used to classify parameter schemes and align the labels with a given ground truth, and the remaining functions are used for plotting results.
- `global_run_and_plot.py` runs the global experiments specified at the top. That is, using its function `run_experiment`, it classifies parameter schemes into three long term tumour outcomes, using the specified magnitude features, and it plots the resulting classification together with purities.
- `global_extras.py` contains extra code connected to the global application that is not used to produce results relevant for the above preprint but that could be useful for further work.
- `morans_i.py` contains code used to benchmark pairwise inclusion-exclusion magnitude features with Moran's I features in the global application (in the appendix of the preprint).


## Reproduction of Results
To reproduce results from the above preprint, use Python 3.12.2 and proceed as follows:
- `requirements.txt` contains all necessary Python libraries and can be installed using:
```
pip install -r requirements.txt
```
- Download the data `all2Params_t500.zip` from the GitHub repository of [[1]](#1) [here](https://github.com/JABull1066/MacrophageSensitivityABM/releases/tag/2-param-data), along with the parameter information `params_2ParamSweep.csv`. Unzipping `all2Params_t500.zip` gives a folder named `17082022_all2Params_t500` that contains the individual simulation outcomes.
- If you wish to avoid lengthy computations, make sure to also download the precomputed global magnitudes and/or the precomputed local magnitudes, local cellcounts and average pairwise distances [in this release](https://github.com/j-sollberger/tumor-magnitude/releases/tag/precomputed-magnitudes).
- Make sure the relative data directories at the top of `local_run_and_plot.py` and `global_run_and_plot` are correct. That is:
    -  DATA_DIR must point to the individual data files within `17082022_all2Params_t500` and the parameter information `params_2ParamSweep.csv` must sit in the same parent folder as `17082022_all2Params_t500`.
    - Optionally: LOC_MAG_DIR and MAG_DIR must point to precomputed local and global magnitudes, respectively.

    Set the directory for results as you wish (RESULT_DIR).
- Run `local_run_and_plot.py` to reproduce all local results, and run `global_run_and_plot.py` to reproduce all global results. Run `morans_i.py` to reproduce the comparison with Moran's I features (in the appendix of the preprint).


# References
<a id="1">[1]</a> J. Bull and H. Byrne, _Quantification of spatial and phenotypic heterogeneity in an agent-based model of tumour-macrophage interactions_, PLOS Computational Biology, 3 (2023),
p. e1010994. https://doi.org/10.1371/journal.pcbi.1010994
