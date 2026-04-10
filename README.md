# Magnitude-based features for multispecies data
This repository contains the code used in the preprint TODO.


### Description
We use magnitude based features to analyze synthetic data of the tumour microenvironment in two different manners: (1) We apply magnitude __locally__ to create distinct local signatures that capture spatial heterogeneity, and (2) we apply magnitude __globally__ to classify underlying parameter schemes into three long-term tumour outcomes: elimination, equilibrium and escape. In the local application (1) we use simple magnitude feature vectors and further compare our method to simply summary statistics such as cell counts and average pairwise distances per cell type. In the global application (2) we use more intricate magnitude features and compare results with those from simpler magnitude features.


### Data
We use outputs from an agent based model that models the tumor microenvironment by Joshua A. Bull and Helen M. Byrne [[1]](#1).Specifically, we work with simulation outputs at a single late time point, obtained under variation of two parameters $\Chi_c^m$ and $c_{1/2}$ that influence the behaviour of macrophages.


## Code Structure
- `data_preparation_precomputation.py` contains functions used to read existing file numbers in the given data, to compute magnitude, and to save and read precomputed magnitudes (both local and global) along with local cell counts and pairwise distances.
- `matplotlib_config.py`contains color palettes and shared configurations for plots.


### Local
- `local_utils.py` contains alone functions used in the local application.
- `local_run_and_plot.py` runs the local experiments specified at the top. That is, it clusters local neighbourhoods across all simulation outcomes into the specified number of local signatures, using the specified features, and it plots the results in the specified ways. It further runs experiments to analyse cluster quality depending on the number of clusters and the used features.
- `local_extras.py` contains extra code connected to the local application that is not used to produce results relevant for the above preprint but that could be useful for further work.


### Global
- `global_utils.py` contains alone functions used in the global application.
- `global_run_and_plot.py` runs the global experiments specified at the top. That is, it classifies parameter schemes into three long term tumour outcomes, using the specified magnitude features, and it plots the resulting classification together with purities.
- `global_extras.py` contains extra code connected to the global application that is not used to produce results relevant for the above preprint but that could be useful for further work.


## Reproduction of Results
To reproduce results from the above preprint, proceed as follows:
- `requirements.txt` contains all necessary Python libraries and can be installed using:
```
pip install -r requirements.txt
```
- Download the data in the folder `Data > 17082022_all2Params_t500`. If you wish to avoid lengthy computations, make sure to also download the precomputed global magnitudes in the folder `Data > savedMagnitudes`, and the precomputed local magnitudes, local cellcounts and pairwise distances in the folder `Data > savedLocalMagnitudes`.
- Make sure the relative data directories at the top of `local_run_and_plot.py` and `global_run_and_plot` are correct (DATA_DIR for the data, and LOC_MAG_DIR or MAG_DIR for local or global precomputed magnitudes) and set the directory for results as you wish (RESULT_DIR).
- Run `local_run_and_plot.py` to reproduce all local results, and run `global_run_and_plot.py` to reproduce all global results.


# References
<a id="1">[1]</a> Bull JA, Byrne HM (2023) Quantification of spatial and phenotypic heterogeneity in an agent-based model of tumour-macrophage interactions. PLoS Comput Biol 19(3): e1010994. https://doi.org/10.1371/journal.pcbi.1010994