# Multi-omics integration with Supervised Autoencoders and Concrete Supervised Autoencoders

This repository is supposed to be a repository containing different multi-omics autoencoder-based models for survival prediction and stratification.

Feel free to contribute to the code. And feel free to open any issues if you encounter any problems.

## LOD2022 Multi-Omic Data Integration and Feature Selection for Survival-based Patient Stratification via Supervised Concrete Autoencoders

For the exact version used for the LOD2022 paper, please see the [LOD2022 branch](https://github.com/phcavelar/multi-omics-lusc-tcga/tree/LOD2022)

### Abstract
Cancer is a complex disease with significant social and economic impact. Advancements in high-throughput molecular assays and the reduced cost for performing high-quality multi-omic measurements have fuelled insights through machine learning . Previous studies have shown promise on using multiple omic layers to predict survival and stratify cancer patients. In this paper, we develop and report a Supervised Autoencoder (SAE) model for survival-based multi-omic integration, which improves upon previous work, as well as a Concrete Supervised Autoencoder model (CSAE) which uses feature selection to jointly reconstruct the input features as well as to predict survival. Our results show that our models either outperform or are on par with some of the most commonly used baselines, while either providing a better survival separation (SAE) or being more interpretable (CSAE). Feature selection stability analysis on our models shows a power-law relationship with features commonly associated with survival.

## Set-up

To set up some of the baselines, you might need to download the following github repositories, and either install or unpack their respective library folders in the root of this repository:
* Maui: https://github.com/phcavelar/maui
* HierAE: https://github.com/phcavelar/hierae

I've used a personal fork to ensure that you can use the same modifications I used. If you use any of the baselines in your work, please cite their respective papers.

## Attribution

If you use this code please cite the following papers:
* Pedro Henrique da Costa Avelar, Roman Laddach, Sophia N Karagiannis, Min Wu, Sophia Tsoka. Multi-Omic Data Integration and Feature Selection for Survival-based Patient Stratification via Supervised Concrete Autoencoders. The 8th International Conference on machine Learning, Optimization and Data science - LOD 2022

## Dependencies

This repository should be run with python>=3.9

```
mamba create -n lusc -c conda-forge -c pytorch -c bioconda "python>=3.9" pypgatk pytorch torchvision torchaudio cpuonly pycox lifelines statsmodels pandas scikit-learn seaborn fire biopython scipy tqdm ipykernel ipywidgets
conda activate lusc
python -m ipykernel install --user --name=lusc
conda deactivate
jupyter notebook
```

To create the environment with the exact tested version for this repository run the following:

```
mamba create -n lusc -c conda-forge -c pytorch -c bioconda "python=3.9.7" "pypgatk=0.0.19" "pytorch=1.10.0" "torchvision=0.10.1" "torchaudio=0.10.0" "cpuonly=2.0" "pycox=0.2.3" "lifelines=0.26.3" "statsmodels=0.13.1" "pandas=1.3.4" "scikit-learn=" "seaborn=0.11.2" "fire=0.4.0" "biopython=1.79" "scipy=1.7.2" "tqdm=4.62.3" "ipykernel=6.5.0" "ipywidgets=7.6.5"
```
