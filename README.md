# Multi-omics integration with Supervised Autoencoders and Concrete Supervised Autoencoders

This repository is supposed to be a repository containing different multi-omics autoencoder-based models for survival prediction and stratification.

Feel free to contribute to the code. And feel free to open any issues if you encounter any problems.

## LOD2022 Multi-Omic Data Integration and Feature Selection for Survival-based Patient Stratification via Supervised Concrete Autoencoders

For the exact version used for the LOD2022 paper, please see the [LOD2022 branch](https://github.com/phcavelar/multi-omics-lusc-tcga/tree/LOD2022)

### Abstract
Cancer is a complex disease with significant social and economic impact. Advancements in high-throughput molecular assays and the reduced cost for performing high-quality multi-omic measurements have fuelled insights through machine learning . Previous studies have shown promise on using multiple omic layers to predict survival and stratify cancer patients. In this paper, we develop and report a Supervised Autoencoder (SAE) model for survival-based multi-omic integration, which improves upon previous work, as well as a Concrete Supervised Autoencoder model (CSAE) which uses feature selection to jointly reconstruct the input features as well as to predict survival. Our results show that our models either outperform or are on par with some of the most commonly used baselines, while either providing a better survival separation (SAE) or being more interpretable (CSAE). Feature selection stability analysis on our models shows a power-law relationship with features commonly associated with survival.

## Set-up

To set up some of the baselines, you might need to download the following github repositories, and either install or unpack their respective library folders in the root of this repository:
* https://github.com/phcavelar/maui

I've used a personal fork to ensure that you can use the same modifications I used. If you use any of the baselines in your work, please cite their respective papers.

## Attribution

If you use this code please cite the following papers:
* Pedro Henrique da Costa Avelar, Roman Laddach, Sophia N Karagiannis, Min Wu, Sophia Tsoka. Multi-Omic Data Integration and Feature Selection for Survival-based Patient Stratification via Supervised Concrete Autoencoders. The 8th International Conference on machine Learning, Optimization and Data science - LOD 2022

## Dependencies

```
mamba create -n lusc -c conda-forge -c pytorch -c bioconda pypgatk pytorch torchvision torchaudio cpuonly lifelines statsmodels pandas scikit-learn seaborn fire biopython scipy tqdm ipykernel ipywidgets
conda activate lusc
python -m ipykernel install --user --name=lusc
conda deactivate
jupyter notebook
```

