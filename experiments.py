import os
import warnings
import random
import functools

from tqdm import tqdm, trange
from fire import Fire

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lifelines

from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

import coxae
from coxae.model import CoxAutoencoderClustering, ConcreteCoxAutoencoderClustering
from coxae.baselines.ae import AutoencoderClustering
from coxae.baselines.maui import MauiClustering
from coxae.baselines.pca import PCAClustering
from coxae.feature_selection import CoxPHFeatureSelector
from coxae.utils import get_kmfs

def dropna(x):
    x = np.array(x)
    return x[~np.isnan(x)]

def get_results(fold_results, metric, evaluation_set):
    return [r[evaluation_set,metric] for r in fold_results]

def main(
        datasets:list[str] = None,
        data_directory_template:str = "./data/hierae_data/processed/{dset}/merged",

        models:list = None,
        num_reps:int = 4,
        n_splits:int = 10,

        figure_formats:list[str] = None,
        figsize:tuple[float,float] = (15,7),

        limit_significant:int = 20,
        deactivate_tqdm:bool = False,
        ):
    datasets = datasets if datasets is not None else ['BLCA', 'BRCA', 'COAD', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC']
    models = models if models is not None else [ConcreteCoxAutoencoderClustering, CoxAutoencoderClustering, MauiClustering, AutoencoderClustering, PCAClustering]
    models = [cls if not issubclass(type(cls), str) else globals()[cls] for cls in models]
    figure_formats = figure_formats if figure_formats is not None else ["png", "pdf", "svg"]
    
    if issubclass(type(datasets), str):
        datasets = [datasets]

    dim_reduction = TSNE
    dim_reduction_label = "TSNE_{}"

    for fmt in figure_formats:
        os.makedirs("./figs/{fmt}/".format(fmt=fmt), exist_ok=True)
    os.makedirs("./results/".format(fmt=fmt), exist_ok=True)

    for dset in tqdm(datasets, desc="dset", disable=deactivate_tqdm):
        tqdm.write(dset)

        data_directory = data_directory_template.format(dset=dset)

        data = pd.read_csv(os.path.join(data_directory, 'new_clinical_multi_omics_with_rppa.csv'))

        data_no_na = data.dropna(axis="columns")
        data_no_na.shape

        survival = data_no_na[["OS","OS.time"]].rename(columns={"OS":"observed","OS.time":"duration"})
        survival.head()

        kmf = lifelines.KaplanMeierFitter(label="Overall Survival")
        kmf.fit(durations=survival["duration"],
                event_observed=survival["observed"])
        kmfall_fig = plt.figure(figsize=figsize)
        kmf.plot(show_censors=1, ci_show=1, at_risk_counts=True)
        for fmt in figure_formats:
            kmfall_fig.savefig("./figs/{fmt}/{dset}_os.{fmt}".format(dset=dset, fmt=fmt))
        kmfall_fig = None

        df_clin = survival
        df_clinical_features = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="clinical"]]
        df_cnv = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="cnv"]]
        df_gex = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="gex"]].apply(lambda x: np.log2(np.where(x<0,np.zeros_like(x),x)+1)) # Max between the value and x is taken due to some gene expression values being less than one for some reason
        df_meth = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="meth"]]
        df_mirna = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="mirna"]].apply(lambda x: np.log2(np.where(x<0,np.zeros_like(x),x)+1))
        df_mutation = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="mutation"]]
        df_rppa = data_no_na[[col for col in data_no_na.columns if col.split("_")[0]=="rppa"]]

        feature_dfs = {
            "clin": df_clinical_features,
            "cnv": df_cnv,
            "gex": df_gex,
            "meth": df_meth,
            "mirna": df_mirna,
            "mut": df_mutation,
            "rppa": df_rppa
        }

        feature_df_list = [feature_dfs[k] for k in feature_dfs]
        df_all = functools.reduce(lambda x,y: x.join(y), feature_df_list[1:], feature_df_list[0])
        X = {k:feature_dfs[k].values for k in feature_dfs}
        durations = df_clin["duration"].values
        events = df_clin["observed"].values

        n_components = 2
        df_plot = pd.DataFrame(dim_reduction(n_components).fit_transform(df_all.values), index=df_all.index, columns=[dim_reduction_label.format(i) for i in range(n_components)])
        df_plot = df_plot.join(df_clin)

        red2dobs_fig = plt.figure(figsize=figsize)
        sns.scatterplot(data=df_plot, x=dim_reduction_label.format(0), y=dim_reduction_label.format(1), hue="observed")
        for fmt in figure_formats:
            red2dobs_fig.savefig("./figs/{fmt}/{dset}_2d_observed.{fmt}".format(dset=dset, fmt=fmt))
        red2dobs_fig = None

        red2ddur_fig = plt.figure(figsize=figsize)
        sns.scatterplot(data=df_plot[df_plot["observed"]==1], x=dim_reduction_label.format(0), y=dim_reduction_label.format(1), hue="duration")
        for fmt in figure_formats:
            red2ddur_fig.savefig("./figs/{fmt}/{dset}_2d_observed_duration.{fmt}".format(dset=dset, fmt=fmt))
        red2ddur_fig = None
        df_plot

        model_fold_results = {
            "model":[],
            "rep":[],
            "fold":[],
            "partition":[],
            "p_value":[],
            "c_index":[],
        }
        try:
            for ModelClass in tqdm(models, desc="model", leave=False, disable=deactivate_tqdm):
                tqdm.write(ModelClass.__name__)
                for rep in trange(num_reps, desc="rep", leave=False, disable=deactivate_tqdm):
                    np.random.seed(rep)
                    random.seed(rep)
                    for fold, (train_index, test_index) in tqdm(enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=rep).split(df_all.values)), total=n_splits, desc="fold", leave=False, disable=deactivate_tqdm):
                        try:
                            model = ModelClass(encoding_feature_selector=CoxPHFeatureSelector(limit_significant=limit_significant, get_most_significant_combination_time_limit=0))
                        except (TypeError, ValueError):
                            model = ModelClass()
                        model.fit({k:X[k][train_index] for k in X}, durations[train_index], events[train_index])
                        clusters = model.cluster(X)
                        hazards = model.hazard(X)
                        for partition, indexes in zip(["All", "Train", "Test"], [np.concatenate([train_index, test_index]), train_index, test_index]):
                            _, p_value = model.logrank_p_score(clusters[indexes], durations[indexes], events[indexes])
                            c_index = model.concordance_index(hazards[indexes], durations[indexes], events[indexes])

                            if not np.isnan(p_value):
                                partition_fig = plt.figure(figsize=figsize)
                                ax = plt.gca()
                                kmfs, _ = get_kmfs(clusters[indexes], durations[indexes], events[indexes])
                                for kmf in kmfs:
                                    kmf.plot(show_censors=1, ci_show=1, ax=ax)
                                lifelines.plotting.add_at_risk_counts(*kmfs, ax=ax)
                                plt.title(
                                    """{model} Fold-{fold}, {partition}-dataset
                                    logrank-p: {p_value:.6e}
                                    concordance-index: {c_index:.6f}""".format(
                                        model=ModelClass.__name__,
                                        fold=fold,
                                        partition=partition,
                                        p_value=p_value,
                                        c_index=c_index,
                                    )
                                )
                                for fmt in figure_formats:
                                    partition_fig.savefig("./figs/{fmt}/{dset}_sep_{model}_{partition}_{rep}_{fold}.{fmt}".format(dset=dset, model = ModelClass.__name__, partition=partition, rep=rep, fold=fold, fmt=fmt))
                                partition_fig = None

                            model_fold_results["model"].append(ModelClass.__name__)
                            model_fold_results["rep"].append(rep)
                            model_fold_results["fold"].append(fold)
                            model_fold_results["partition"].append(partition)
                            model_fold_results["p_value"].append(p_value)
                            model_fold_results["c_index"].append(c_index)
                            pd.DataFrame(model_fold_results).to_csv("./results/{dset}.csv".format(dset=dset))
        except KeyboardInterrupt:
            pd.DataFrame(model_fold_results).to_csv("./results/{dset}.csv".format(dset=dset))
            return
        result_df = pd.DataFrame(model_fold_results)
        result_df.to_csv("./results/{dset}.csv".format(dset=dset))
        for partition in ["All", "Train", "Test"]:
            for metric in ["c_index", "p_value"]:
                results_fig, ax = plt.subplots(figsize=figsize)
                sns.boxplot(data=result_df[result_df["partition"]==partition], x="model", y=metric, ax=ax)
                if metric=="c_index":
                    ax.set_ylim(0, 1)
                for fmt in figure_formats:
                    results_fig.savefig("./figs/{fmt}/{dset}_comparison_{metric}_{partition}.{fmt}".format(dset=dset, metric=metric, partition=partition, fmt=fmt))
                results_fig = None

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.set_theme(style="darkgrid")
        Fire(main)
