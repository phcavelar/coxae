from genericpath import isfile
import os
import shutil
import io
import urllib
import requests
import json
import tarfile
import re
import functools

import pandas as pd

from pypgatk.cgenomes.cbioportal_downloader import CbioPortalDownloadService

def download_maui_data(
        output_directory = "./data/maui_data",
        **kwargs
        ):
    """Downloads the data used in Maui's vignette. Taken from: https://github.com/BIMSBbioinfo/maui/blob/7d329c736b681216093fd725b134a68e6e914c8e/vignette/maui_vignette.ipynb
    """
    
    os.makedirs(output_directory, exist_ok=True)

    if not os.path.isfile(os.path.join(output_directory, 'cnv.csv')):
        urllib.request.urlretrieve(
            url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_cnv.csv',
            filename=os.path.join(output_directory, 'cnv.csv')
        )

    if not os.path.isfile(os.path.join(output_directory, 'gex.csv')):
        urllib.request.urlretrieve(
            url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_gex.csv',
            filename=os.path.join(output_directory, 'gex.csv')
        )

    if not os.path.isfile(os.path.join(output_directory, 'mut.csv')):
        urllib.request.urlretrieve(
            url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_mut.csv',
            filename=os.path.join(output_directory, 'mut.csv')
        )

    if not os.path.isfile(os.path.join(output_directory, 'subtypes.csv')):
        urllib.request.urlretrieve(
            url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_subtypes.csv',
            filename=os.path.join(output_directory, 'subtypes.csv')
        )

    if not os.path.isfile(os.path.join(output_directory, 'survival.csv')):
        urllib.request.urlretrieve(
            url='http://bimsbstatic.mdc-berlin.de/akalin/maui-data/coread_for_paper_survival.csv',
            filename=os.path.join(output_directory, 'survival.csv')
        )

## LUSC data

def untar_study(study, output_directory, fmt=".tar.gz"):
    file = tarfile.open(
        os.path.join(
            output_directory,
            "{study}{format}".format(study=study, format=fmt)
            )
        )
    file.extractall(output_directory)
    file.close()

def download_cbioportal_study(
        config_file,
        study,
        output_directory = "./data",
        list_studies = False,
        multithreading = True,
        **kwargs
        ):
    pipeline_arguments = {
        CbioPortalDownloadService.CONFIG_OUTPUT_DIRECTORY: output_directory,
        CbioPortalDownloadService.CONFIG_LIST_STUDIES: list_studies,
        CbioPortalDownloadService.CONFIG_MULTITHREADING: multithreading,
        **kwargs,
    } 
    cbioportal_downloader_service = CbioPortalDownloadService(config_file, pipeline_arguments)
    cbioportal_downloader_service.download_study(study)

def get_mirna_files(
        project_id="TCGA-LUSC",
        maxfiles=10000,
        cases_endpt = "https://api.gdc.cancer.gov/files",
        data_endpt = "https://api.gdc.cancer.gov/data"
        ):


    # Retrieve associated file names
    filters = {
        "op": "and",
        "content":[
            {"op": "=",
            "content":{
                "field": "cases.project.project_id",
                "value": ["TCGA-LUSC"]
                }
            },
            {"op": "=",
            "content":{
                "field": "files.experimental_strategy",
                "value": ["miRNA-Seq"]
                }
            },
            {"op": "=",
            "content":{
                "field": "files.data_category",
                "value": ["Transcriptome Profiling"]
                }
            },
            {"op": "=",
            "content":{
                "field": "files.data_type",
                "value": ["miRNA Expression Quantification"]
                }
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(["cases.samples.submitter_id","file_name", "Entity ID"]),
        "format": "TSV",
        "size": str(maxfiles)
    }

    response = requests.get(cases_endpt, params = params)
    files_df = pd.read_csv(io.StringIO(response.text), sep="\t")
    return files_df

def download_mirna_files(
        files_df,
        output_directory = "./data",
        project_id="TCGA-LUSC",
        maxfiles=10000,
        cases_endpt = "https://api.gdc.cancer.gov/files",
        data_endpt = "https://api.gdc.cancer.gov/data"
        ):
    
    params = {"ids": files_df["id"].tolist()}

    response = requests.post(data_endpt,
                            data = json.dumps(params),
                            headers={
                                "Content-Type": "application/json"
                                })

    response_head_cd = response.headers["Content-Disposition"]
    file_name = re.findall("filename=(.+)", response_head_cd)[0]
    with open(os.path.join(output_directory, file_name), "wb") as output_file:
        output_file.write(response.content)
    return file_name

def untar_and_merge_mirna_files(
        files_df,
        file_name,
        output_directory = "./data",
        cleanup=True
        ):
    untar_study(file_name, output_directory, fmt="")
    
    miRNA_IDs = set()
    patient_dfs = {}
    patient_folders = []
    for i in range(len(files_df)):
        patient_id = files_df["cases.0.samples.0.submitter_id"].iloc[i]
        foldername = files_df["id"].iloc[i]
        patient_fname = os.listdir(os.path.join(output_directory, foldername))[0]
        patient_df = pd.read_csv(os.path.join(output_directory, foldername, patient_fname), sep="\t")
        
        miRNA_IDs.update(patient_df["miRNA_ID"].tolist())
        patient_dfs[patient_id] = patient_df
        patient_folders.append(foldername)
    
    miRNA_df = pd.DataFrame({"patient_id":[], **{k:[] for k in miRNA_IDs}}).set_index("patient_id")
    for patient_id in patient_dfs:
        patient_df = patient_dfs[patient_id]
        cbioportal_patient_id = patient_id[:-1] # This line is to match cBioPortal's format
        transposed_patient_df = patient_df[["miRNA_ID","reads_per_million_miRNA_mapped"]].set_index("miRNA_ID").transpose()
        transposed_patient_df["patient_id"] = [cbioportal_patient_id]
        transposed_patient_df = transposed_patient_df.set_index("patient_id")
        miRNA_df.loc[cbioportal_patient_id,miRNA_df.columns] = transposed_patient_df[miRNA_df.columns].values.flatten()
        
    if cleanup:
        for patient_folder in patient_folders:
            shutil.rmtree(os.path.join(output_directory, patient_folder))
    
    return miRNA_df

def download_lusc_data(
        output_directory = "./data/lusc_data",
        temp_directory = "./.tmp",
        cbioportal_config = "./config/cbioportal_config.yaml",
        study_name = "lusc_tcga",
        **kwargs
        ):
    os.makedirs(output_directory, exist_ok=True)

    missing_cbioportal = any(not os.path.isfile(os.path.join(output_directory, fname)) for fname in ["rnaseq.csv", "linearcna.csv", "methylation.csv", "survival.csv"])
    missing_mirna = not os.path.isfile(os.path.join(output_directory, "mirna.csv"))
    if missing_cbioportal or missing_mirna:
        download_cbioportal_study(cbioportal_config, study_name, temp_directory)
        try:
            untar_study(study_name, temp_directory, fmt=".tar")
        except FileNotFoundError:
            # Sometimes the downloaded file comes without gunzip compression
            untar_study(study_name, temp_directory, fmt=".tar.gz")
        try:
            survival = pd.read_csv(os.path.join(temp_directory, study_name, "data_clinical_patient.txt"), comment="#", sep="\t")
        except FileNotFoundError:
            # Sometimes the downloaded files have different filenames
            survival = pd.read_csv(os.path.join(temp_directory, study_name, "data_bcr_clinical_data_patient.txt"), comment="#", sep="\t")
        survival = survival.set_index("PATIENT_ID").drop(columns="OTHER_PATIENT_ID")
        # Get the first sample for each patient
        survival = survival.set_index(survival.index + "-01")
        # Drop unused columns
        survival = survival[["OS_STATUS", "OS_MONTHS"]]
        # Add a binary column with the status
        survival["OS_STATUS"] = survival["OS_STATUS"]=="1:DECEASED"
        survival["OS_MONTHS"] = pd.to_numeric(survival["OS_MONTHS"], errors="coerce")
        survival = survival.dropna()
        survival.index = survival.index.rename("patient_id")
        survival = survival.rename({"OS_STATUS": "observed", "OS_MONTHS": "duration"}, axis='columns')

        try:
            rnaseq = pd.read_csv(os.path.join(temp_directory, study_name, "data_mrna_seq_v2_rsem.txt"), comment="#", sep="\t")
        except FileNotFoundError:
            # Sometimes the downloaded file has a different name
            rnaseq = pd.read_csv(os.path.join(temp_directory, study_name, "data_RNA_Seq_v2_expression_median.txt"), comment="#", sep="\t")
        try:
            linearcna = pd.read_csv(os.path.join(temp_directory, study_name, "data_linear_cna.txt"), comment="#", sep="\t")
        except FileNotFoundError:
            # Sometimes the downloaded file has a different name
            linearcna = pd.read_csv(os.path.join(temp_directory, study_name, "data_linear_CNA.txt"), comment="#", sep="\t")
        # The methylation file had a consistent filename along the runs, so no try-catch is attempted.
        methylation = pd.read_csv(os.path.join(temp_directory, study_name, "data_methylation_hm450.txt"), comment="#", sep="\t")

        rnaseq = rnaseq.set_index("Hugo_Symbol").drop(columns="Entrez_Gene_Id").dropna(axis="rows")
        rnaseq = rnaseq.rename(lambda x: "rnaseq_{}".format(x), axis="rows")
        linearcna = linearcna.set_index("Hugo_Symbol").drop(columns="Entrez_Gene_Id").dropna(axis="rows")
        linearcna = linearcna.rename(lambda x: "linearcna_{}".format(x), axis="rows")
        methylation = methylation.set_index("Hugo_Symbol").drop(columns="Entrez_Gene_Id").dropna(axis="rows")
        methylation = methylation.rename(lambda x: "methylation_{}".format(x), axis="rows")

        mirna_files_df = get_mirna_files()
        mirna_fname = download_mirna_files(mirna_files_df, temp_directory)
        mirna = untar_and_merge_mirna_files(mirna_files_df, mirna_fname, temp_directory).T.dropna(axis="rows")
    
        # Limit the files to the values existing in all layers
        survival_patients = set(survival.index)
        rnaseq_patients = set(rnaseq.T.index)
        linearcna_patients = set(linearcna.T.index)
        methylation_patients = set(methylation.T.index)
        mirna_patients = set(mirna.T.index)
        all_sets = [survival_patients, rnaseq_patients, linearcna_patients, methylation_patients, mirna_patients]
        all_patients = functools.reduce(lambda x, y: x|y, all_sets, set())
        patients_on_all_datasets = functools.reduce(lambda x, y: x&y, all_sets, all_patients)

        survival = survival.loc[patients_on_all_datasets]
        rnaseq = rnaseq.T.loc[patients_on_all_datasets].T
        linearcna = linearcna.T.loc[patients_on_all_datasets].T
        methylation = methylation.T.loc[patients_on_all_datasets].T
        mirna = mirna.T.loc[patients_on_all_datasets].T

        survival.to_csv(os.path.join(output_directory, "survival.csv"), index_label="patient_id")
        rnaseq.to_csv(os.path.join(output_directory, "rnaseq.csv"), index_label="rnaseq")
        linearcna.to_csv(os.path.join(output_directory, "linearcna.csv"), index_label="linearcna")
        methylation.to_csv(os.path.join(output_directory, "methylation.csv"), index_label="methylation")
        mirna.to_csv(os.path.join(output_directory, "mirna.csv"), index_label="mirna")

        shutil.rmtree(os.path.join(temp_directory))

    