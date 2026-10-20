<h1 align="center">AIO Part I</h1>

<p align="center">
  <strong>Interpretable ADMET Prediction for Cyclic Peptides and Macrocycles</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Macrocycles-506D84?style=flat-square" alt="Macrocycles">
  <img src="https://img.shields.io/badge/Interpretable_AI-537D65?style=flat-square" alt="Interpretable AI">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#key-features">Key Features</a> ·
  <a href="#model-access">Model Access</a> ·
  <a href="#installation-and-setup">Installation</a> ·
  <a href="#running-inference">Inference</a> ·
  <a href="#reproducing-figures-and-results">Reproducibility</a> ·
  <a href="#data-and-preprocessing">Data</a> ·
  <a href="#citation">Citation</a> ·
  <a href="#license-and-usage">License &amp; Usage</a> ·
  <a href="#contact">Contact</a>
</p>

---

AIO is an interpretable deep learning framework for multi-endpoint ADMET prediction of cyclic peptides and macrocycles. It integrates molecular structure, solvent-dependent conformations, and physicochemical features to predict ADMET properties and guide molecular optimization.

This repository provides pretraining datasets, preprocessing scripts for permeability, reproducibility resources, and licensed inference scripts associated with **AIO Part I**.

## Overview

AIO supports the prediction of multiple ADMET properties, including:

| Category | Endpoints |
|:---|:---|
| Absorption | Solubility, PAMPA, Caco-2, RRCK, MDCK permeability |
| Distribution | Plasma unbound fraction, volume of distribution |
| Metabolism and stability | Plasma stability, liver microsomal intrinsic clearance |
| Pharmacokinetics | Systemic clearance, terminal half-life, oral bioavailability |
| Toxicity | Hemolysis and cytotoxicity |

The framework also supports substructure-level interpretability to help identify molecular regions associated with predicted ADMET properties.

## Key Features

- **Multi-endpoint prediction:** covers absorption, distribution, metabolism, pharmacokinetics, and toxicity endpoints.
- **Macrocycle-aware modeling:** combines molecular structure, solvent-dependent conformations, and physicochemical descriptors.
- **Interpretable outputs:** highlights substructures associated with predicted ADMET properties.
- **Reproducible workflows:** includes datasets, preprocessing scripts, saved results, and figure-reproduction resources.
- **Licensed inference:** distributes encrypted model packages for credentialed use.

## Model Access

> **Important:** Model inference requires an access key and license issued by the authors.

The encrypted model packages can be downloaded, but valid credentials are required to run inference. Licenses may have an expiration date.

Public datasets, preprocessing scripts, saved prediction results, and figure reproduction resources are provided separately to support reproducibility.

A web-based inference platform may be made available in the future.

## Installation and Setup

1. Obtain an access key and license from the authors.
2. Clone this repository.
3. Download encrypted_models.zip and data.zip from Zenodo (DOI: 10.5281/zenodo.23486930) and extract both archives into the repository root directory, resulting in `encrypted_models/*.encpkg` and the `data/` directory.
4. Extract cached molecular graphs for the validation tasks.
5. Install the required dependencies.

```bash
pip install -r requirements.txt
```

For validation, extract both graph caches:

```bash
mkdir -p data/permeability_data/pampa_cache/graph
mkdir -p data/small_mol_PharmaBench/cache/graph
mkdir -p data/external_five_case/cache/graph

tar -xzf data/permeability_data/pampa_cache/graph_pt_files.tar.gz \
  -C data/permeability_data/pampa_cache/graph

tar -xzf data/small_mol_PharmaBench/cache/graph_pt_files.tar.gz \
  -C data/small_mol_PharmaBench/cache/graph

tar -xzf data/external_five_case/cache/graph_pt_files.tar.gz \
  -C data/external_five_case/cache/graph
```

> **Tip:** GPU acceleration is recommended for inference. Ensure that the installed PyTorch version is compatible with your CUDA environment.

## Running Inference

Previously generated results are provided under `run_test/`. Expand any workflow below to view the command used to regenerate its predictions with valid credentials.

<details>
<summary><strong>1. External Case Studies</strong></summary>

Reproduce predictions for the five external macrocyclic compounds investigated in the study.

```bash
ACCESS_DIR=/path/to/key/folder/

python predict_five_external_cases.py \
  --license "$ACCESS_DIR/license.dat" \
  --key-dir "$ACCESS_DIR/endpoint_keys" \
  --encrypted-model-dir encrypted_models \
  --results-dir run_test/external_cases \
  --device cuda:0
```

</details>

<details>
<summary><strong>2. PAMPA Permeability Validation</strong></summary>

```bash
ACCESS_DIR=/path/to/key/folder/

python predict_pampa.py \
  --license "$ACCESS_DIR/license.dat" \
  --key-dir "$ACCESS_DIR/endpoint_keys" \
  --encrypted-model-dir encrypted_models \
  --results-dir run_test/pampa \
  --device cuda:0
```

</details>

<details>
<summary><strong>3. Small-Molecule Generalization</strong></summary>

Evaluate the transferability of AIO to human liver microsomal intrinsic clearance prediction using the PharmaBench dataset.

```bash
ACCESS_DIR=/path/to/key/folder/

python predict_small_mol.py \
  --license "$ACCESS_DIR/license.dat" \
  --key-dir "$ACCESS_DIR/endpoint_keys" \
  --encrypted-model-dir encrypted_models \
  --results-dir run_test/small_mol_PharmaBench \
  --device cuda:0
```

</details>

## Reproducing Figures and Results

Scripts, notebooks, and saved results for reproducing key analyses in the paper are available under `./fig_reproduce/`.

The following notebooks can be run directly with the provided data:

- `figure3_table2_reproduced.ipynb` — reproduction of Figure 3 and Table 2.
- `nonspecific_binding_for_clearance.ipynb` — analysis of nonspecific binding in clearance measurements (see DOI 10.17632/3bs5526htd.1 for more information about dataset).

To regenerate distribution and segmentation plots for icotrokinra:

```bash
python plot_for_icotrokinra_F.py
```

Some figures can be reproduced directly from saved predictions without rerunning encrypted model inference.

## Data and Preprocessing

Datasets and associated preprocessing scripts are available under `./data/`.

### 1. Macrocycle Pretraining

Raw datasets are provided under `./data/pretrain_data/`.

- `define_macrocycles.py` — macrocycle identification and filtering.
- `merge_analysis.ipynb` — dataset integration, analysis, and preprocessing.

### 2. Permeability Data

Raw datasets are provided under `./data/permeability_data/`.

- `cycpeptmpdb_analysis.ipynb` — deduplication, integration, and extraction of PAMPA, Caco-2, RRCK, and MDCK permeability measurements from [CycPeptMPDB](http://cycpeptmpdb.com/).
- `merge_analysis.ipynb` — filtering and integration of measurements from CycPeptMPDB and [NPMMP](https://swemacrocycledb.com/).

### 3. Human Liver Microsomal Stability

The processed human liver microsomal intrinsic clearance dataset is derived from [PharmaBench](https://github.com/mindrank-ai/PharmaBench).

Run `data_prep.ipynb` to generate InChIKeys and prepare the dataset splits.

## Citation

If you use AIO, its associated datasets, or the provided analysis workflows, please cite:

**AIO Part I: Interpretable ADMET Prediction for Cyclic Peptides and Macrocycles**

*Citation details will be updated following manuscript availability.*

## License and Usage

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

Under this license, the material may be shared and adapted for non-commercial purposes with appropriate attribution. Commercial use requires prior written permission from the authors.

Model inference additionally requires valid author-issued credentials. Access to the source code or datasets does not grant access to the encrypted model packages or permission to run model inference.

For commercial licensing or other permissions, please contact the authors.

## Contact

For access requests, reproducibility questions, or technical issues, please contact the authors or open an issue in this repository.
