# ECG Classification with 1-D SE-ResNet-18 & Robust Augmentations
A clean PyTorch pipeline for multi-label classification of 12-lead ECGs using a 1-D SE-ResNet-18 backbone with time-domain augmentations.
Originally inspired by PhysioNet/CinC Challenge entries, this refactor generalizes to source-aware analyses across multiple datasets and provides reproducible training, validation, and evaluation utilities.

Challenge background: PhysioNet/CinC 2020/2021 ECG classification. 

SE blocks: Squeeze-and-Excitation Networks (Hu et al.). 

Labeling: SNOMED CT mapping utilities. 

Stratification: Multi-label iterative stratification for fair splits. 

# What’s Inside

1-End-to-end training & evaluation for multi-label ECG diagnosis (14–17 classes, configurable).

2-Source-aware 5-fold stratified protocol preserving per-class prevalence and label co-occurrence.

3-1-D SE-ResNet-18 with optional demographics fusion (age, sex).

4-Augmentations: Multiply–Triangle (scaling), random crop/jitter, optional light noise, optional lead dropout.

5-Metrics: AUROC/AP (micro & macro) + optional PhysioNet/CinC score for compatibility. 

6-Label mapping utilities for converting non-SNOMED labels → SNOMED CT.





# Quick Start

Install the required pip packages from `requirements.txt` file using the following command:

```
pip install -r requirements.txt
```
Tested Python: 3.10.4

Major deps: torch, numpy, scipy, pandas, scikit-learn, torchmetrics, iterative-stratification (for multilabel stratified splits). 
PyPI

Tip: For reproducibility, pin your CUDA-compatible PyTorch build if using GPU.



# Data

See /notebooks/1_introduction_data_handling.ipynb for data access, preprocessing, and segmentation.
This repo is designed to work with consolidated 12-lead ECG corpora (e.g., PhysioNet/CinC Challenge datasets). Always follow dataset licenses and access policies.

# In a nutshell

In case you intend to carry out data preprocessing, you have the option to utilize the preprocess_data.py script. Although this step is not obligatory for utilizing the repository, it's important to note that if certain transformations (such as BandPassFilter) are employed during the training phase, the training process could experience notable delays. To initiate data preprocessing, execute the following command:


# CSV Format


Utilizing the create_data_csvs.py script, you can partition the data through either a stratified split approach or a database-wise split method. In the case of stratified division, the script employs the MultilabelStratifiedShuffleSplit function from the iterative-stratification package. This process generates CSV files containing distinct training and validation sets. These CSV files are subsequently employed during the model's training phase, featuring columns such as path (indicating the ECG recording's file path in .mat format), age, gender, and all relevant diagnoses represented by SNOMED CT codes, which serve as labels for classification. Additionally, the script also generates CSV files for the test data. Conversely, the database-wise split strategy leverages the inherent directory structure from which the data is sourced.


The pipeline expects split CSVs with demographics and SNOMED-coded labels:

| path               | age | gender | 10370003 | 111975006 | 164890007 |  … |
| ------------------ | --: | ------ | -------: | --------: | --------: | -: |
| `./Data/A0002.mat` |  49 | Female |        0 |         0 |         1 |  … |
| `./Data/A0003.mat` |  81 | Female |        0 |         1 |         1 |  … |
| `./Data/A0004.mat` |  45 | Male   |        1 |         0 |         0 |  … |

SNOMED CT is a comprehensive clinical terminology used for robust, interoperable labeling.


# In a Nutshell


```
python preprocess_data.py
```
Note: If you use heavy transforms (e.g., band-pass filtering) during training, expect slower epochs.

Consider checking the `configs` directory for yaml configurations:

* Yaml files in the `training` directory are used to train a model
* Yaml files in the `predicting` directory are used to test and evaluate a model

Two notebooks are available for creating training and testing yaml files based on the data splitting performed with the `create_data_csvs.py` script: [Yaml files of database-wise split for training and testing](/notebooks/2_physionet_DBwise_yaml_files.ipynb) and [Yaml files of stratified split for training and testing](/notebooks/2_physionet_stratified_yaml_files.ipynb). Be sure to perform the data splitting first.

1) For splitting the data into training and testing sets for model usage, employ the subsequent command.

```
python create_data_split_csvs.py
```

Stratified split uses MultilabelStratifiedShuffleSplit to preserve label co-occurrence.


Learn more in /notebooks/2_physionet_DBwise_yaml_files.ipynb and /notebooks/2_physionet_stratified_yaml_files.ipynb.
Background on multilabel iterative stratification: PyPI and original implementations.


Note! There are attributes to be considered *before* running the script. Check the notebook [Introduction to data handling](/notebooks/1_introduction_data_handling.ipynb) for further instructions. 

2) In order to initiate model training, you have the option to provide either a yaml file or a directory as an argument, followed by the execution of one of the subsequent commands.

```
python train_model.py train_smoke.yaml
python train_model.py train_stratified_smoke
```

# Configure

Check the configs/ directory:

```
configs/training/*.yaml → training configs

configs/predicting/*.yaml → evaluation configs
```

# Train

Pass a single YAML file or a directory of YAMLs:

```
# Single config
python train_model.py train_smoke.yaml

# Directory (runs each YAML separately)
python train_model.py train_stratified_smoke
```
See /notebooks/3_introduction_training.ipynb for details.

# Evaluate / Predict

```
# Single config
python run_model.py predict_smoke.yaml

# Directory of configs
python run_model.py predict_stratified_smoke
```

See /notebooks/4_introduction_testing_evaluation.ipynb.

# Model & Augmentations

## 1-D SE-ResNet-18


1-Residual blocks operate along time, producing lead-wise feature maps.

2-Each block integrates a Squeeze-and-Excitation (SE) module (global temporal pooling + channel re-weighting) prior to the skip addition. 

3-Optional demographics fusion (age, sex) concatenated to pooled features before the classification head.

4-Output head uses independent sigmoids (one per class) with class-weighted BCE for imbalance.

## Augmentations

1-Multiply–Triangle scaling (proposed): piecewise amplitude scaling to strengthen robustness.

Random crop

Optional light noise

Optional lead dropout
.
.
.
**Sinuside Time-Amplitude Resampling**

All augmentations are configurable via YAMLs. Keep evaluation strictly augmentation-free.

## Metrics

1-AUROC (micro & macro), Average Precision (micro & macro) via torchmetrics/sklearn.

# Project Structure


```
.
├── configs
│   ├── data_splitting/      # DB-wise and stratified split YAMLs
│   ├── predicting/          # Evaluation/prediction YAMLs
│   └── training/            # Training YAMLs
│
├── data
│   ├── smoke_data/          # Sample ECGs for smoke testing
│   └── split_csvs/          # Output CSVs from splitting scripts
│
├── notebooks/               # EDA + “how to use” notebooks
│
├── src
│   ├── dataloader/
│   │   ├── dataset.py       # ECG Dataset / DataLoader
│   │   ├── dataset_utils.py # Preprocessing helpers
│   │   └── transforms.py    # Time-domain transforms
│   └── modeling/
│       ├── models/
│       │   └── seresnet18.py    # 1-D SE-ResNet-18
│       ├── metrics.py           # AUROC/AP, optional CinC score
│       ├── predict_utils.py     # Inference helpers
│       └── train_utils.py       # Optimizer, loss, loop, callbacks
│
├── label_mapping.py         # Map dataset labels → SNOMED CT
├── create_data_split_csvs.py# Build stratified / DB-wise CSV splits
├── preprocess_data.py       # Optional preprocessing
├── run_model.py             # Evaluate / test
├── train_model.py           # Train
├── utils.py                 # YAML & misc utilities
├── requirements.txt
├── LICENSE / LICENSE.txt
└── README.md


```

# Reproducibility Tips

1-Fix random seeds (data loader, torch, numpy).

2-Log versions of all dependencies.

3-Prefer deterministic ops when feasible.

4-Keep source-aware splits to avoid leakage across sites.





