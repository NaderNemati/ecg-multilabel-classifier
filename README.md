# ECG Classification using Deep Learning (SE-ResNet18 + Augmentations)
This repository provides a PyTorch pipeline for multi-label classification of 12-lead ECGs using a 1-D SE-ResNet18 backbone plus time-domain data augmentations.
It is refactored from a PhysioNet/CinC 2020 entry and updated for broader, source-aware analysis across multiple datasets.
This version is refactored for more general analysis.

Challenge background: PhysioNet/CinC 2020/2021 ECG classification challenges. 

SE blocks: Squeeze-and-Excitation Networks (Hu et al., CVPR 2018).

# What’s included

1-End-to-end training & evaluation for multi-label ECG diagnosis (14–17 classes depending on config).

2-Source-aware 5-fold stratified protocol that preserves per-class prevalence and label co-occurrence.

3-SE-ResNet18 (1-D) with demographics fusion (age, sex).

4-Augmentations: Multiply–Triangle scaling (proposed), random crop/jitter, optional light noise, optional lead dropout.

5-Metrics: AUROC/AP (micro & macro) and optional PhysioNet/CinC score.

6-Label mapping utilities to convert non-SNOMED labels to SNOMED CT.





# Usage

Install the required pip packages from `requirements.txt` file using the following command:

```
pip install -r requirements.txt
```

Recommended Python version 3.10.4 (tested with Python 3.10.4).


# Data

Feel free to explore the notebook titled Getting Started with Data Management located within the /notebooks/ directory. This resource provides comprehensive details about data retrieval, preprocessing, and segmentation processes.


# In a nutshell

In case you intend to carry out data preprocessing, you have the option to utilize the preprocess_data.py script. Although this step is not obligatory for utilizing the repository, it's important to note that if certain transformations (such as BandPassFilter) are employed during the training phase, the training process could experience notable delays. To initiate data preprocessing, execute the following command:

```
python preprocess_data.py
```

Consider checking the `configs` directory for yaml configurations:

* Yaml files in the `training` directory are used to train a model
* Yaml files in the `predicting` directory are used to test and evaluate a model

Two notebooks are available for creating training and testing yaml files based on the data splitting performed with the `create_data_csvs.py` script: [Yaml files of database-wise split for training and testing](/notebooks/2_physionet_DBwise_yaml_files.ipynb) and [Yaml files of stratified split for training and testing](/notebooks/2_physionet_stratified_yaml_files.ipynb). Be sure to perform the data splitting first.

1) For splitting the data into training and testing sets for model usage, employ the subsequent command.

```
python create_data_split_csvs.py
```

Utilizing the create_data_csvs.py script, you can partition the data through either a stratified split approach or a database-wise split method. In the case of stratified division, the script employs the MultilabelStratifiedShuffleSplit function from the iterative-stratification package. This process generates CSV files containing distinct training and validation sets. These CSV files are subsequently employed during the model's training phase, featuring columns such as path (indicating the ECG recording's file path in .mat format), age, gender, and all relevant diagnoses represented by SNOMED CT codes, which serve as labels for classification. Additionally, the script also generates CSV files for the test data. Conversely, the database-wise split strategy leverages the inherent directory structure from which the data is sourced.

The main structure of csv files are as follows:


| path  | age  | gender  | 10370003  | 111975006 | 164890007 | *other diagnoses...* |
| ------------- |-------------|-------------| ------------- |-------------|-------------|-------------|
| ./Data/A0002.mat | 49.0 | Female | 0 | 0 | 1 | ... |
| ./Data/A0003.mat | 81.0 | Female | 0 | 1 | 1 | ... |
| ./Data/A0004.mat | 45.0 |  Male  | 1 | 0 | 0 | ... |
| ... | ... |  ...  | ... | ... | ... | ... |

Note! There are attributes to be considered *before* running the script. Check the notebook [Introduction to data handling](/notebooks/1_introduction_data_handling.ipynb) for further instructions. 

2) In order to initiate model training, you have the option to provide either a yaml file or a directory as an argument, followed by the execution of one of the subsequent commands.

```
python train_model.py train_smoke.yaml
python train_model.py train_stratified_smoke
```

where `train_data.yaml` consists of needed arguments for the training in a yaml format, and `train_multiple_smoke` is a directory containing several yaml files. When using multiple yaml files at the same time, each yaml file is loaded and run separately. More detailed information about training is available in the notebook [Introduction to training models](/notebooks/3_introduction_training.ipynb).

3) To test and evaluate a trained model, you'll need one of the following commands

```
python run_model.py predict_smoke.yaml
python run_model.py predict_stratified_smoke
```

The train_data.yaml file includes the essential training arguments in YAML format, while the train_multiple_smoke directory encompasses numerous YAML files. When employing multiple YAML files concurrently, each file is loaded and executed individually. For comprehensive training details, you can refer to the accompanying notebook. [Introduction to testing and evaluating models](/notebooks/4_introduction_testing_evaluation.ipynb).


# Repository in details

```
.
├── configs                      
│   ├── data_splitting           # Yaml files considering a database-wise split and a stratified split   
│   ├── predicting               # Yaml files considering the prediction and evaluation phase
│   └── training                 # Yaml files considering the training phase
│   
├── data
│   ├── smoke_data               # Samples from the Physionet 2021 Challenge data as well as
|   |                              Shandong Provincial Hospital data for smoke testing
│   └── split_csvs               # Csv files of ECGs, either database-wise or stratified splitted
│
├── notebooks                    # Jupyter notebooks for data exploration and 
│                                  information about the use of the repository
├── src        
│   ├── dataloader 
│   │   ├── __init__.py
│   │   ├── dataset.py           # Script for custom DataLoader for ECG data
│   │   ├── dataset_utils.py     # Script for preprocessing ECG data
│   │   └── transforms.py        # Script for tranforms
│   │
│   └── modeling 
│       ├── models               # All model architectures
│       │   └── seresnet18.py    # PyTorch implementation of the SE-ResNet18 model
│       ├──__init__.py
│       ├── metrics.py           # Script for evaluation metrics
│       ├── predict_utils.py     # Script for making predictions with a trained model
│       └── train_utils.py       # Setting up optimizer, loss, model, evaluation metrics
│                                  and the training loop
│
├── .gitignore
├── label_mapping.py             # Script to convert other diagnostic codes to SNOMED CT Codes
├── LICENSE
├── LICENSE.txt
├── __init__.py
├── create_data_csvs.py          # Script to perform database-wise data split or split by
│                                  the cross-validatior ´Multilabel Stratified ShuffleSplit´ 
├── preprocess_data.py           # Script for preprocessing data
├── README.md
├── requirements.txt             # The requirements needed to run the repository
├── run_model.py                # Script to test and evaluate a trained model
├── train_model.py               # Script to train a model
└── utils.py                     # Script for yaml configuration

```
