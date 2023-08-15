# ECG Classification using Deep Learning

This repository is dedicated to the classification of ECG signals using deep learning techniques. The original version of this repository is available [here](https://github.com/ZhaoZhibin/Physionet2020model). It features a PyTorch implementation of the ResNet model developed by *Between_a_ROC_and_a_heart_place* for the PhysioNet/Computing in Cardiology Challenge 2020. The associated paper, titled "[**Adaptive Lead Weighted ResNet Trained With Different Duration Signals for Classifying 12-lead ECGs**](https://physionetchallenges.github.io/2020/papers/112.pdf)," was accepted by CinC2020.

This version of the repository has been refactored to accommodate a broader range of analysis tasks.

## Usage

To begin, install the required Python packages using the following command:

```shell

```
pip install -r requirements.txt
```

Recommended Python version: 3.10.4 (tested with Python 3.10.4).


# Data

Check out the notebook [Introduction to data handling](/notebooks/1_introduction_data_handling.ipynb) in `/notebooks/` directory for detailed information on downloading, preprocessing, and splitting the data.

# Quick Start

If you wish to preprocess the data, you can utilize the preprocess_data.py script. While this step is not mandatory, it's important to note that certain transforms (e.g., BandPassFilter) used during training could significantly slow down the training process. To preprocess the data, execute the following command:
```
python preprocess_data.py

```

Take a look at the configs directory for YAML configurations:

* YAML files within the training directory are employed for model training.
* YAML files within the predicting directory are employed for model testing and evaluation.

Two notebooks are provided for generating training and testing YAML files based on the data splits performed using the create_data_split_csvs.py script: [Yaml files of database-wise split for training and testing](/notebooks/2_physionet_DBwise_yaml_files.ipynb) and [Yaml files of stratified split for training and testing](/notebooks/2_physionet_stratified_yaml_files.ipynb). Be sure to perform the data splitting prior to using these notebooks.

1) To split the data for model usage in training and testing, use the following command:

```
python create_data_split_csvs.py

```

The create_data_split_csvs.py script employs either stratified split or database-wise split. For stratified splitting, it utilizes the MultilabelStratifiedShuffleSplit implementation from the iterative-stratification package. This results in CSV files containing the training and validation sets. These CSV files are subsequently used in the model training phase and include columns such as path (ECG recording path in .mat format), age, gender, and various diagnoses represented as SNOMED CT codes. Test data CSV files are also generated. The database-wise split relies on the structure of the loaded data directory.

The main structure of the CSV files is outlined as follows:


| path  | age  | gender  | 10370003  | 111975006 | 164890007 | *other diagnoses...* |
| ------------- |-------------|-------------| ------------- |-------------|-------------|-------------|
| ./Data/A0002.mat | 49.0 | Female | 0 | 0 | 1 | ... |
| ./Data/A0003.mat | 81.0 | Female | 0 | 1 | 1 | ... |
| ./Data/A0004.mat | 45.0 |  Male  | 1 | 0 | 0 | ... |
| ... | ... |  ...  | ... | ... | ... | ... |

Note: Ensure you review the necessary attributes before running the script. Further instructions are provided in the notebook [Introduction to data handling (/notebooks/1_introduction_data_handling.ipynb). 

2) To train a model, use either a YAML file or a directory as an argument and choose one of the following commands:

```
python train_model.py train_smoke.yaml
python train_model.py train_stratified_smoke

```

The train_data.yaml file contains required training arguments in YAML format, while train_multiple_smoke is a directory with multiple YAML files. When multiple YAML files are used simultaneously, each is loaded and executed separately. More detailed training information is available in the notebook [Introduction to training models](/notebooks/3_introduction_training.ipynb).

3) To test and evaluate a trained model, use one of the following commands:

```
python test_model.py predict_smoke.yaml
python test_model.py predict_stratified_smoke

```

The predict_smoke.yaml file contains necessary prediction phase arguments in YAML format, and predict_multiple_smoke is a directory with multiple YAML files. Similar to training, when multiple YAML files are used concurrently, each is loaded and executed individually. Detailed information about prediction and evaluation is provided in the notebook [Introduction to testing and evaluating models](/notebooks/4_introduction_testing_evaluation.ipynb).


# Repository in details

```
- configs
    - data_splitting
    - predicting
    - training
- data
    - smoke_data
    - split_csvs
- notebooks
- src
    - dataloader
        - __init__.py
        - dataset.py
        - dataset_utils.py
        - transforms.py
    - modeling
        - models
            - seresnet18.py
        - __init__.py
        - metrics.py
        - predict_utils.py
        - train_utils.py
- .gitignore
- label_mapping.py
- LICENSE
- __init__.py
- create_data_csvs.py
- preprocess_data.py
- README.md
- requirements.txt
- run_model.py
- train_model.py
- utils.py




```

Feel free to explore the repository for detailed implementation and utilization of deep learning techniques for ECG classification.
