# ECG Classification using Deep Learning

This repository contains a refactored version of the original implementation by Between_a_ROC_and_a_heart_place, which was designed for the PhysioNet/Computing in Cardiology Challenge 2020. The related paper was titled "Adaptive Lead Weighted ResNet Trained With Different Duration Signals for Classifying 12-lead ECGs". The original repository can be found [here](original_repository_link).

## Usage

To get started, follow these steps:

1. Install the required Python packages from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt

Note: Recommended Python version is 3.10.4 (tested with Python 3.10.4).


Data handling: Check out the notebook Introduction to data handling in the notebooks directory for more information on downloading, preprocessing, and splitting data.

Preprocessing: If you want to preprocess data, you can use the preprocess_data.py script. This step is optional but may impact training speed. To preprocess the data, use:
python preprocess_data.py



