import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import math
from .dataset_utils import load_data, encode_metadata
from .transforms import Compose, RandomClip, Normalize, ValClip, MultiplySine, NotchFilter, Retype, Flipy, Flipx, RandomStretch, ResampleSine, ResampleLinear, ResampleTriangle, EqualSegmentResampler
from .transforms import *
from scipy.signal import find_peaks
def get_transforms(dataset_type):
    ''' Get transforms for ECG data based on the dataset type (train, validation, test)
    '''
    seq_length = 4096
    normalizetype = '0-1'
    fs = 250
    #notch_freq = 30

    data_transforms = {
        
        'train': Compose([
            RandomClip(w=seq_length),
            Normalize(normalizetype),
            Retype() 
        ], p = 1.0),
        

        'val': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p =  1.0),

        'augment': Compose([
            RandomClip(w=seq_length),
            Normalize(normalizetype),
            #AddNoise(sigma = 0.3, p = 0.5),
            #Roll(p = 1.0),
            #Flipx(p = 0.5),
            #RandomStretch(scale = 0.5, p = 1.0),
            #ResampleSine(fs = 250, freq_lo = 1.0, freq_hi = 2.3, scale_lo = 0.0, scale_hi = 2.0, p=1),
            #ResampleLinear(scale = 1.2, p = 1.0),
            #NotchFilter(fs, Q = 2.5, p = 1.0),
            #MultiplyLinear(multiplier = 2.5, p = 1.0),
            #ResampleTriangle(scale = 4, p = 0.1),
            #EqualSegmentResampler(A2 = 1.3, A3 = 0.8, p = 1.0),
            #ResampleTriangle(scale = 1.35, p = 1),
            MultiplyTriangle(scale = 4, p = 1.0),
            #MultiplySine(fs = 250, f = 1.2, a = 0.3, p = 0.1),
            #ResampleLinearAlign1stPeak(scale= 1.2, p =1.0),
            Retype()
        ], p = 1.0),
 

        'test': Compose([
            ValClip(w=seq_length),
            Normalize(normalizetype),
            Retype()
        ], p = 1.0)
    }
    return data_transforms.get(dataset_type, data_transforms)


class ECGDatasetAug(Dataset):
    ''' Class implementation of Dataset of ECG recordings that are related to training and augmented data
    
    :param path: The directory of the data used
    :type path: str
    :param preprocess: Preprocess transforms for ECG recording
    :type preprocess: datasets.transforms.Compose
    :param transform: The other transforms used for ECG recording
    :type transform: datasets.transforms.Compose
    '''

    def __init__(self, path, transforms, augment):
        df = pd.read_csv(path)
        self.data = df['path'].tolist()
        labels = df.iloc[:, 4:].values
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        
        self.age = df['age'].tolist()
        self.gender = df['gender'].tolist()
        self.fs = df['fs'].tolist()

        self.transforms = transforms
        self.augment = augment
        self.channels = 12

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file_name = self.data[item]
        fs = self.fs[item]
        tra = load_data(file_name)
        
        aug = self.transforms(tra)
        ecg = self.augment(aug)
        #print('First training data dimention:',tra.shape)
        #ecg = np.concatenate((tra, aug), axis=1)
        #print('ECG input of network dimention:',ecg.shape)
    
        label = self.multi_labels[item]
        
        age = self.age[item]
        gender = self.gender[item]
        age_gender = encode_metadata(age, gender)
        return ecg, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()



class ECGDatasetVal(Dataset):
    ''' Class implementation of Dataset of ECG recordings that are related to validation data
    
    :param path: The directory of the data used
    :type path: str
    :param preprocess: Preprocess transforms for ECG recording
    :type preprocess: datasets.transforms.Compose
    :param transform: The other transforms used for ECG recording
    :type transform: datasets.transforms.Compose
    '''

    def __init__(self, path, transforms):
        df = pd.read_csv(path)
        self.data = df['path'].tolist()
        labels = df.iloc[:, 4:].values
        self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
        
        self.age = df['age'].tolist()
        self.gender = df['gender'].tolist()
        self.fs = df['fs'].tolist()

        self.transforms = transforms
        self.channels = 12
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file_name = self.data[item]
        fs = self.fs[item]
        tra = load_data(file_name)
        
        ecg = self.transforms(tra)
        
        label = self.multi_labels[item]
        
        age = self.age[item]
        gender = self.gender[item]
        age_gender = encode_metadata(age, gender)
        return ecg, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()
      