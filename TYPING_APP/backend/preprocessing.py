# backend/preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import codecs
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def parse_ascii_code(x):
    try:
        if pd.isnull(x):
            return None
        if isinstance(x, str) and len(x) > 1 and x.startswith('\\'):
            # Convertir chaîne comme '\\b' en vrai caractère \b
            real_char = codecs.decode(x, 'unicode_escape')
            if len(real_char) == 1:
                return ord(real_char)
            else:
                return None
        elif len(x) == 1:
            return ord(x)
        return None
    except:
        return None

def pre_traitement(filepath: str):
    df = pd.read_csv(filepath)
    df=df.drop(columns=['answer','emotionIndex','index'])
    df1 = df.dropna()
    df1['ascii_code'] = df1['keyCode'].apply(parse_ascii_code)
    return df1

def prepare_data_emotion_user_sequence(df, features_cols):
    features = df[features_cols].values.astype(np.float32)

    if len(features) < 65:
        pad_len = 65 - len(features)
        pad = np.zeros((pad_len, features.shape[1]), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[:65]

    return torch.tensor(features, dtype=torch.float32)


def standardize_data(X_data):
    mean = np.array([100.53652, 439.7143 , 339.35602, 337.61746, 237.0325 , 777.6308 ], dtype=np.float32)
     #[100.62298, 439.72952, 339.26392, 337.47845, 236.88953, 777.6949 ]
     #[ 42.5668 , 366.17056, 367.8127 , 362.9565 , 366.85446, 552.81024]
    std=np.array([ 42.839725, 365.99536 , 368.1581  , 363.02792 , 366.84238 ,552.35223],dtype=np.float32)
    means = torch.tensor(mean, dtype=torch.float32)
    stds = torch.tensor(std, dtype=torch.float32)
    X_list_standardized = []
    for seq in X_data:
        # shape: [seq_len, nb_features]
        standardized = (seq - means) / stds
        X_list_standardized.append(standardized)

    return X_list_standardized


class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]