import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam

from model import build_autoencoder


def load_credential_data(csv_path: str):
    """
    Load the Credit Card Fraud dataset from a CSV file.

    Expected to have a 'Class' column where:
      - 0 = normal transaction
      - 1 = fraud (anomaly)
    """
    df = pd.read_csv(csv_path)
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column for labels.")
    