import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# setup
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# read data
apple_df = pd.read_csv("datasets/Apple_data.csv")
amazon_df = pd.read_csv("datasets/Amazon_data.csv")
tesla_df = pd.read_csv("datasets/Tesla_data.csv")

# dictionary with key
datasets = {
    'Apple': apple_df,
    'Amazon': amazon_df,
    'Tesla' : tesla_df
}

# overall summary
for name, df in datasets.items():
    print(f'{name} Dataset')
    print('Shape:', df.shape)
    print('\nColumns:')
    print(df.columns)
    print('\nData Types:')
    print(df.dtypes)
    print('\nFirst 5 Rows:')
    print(df.head())

# check for duplicates
for name, df in datasets.items():
    duplicates = df.duplicated().sum()
    print(f'{name} Duplicated Rows: {duplicates}')

# mathematical statistics of datasets
for name, df in datasets.items():
    print(f'Summary Statistics: {name}')
    print(df.describe())

