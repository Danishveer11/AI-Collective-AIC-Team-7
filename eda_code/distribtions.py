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

# Target Distribution
for name, df in datasets.items():
    plt.figure(figsize=(7,5))
    sns.countplot(x='Target', data=df)
    plt.title(f'Target Distribution - {name}')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.savefig(f'target_distribution_{name}.png')

# Sentiment Distribution
for name, df in datasets.items():
    plt.figure(figsize=(7,5))
    sns.countplot(x='Sentiment', data=df)
    plt.title(f'Sentiment Distribution - {name}')
    plt.savefig(f'sentiment_distribution_{name}.png')

numerical_columns = [
    'Adj Close',
    'Close',
    'High',
    'Low',
    'Open',
    'Volume',
    'Score',
    'Comments',
    'Sentiment_Score'
]

# Boxplots
for name, df in datasets.items():
    for col in numerical_columns:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col} - {name}')
        plt.savefig(f'{col}_distribution_{name}.png')


edited_numerical_columns = [
    'Close',
    'High',
    'Low',
    'Open',
    'Volume'
]

# Correlation Matrix Heatmap
for name, df in datasets.items():
    corr = df[edited_numerical_columns + ['Target']].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, fmt='.4f', cmap='coolwarm')
    plt.title(f'Correlation Heatmap - {name}')
    plt.savefig(f'corr_heatmap_{name}.png')