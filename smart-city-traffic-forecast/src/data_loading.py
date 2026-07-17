"""
data_loading.py
===============
Module for loading and initial inspection of the traffic dataset.
"""

import os
import pandas as pd


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the traffic dataset from CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'data/traffic.csv'
        relative to the project root.

    Returns
    -------
    pd.DataFrame
        Raw traffic dataset.
    """
    if filepath is None:
        # Resolve path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(project_root, "data", "traffic.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform initial inspection of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Dictionary containing inspection results.
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "head": df.head(10),
        "describe": df.describe(),
    }

    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"\nShape: {info['shape']}")
    print(f"\nColumns: {info['columns']}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nNull Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {info['duplicates']}")
    print(f"\nFirst 10 Rows:\n{info['head']}")
    print(f"\nStatistical Summary:\n{info['describe']}")
    print(f"\nUnique Junctions: {df['Junction'].unique()}")
    print(f"\nRecords per Junction:\n{df['Junction'].value_counts().sort_index()}")
    print("=" * 60)

    return info


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
