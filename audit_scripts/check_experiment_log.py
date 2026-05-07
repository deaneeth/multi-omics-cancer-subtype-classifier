"""
Audit Script: Experiment log integrity check.
Run from project root: python audit_scripts/check_experiment_log.py
"""
import pandas as pd

df = pd.read_csv("experiment_log.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nNull counts:")
print(df.isnull().sum())
print(f"\nDuplicate rows (excluding timestamp): {df.drop(columns=['timestamp']).duplicated().sum()}")
print(f"\nNull artifact_path rows: {df['artifact_path'].isna().sum()}")
print(f"\nNull artifact_path by experiment:")
print(df[df['artifact_path'].isna()]['experiment_name'].value_counts())
print(f"\nRuns per experiment (total, including reruns):")
print(df.groupby('experiment_name').size().sort_values(ascending=False))
