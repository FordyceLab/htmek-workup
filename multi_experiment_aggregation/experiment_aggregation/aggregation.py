# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import plotly.express as px
from sklearn.metrics import r2_score
import re

from scipy import optimize
import ast

# =============================================================================
# Data formatting
# =============================================================================

# add df tags
def add_df_tags(df):

    # create name column from MutantID; remove sample plate well number
    if 'MutantID' in df.columns:
        df.loc[df['MutantID'].str.contains("^\d{1,2}[A-Za-z]{1,2}\d{1,2}", regex=True), 'name'] = df.MutantID.str.split('_').str[1:].str.join("_")
        df.loc[df['MutantID'].str.contains("BLANK", regex=True), 'name'] = "BLANK"

    if 'n_mutations' in df.columns:
        df['n_mutations'] = pd.to_numeric(df['n_mutations'])

    # replace hyphens in sequence data input
    df['name'] = df['name'].str.replace('-', ' ')
    df['name'] = df['name'].str.replace('.', ' ')

    # add mutation column if single mutant
    df['mutation'] = df.name.str.extract(r'([A-Z]\d{1,2}[A-Z])', expand=False).fillna('')

    # add species labels
    df.loc[df['name'].str.contains("sbenthica"), 'species'] = 'sbenthica'
    df.loc[df['name'].str.contains("hsapiens2"), 'species'] = 'hsapiens'
    df.loc[df['name'].str.contains("human"), 'species'] = 'hsapiens'
    df.loc[df['name'].str.contains("phorikoshii"), 'species'] = 'phorikoshii'
    df.loc[df['name'].str.contains("hypf"), 'species'] = 'hypf'
    df.loc[df['name'].str.contains("\w\d{5,10}\w", regex=True), 'species'] = 'uncharacterized' # matchs the naming convention that Clara used to describe the uncharacterized bacterial ACYPs
    df.loc[df['name'].str.contains("cons"), 'species'] = 'consensus'
    df.loc[df['name'].str.contains("acylphosphatase"), 'species'] = df.name.str.replace('_wt', '').str.split('acylphosphatase').str[-1].str.split('_').str[1].str.replace('-', ' ')

    # add origin labels
    df.loc[df['name'].str.contains("[A-Z]\d{1,2}[A-Z]", regex=True), 'origin'] = 'mutant'
    df.loc[df['name'].str.contains("wt", regex=False), 'origin'] = 'WT'
    df.loc[df['name'].str.contains("pross"), 'origin'] = 'pross'
    df.loc[df['name'].str.contains("lambda"), 'origin'] = 'progen'
    df.loc[df['name'].str.contains("cons"), 'origin'] = 'consensus'
    df.loc[df['name'].str.contains("acylphosphatase"), 'origin'] = 'WT'

    # extract replicate numbers from pross and progen chambers
    df.loc[df.name.str.contains('pross'), ['sampling_replicate']] = df.name.str.split('_').str[-1]
    df.loc[df.name.str.contains('lambda'), ['sampling_replicate']] = df.name.str.split('_').str[-1]

    # rename mutants
    df["name"] = df["name"].apply(lambda x: re.sub(r"human_ACYP2", "hsapiens2", x))

    # extract progen design features
    df.loc[df.name.str.contains('highlambda'), ['lambda']] = 'high'
    df.loc[df.name.str.contains('lowlambda'), ['lambda']] = 'low'
    df.loc[df.name.str.contains('highid'), ['seq_id']] = 'high'
    df.loc[df.name.str.contains('lowid'), ['seq_id']] = 'low'
    
    # reformat sequence column
    if 'sequence' in df.columns:
        df['protein_sequence'] = df['sequence']
        df = df.drop('sequence', axis=1)
        
        # add wt protein sequence to each row
        WT_df = df[(df['origin'] == 'WT') | (df['origin'] == 'consensus')]
        WT_df = WT_df.drop_duplicates(subset=['species'])

        # add wt sequence to each row
        df = pd.merge(df, WT_df[['species', 'protein_sequence']], on='species', suffixes=['', '_wt'])

    # fill na values with an empty character
    # df = df.fillna('')

    return df


