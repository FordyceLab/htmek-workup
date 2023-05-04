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



# =============================================================================
# Data aggregation plotting
# =============================================================================

def plot_comparisons(df_merged_pivot, error_type, parameter, labels, color=None, scale='linear'):
    """
    This function takes a dataframe with median and standard deviation values for each
    experiment and plots a scatter plot of the median values for each experiment, where
    each point represents the same mutant in each experiment.

    Args:
        df_merged_pivot (dataframe): dataframe with median and standard deviation values
        error_type (str): type of error to plot
        parameter (str): name of parameter to plot
        labels (list): list of labels for each experiment
        scale (str): scale of the x and y axes, either 'linear' or 'log'

    Returns:
        fig: plotly figure object
    """

    fig = px.scatter(x=df_merged_pivot[('median', labels[0])], y=df_merged_pivot[('median', labels[1])],
                        error_x=df_merged_pivot[(error_type, labels[0])], error_y=df_merged_pivot[(error_type, labels[1])],
                        hover_name=df_merged_pivot['MutantID'],
                        title=f'{parameter} Replicates', width=800, height=500, color=df_merged_pivot[color])
    
    # name colorbar
    fig.update_layout(coloraxis_colorbar=dict(title=color))


    # plot identity line from min to max of x and y axes
    max_x = max(df_merged_pivot[('median', labels[0])].max(), df_merged_pivot[('median', labels[1])].max())
    min_x = min(df_merged_pivot[('median', labels[0])].min(), df_merged_pivot[('median', labels[1])].min())
    fig.add_shape(type='line', x0=min_x, y0=min_x, x1=max_x, y1=max_x, line=dict(color='red', width=3))
    
    # if scale is log, transform data
    if scale == 'log':
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')

    # label axes
    fig.update_xaxes(title_text=labels[0] + '\n' + parameter)
    fig.update_yaxes(title_text=labels[1] + '\n' + parameter)
    
    # add to x axis dict
    fig.update_xaxes(mirror=True, linewidth=2, linecolor='black', showgrid=False, gridcolor='White')
    fig.update_yaxes(mirror=True, linewidth=2, linecolor='black', showgrid=False, gridcolor='White')
    # fig.update_traces(marker=dict(color='black', size=10))
    # fig.update_traces(error_x_color='lightgray', error_y_color='lightgray')
    # fig.update_layout(plot_bgcolor='white')
    fig.update_layout(font=dict(size=20))
    fig.update_traces(text=df_merged_pivot['MutantID'])

    # add legend
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font=dict(size=15)
    ))

    return fig

def compare_replicates(dfs, parameter, parameter_label, color=None, scale='linear'):
    """
    This function takes a list of dataframes and a parameter name, and plots a scatter 
    plot of the parameter values for each dataframe, where each point represents the same
    mutant in each experiment.

    Args:
        dfs (dict): dictionary of dataframes
        parameter (str): name of parameter to plot
        labels (list): list of labels for each experiment
        scale (str): scale of the x and y axes, either 'linear' or 'log'
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    # define error type
    error_type = 'sem'

    # get labels from df names
    labels = list(dfs.keys())

    # first merge the two dataframes
    df_merged = pd.concat(list(dfs.values()), keys=labels, names=['Date'])
    df_merged = df_merged.reset_index()

    # get min Rsq_kobs for each mutant
    if color is not None:
        df_merged_color = df_merged.groupby(['MutantID'])[color].mean().reset_index()

    # groupby mutant and experiment and calculate the median and standard deviation
    df_merged = df_merged.groupby(['MutantID', 'Date'])[parameter].agg(['median', error_type])
    df_merged = df_merged.reset_index()

    # pivot tabe to get the EnzymeConc data for each experiment in a separate column
    df_merged_pivot = df_merged.pivot(index='MutantID', columns='Date', values=['median', error_type])
    if color is not None:
        df_merged_pivot = pd.merge(df_merged_pivot, df_merged_color, on='MutantID')
    df_merged_pivot = df_merged_pivot.reset_index()
    df_merged_pivot = df_merged_pivot.dropna()

    # plot the data
    fig = plot_comparisons(df_merged_pivot, error_type, parameter, labels, color=color, scale=scale)

    # change axis labels
    fig.update_xaxes(title_text=labels[0] + '\n' + parameter_label)
    fig.update_yaxes(title_text=labels[1] + '\n' + parameter_label)

    return fig


# define exponential function amd vectorize
def exponential(t, A, k, y0): 
    return A*(1-np.exp(-k*t))+y0
v_exponential = np.vectorize(exponential)


# fit exponential to data
def exponential_fit(xdata, ydata):
    try:
        popt, pcov = optimize.curve_fit(exponential, xdata, ydata, bounds=([0, 0, -20], [np.inf, 0.05, np.inf])) # A, k, y_0
    except:
        popt = [np.nan, np.nan, np.nan]
    return popt


# define mappable function
def map_exponential_fit(df):
    xdata = ast.literal_eval(df.time_s)
    try:
        ydata = ast.literal_eval(df.kinetic_product_concentration_uM)
    except:
        ydata = np.nan

    try:
        popt = exponential_fit(xdata, ydata)
    except:
        popt = [np.nan, np.nan, np.nan]
        
    return popt


# map exponential fit to dataframe
def get_exponentials(df, conc):
    """
    This function takes a dataframe and a substrate concentration, and returns a dataframe
    with the exponential fit parameters for each mutant at that substrate concentration.

    Args:
        df (dataframe): dataframe with kinetic data
        conc (int): substrate concentration

    Returns:
        df (dataframe): dataframe with exponential fit parameters
    """

    # filter dataframe
    df = df[df['substrate_conc_uM'] == conc]

    # map exponential fit to dataframe
    opts = df.parallel_apply(map_exponential_fit, axis=1)

    df['A'] = opts.apply(lambda x: x[0])
    df['k_obs'] = opts.apply(lambda x: x[1])
    df['y0'] = opts.apply(lambda x: x[2])

    # get R^2 values for each mutant
    df = df.dropna(subset=['k_obs'])
    df['Rsq_kobs'] = df.apply(lambda x: r2_score(ast.literal_eval(x.kinetic_product_concentration_uM), 
                                                 v_exponential(ast.literal_eval(x.time_s), x.A, x.k_obs, x.y0)), axis=1)
    
    return df


