# Standard imports
import pandas as pd
import seaborn as sns
import numpy as np

# plotting
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import groupby
import matplotlib.patches as mpatches

# stats
from scipy.optimize import curve_fit
from scipy import optimize, interpolate
from scipy import stats
import scipy as scp

# symbolic math
import sympy as sy
from sympy.solvers import solve
from sympy import symbols

# image segmenting and manual culling
import io
from PIL import Image
from ipywidgets import widgets

# parallelization
from pandarallel import pandarallel

# exports, formatting
import os
import PyPDF2
from tqdm import tqdm
from time import sleep




# Format imported data
def format_data(standard_data, kinetic_data, substrate, egfp_data):
    """Format dataframes from the processor script

    Parameters
    ----------
    standard_data : pandas dataframe
        Standard data from the processor script
    kinetic_data : pandas dataframe
        Kinetic data from the processor script
    substrate : str
        Substrate name
    egfp_data : pandas dataframe
        EGFP data from the processor script

    Returns
    -------
    standard_data : pandas dataframe
        Standard data with indices and substrate information
    kinetic_data : pandas
        Kinetic data with indices, substrate information, and EGFP values
    """

    # include leading zeros
    def string_with_leading_zero(n):
        if n < 10:
            n = "0" + str(n)
        else:
            n = str(n)

        return n

    # set indices in each dataframe
    standard_data['Indices'] = standard_data.x.apply(string_with_leading_zero) + ',' + standard_data.y.apply(string_with_leading_zero)
    kinetic_data['Indices'] = kinetic_data.x.apply(string_with_leading_zero) + ',' + kinetic_data.y.apply(string_with_leading_zero)
    egfp_data['Indices'] = kinetic_data.x.apply(string_with_leading_zero) + ',' + kinetic_data.y.apply(string_with_leading_zero)

    # replace egfp data to kinetic dataframe (this accounts for egfp data exported from the processor script separately from the kinetic data)
    kinetic_data = pd.merge(egfp_data[['x', 'y', 'summed_button_BGsub']], kinetic_data, on=['x', 'y']) # add egfp values from separate csv
    kinetic_data['summed_button_BGsub_Button_Quant'] = kinetic_data['summed_button_BGsub'] # move values to correct column

    # add substrate information to standard dataframe
    standard_data['substrate'] = substrate
    kinetic_data['substrate_conc_uM'] = kinetic_data.series_index.apply(lambda x: int(x.split('_')[0][:-2]))
    kinetic_data['substrate'] = substrate

    return standard_data, kinetic_data


# Squeeze kinetics dataframe to serialize timepoints
def squeeze_kinetics(kinetic_data):
    """Squeeze kinetics dataframe to serialize timepoints

    Parameters
    ----------
    kinetic_data : pandas dataframe
        Kinetic data from the processor script
    
    Returns
    -------
    squeeze_kinetics : pandas dataframe
        Kinetic data with timepoints serialized
    """

    # squeeze kinetics dataframe to serialize timepoints
    columns = ['x', 'y', 'Indices', 'MutantID', 'substrate', 'substrate_conc_uM', 'summed_button_BGsub_Button_Quant']
    kinetic_median_intensities = kinetic_data.groupby(columns)['median_chamber'].apply(list).reset_index(name='kinetic_median_intensities')
    times = kinetic_data.groupby(columns)['time_s'].apply(list).reset_index(name='time_s')
    squeeze_kinetics = pd.merge(times, kinetic_median_intensities, on=['x', 'y', 'Indices', 'substrate', 'MutantID', 'substrate_conc_uM', 'summed_button_BGsub_Button_Quant'])

    # sort by timepoint
    squeeze_kinetics['kinetic_median_intensities'] = squeeze_kinetics.apply(lambda row: np.array(row['kinetic_median_intensities'])[np.argsort(row['time_s'])].tolist(), axis=1)
    squeeze_kinetics['time_s'] = squeeze_kinetics.apply(lambda row: np.array(row['time_s'])[np.argsort(row['time_s'])].tolist(), axis=1)

    # Print out the first two rows of the dataframe
    display(squeeze_kinetics.head(2))

    return squeeze_kinetics


# Squeeze standard dataframe to serialize standard concentrations
def squeeze_standard(standard_data):
    """Squeeze standard dataframe to serialize standard concentrations
    
    Parameters
    ----------
    standard_data : pandas dataframe
        Standard data from the processor script
    
    Returns
    -------
    squeeze_standards : pandas dataframe
        Standard data with standard concentrations serialized
    """

    # squeeze standard dataframe to serialize standard concentrations
    standard_median_intensities = standard_data.groupby(['x', 'y', 'Indices', 'substrate'])['median_chamber'].apply(list).reset_index(name='standard_median_intensities')
    standard_concentration_uM = standard_data.groupby(['x', 'y', 'Indices', 'substrate'])['concentration_uM'].apply(list).reset_index(name='standard_concentration_uM')
    squeeze_standards = pd.merge(standard_concentration_uM, standard_median_intensities, on=['x', 'y', 'Indices', 'substrate'])

    # sort by standard concentration
    squeeze_standards['standard_median_intensities'] = squeeze_standards.apply(lambda row: np.array(row['standard_median_intensities'])[np.argsort(row['standard_concentration_uM'])].tolist(), axis=1)
    squeeze_standards['standard_concentration_uM'] = squeeze_standards.apply(lambda row: np.array(row['standard_concentration_uM'])[np.argsort(row['standard_concentration_uM'])].tolist(), axis=1)

    # define linear range of standard curve
    n = 5

    for i in np.random.randint(0, len(squeeze_standards), 120):
        # plot the relationship
        plt.scatter(x=squeeze_standards.standard_concentration_uM[i][:-1], y=squeeze_standards.standard_median_intensities[i][:-1], c='grey', alpha=0.2)
        plt.scatter(x=squeeze_standards.standard_concentration_uM[i][1:n], y=squeeze_standards.standard_median_intensities[i][1:n], c='red', alpha=0.2)
        plt.title('Standard Curve \n Linear regime in red')
        plt.xlabel('Standard Concentration (uM)')
        plt.ylabel('Median Standard Chamber Intensity (RFU)')

    # Manually add handles for scatter plots
    handles = [mpatches.Patch(color='grey', label='Standard Curve'), mpatches.Patch(color='red', label='Linear Regime')]
               
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, handles=handles, loc='lower right')

    display(squeeze_standards.head(4))

    return squeeze_standards


# Perform standard curve fitting
def standard_curve_fit(squeeze_standards):
    """Perform standard curve fitting

    Parameters
    ----------
    squeeze_standards : pandas dataframe
        Standard data with standard concentrations serialized
    
    Returns
    -------
    squeeze_standards : pandas dataframe
        Standard data with standard concentrations serialized and curve fit parameters added
    """


    ## Perform standard curve fitting
    # define isotherm function amd vectorize
    def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
    v_isotherm = np.vectorize(isotherm)

    # define curve fit function
    # excluded high concentration
    def standard_curve_fit(df):
        popt, pcov = optimize.curve_fit(isotherm, df.standard_concentration_uM[:-1], df.standard_median_intensities[:-1], bounds=([500, 0, 10, 0], [np.inf, np.inf, np.inf, np.inf])) # A, KD, PS, I_0uMP_i
        return popt

    # fit all standard curves
    print('Performing curve fits...')
    squeeze_standards['standard_popt'] = squeeze_standards.parallel_apply(standard_curve_fit, axis=1)


    ## Plot standard curves to check fit
    # select example chambers
    num_examples = 4
    examples = np.random.randint(0, len(squeeze_standards), num_examples)

    # plot subplots
    fig, axs = plt.subplots(1, num_examples, figsize=(12, 3), sharey=True)

    for k,v in dict(zip(range(num_examples), examples)).items():
        xdata, ydata = squeeze_standards.standard_concentration_uM[v], squeeze_standards.standard_median_intensities[v]
        chamber_idx = squeeze_standards.Indices[v]
        chamber_popt = squeeze_standards.standard_popt[v]

        # finally, define the interpolation
        interp_xdata = np.linspace(-np.min(xdata), np.max(xdata), num=100, endpoint=False)

        axs[k].plot(xdata[:-1], v_isotherm(xdata[:-1], *chamber_popt), 'r-', label='Isotherm Fit')
        axs[k].plot(interp_xdata[:-1], v_isotherm(interp_xdata[:-1], *chamber_popt), 'g--', label='interpolation')
        axs[k].plot(xdata[:-1], ydata[:-1], 'bo', label='data')
        axs[k].set_ylabel('Intensity')
        axs[k].set_xlabel('[Pi] (uM)')
        axs[k].legend(loc='lower right', shadow=True)
        axs[k].set_title('Chamber: ' + chamber_idx)

    fig.suptitle(f'{num_examples} Standard Curves', y=1.1)


    return squeeze_standards


# Merge standard and kinetics dataframes and calculate product concentrations by interpolation of standard curve
def merge_and_get_product_concs(squeeze_kinetics, squeeze_standards):
    """Merge standard and kinetics dataframes and calculate product concentrations by interpolation of standard curve
    
    Parameters
    ----------
    squeeze_kinetics : pandas dataframe
        Kinetics data with substrate concentrations serialized
    squeeze_standards : pandas dataframe
        Standard data with standard concentrations serialized and curve fit parameters added
    
    Returns
    -------
    sq_merged : pandas dataframe
        Kinetics data with substrate concentrations serialized and product concentrations calculated
    """

    # merge standard and kinetics dataframes
    sq_merged = pd.merge(squeeze_kinetics, squeeze_standards, on=['x', 'y', 'Indices', 'substrate'])

    # define interpolation function
    def interpolate(df):
        # define xdata range for interpolation
        xdata = df.standard_concentration_uM
        
        # define isotherm function amd vectorize
        def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
        v_isotherm = np.vectorize(isotherm)
            
        # create interpolated look-up dictionary
        # num corresponds to the number of intervals in the interpolation. 
        # Increasing this number leads to a larger lookup table and better
        # extrapolations of product concs
        int_xdata = np.linspace(-np.min(xdata), np.max(xdata), num=2000, endpoint=False)
        d = dict(zip(v_isotherm(int_xdata, *df.standard_popt), int_xdata))

        # estimate product concentration from lookup dictionary
        product_concs = [ d.get(i, d[min(d.keys(), key=lambda k: abs(k - i))]) for i in df.kinetic_median_intensities]

        return product_concs
    
    # calculate product concentrations
    print('Calculating product concentrations...')
    print('(If interpolating, this will take longer than the curve fitting step)')
    sq_merged['kinetic_product_concentration_uM'] = sq_merged.parallel_apply(interpolate, axis=1)
    sq_merged.head(1)

    return sq_merged


