# Author: Micah Olivas
# Date: 2023-3-31
# Description: This script contains functions for processing HT-MEK data 
# from the image processing script. It is designed to be imported into 
# an ipynb session for interactive plotting and summary pdf generation.



# Standard imports
import pandas as pd
import seaborn as sns
import numpy as np
import re
import bisect
import copy
import ast

# plotting
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import groupby
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from adjustText import adjust_text

# stats
from scipy.optimize import curve_fit
from scipy import optimize, interpolate
from scipy import stats
import scipy as scp
from decimal import Decimal # scientific notation

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
from os import listdir
from os.path import isfile, join




# Format imported data
def format_data(standard_data, kinetic_data, egfp_data, substrate=None):
    """Format dataframes from the processor script. This includes adding indices, substrate information, and EGFP values to the kinetic dataframe.

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

        # convert to string
        n = str(n)

        # add leading zero if necessary
        if len(n) == 1:
            n = '0' + n

        return n

    # set indices in each dataframe
    if standard_data is not None:
        standard_data['Indices'] = standard_data.x.apply(string_with_leading_zero) + ',' + standard_data.y.apply(string_with_leading_zero)
    else:
        standard_data = pd.DataFrame()
    
    if kinetic_data is not None:
        kinetic_data['Indices'] = kinetic_data.x.apply(string_with_leading_zero) + ',' + kinetic_data.y.apply(string_with_leading_zero)
    else:
        kinetic_data = pd.DataFrame()

    if egfp_data is not None:
        egfp_data['Indices'] = kinetic_data.x.apply(string_with_leading_zero) + ',' + kinetic_data.y.apply(string_with_leading_zero)
        
        # replace egfp data to kinetic dataframe (this accounts for egfp data exported from the processor script separately from the kinetic data)
        kinetic_data = pd.merge(egfp_data[['x', 'y', 'summed_button_BGsub']], kinetic_data, on=['x', 'y']) # add egfp values from separate csv
        kinetic_data['summed_button_BGsub_Button_Quant'] = kinetic_data['summed_button_BGsub'] # move values to correct column
    else:
        egfp_data = pd.DataFrame()


    # add substrate information to standard dataframe 
    # standard_data['substrate'] = substrate
    # kinetic_data['substrate_conc_uM'] = kinetic_data.series_index.apply(lambda x: int(x.split('_')[0][:-2]))
    # kinetic_data['substrate'] = substrate

    return standard_data, kinetic_data


# Squeeze kinetics dataframe to serialize timepoints
def squeeze_kinetics(kinetic_data, additional_columns=None, additional_squeeze_columns=None):
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

    # add additional columns if additional_columns is not None
    if additional_columns is not None:
        columns = columns + additional_columns
        
    kinetic_median_intensities = kinetic_data.groupby(columns)['median_chamber'].apply(list).reset_index(name='kinetic_median_intensities')
    times = kinetic_data.groupby(columns)['time_s'].apply(list).reset_index(name='time_s')
    squeeze_kinetics = pd.merge(times, kinetic_median_intensities, on=columns)
    
    if additional_squeeze_columns is not None:
        for key, value in additional_squeeze_columns.items():

            additional_squeeze_data = kinetic_data.groupby(columns)[key].apply(list).reset_index(name=value)
            squeeze_kinetics = pd.merge(additional_squeeze_data, squeeze_kinetics, on=columns)


    # sort by timepoint
    squeeze_kinetics['kinetic_median_intensities'] = squeeze_kinetics.apply(lambda row: np.array(row['kinetic_median_intensities'])[np.argsort(row['time_s'])].tolist(), axis=1)
    squeeze_kinetics['time_s'] = squeeze_kinetics.apply(lambda row: np.array(row['time_s'])[np.argsort(row['time_s'])].tolist(), axis=1)
    if additional_squeeze_columns is not None:
        for key, value in additional_squeeze_columns.items():
            squeeze_kinetics[value] = squeeze_kinetics.apply(lambda row: np.array(row[value])[np.argsort(row['time_s'])].tolist(), axis=1)

    return squeeze_kinetics


# Squeeze standard dataframe to serialize standard concentrations
def squeeze_standard(standard_data, standard_type, experiment_day, linear_range=None, pbp_conc=None, remove_concs=None, manual_concs=None):
    """Squeeze standard dataframe to serialize standard concentrations
    
    Parameters
    ----------
    standard_data : pandas dataframe
        Standard data from the processor script
    standard_type : str
        Type of standard (linear or PBP)
    standard_conc : float or int
        Standard concentration
    linear_range : list
        List of linear range concentrations
    experiment_day : str
        Experiment day
    remove_concs : list
        List of concentrations to remove
    
    Returns
    -------
    squeeze_standards : pandas dataframe
        Standard data with standard concentrations serialized
    """

    # squeeze standard dataframe to serialize standard concentrations
    standard_median_intensities = standard_data.groupby(['x', 'y', 'Indices'])['median_chamber'].apply(list).reset_index(name='standard_median_intensities')
    standard_concentration_uM = standard_data.groupby(['x', 'y', 'Indices'])['concentration_uM'].apply(list).reset_index(name='standard_concentration_uM')
    squeeze_standards = pd.merge(standard_concentration_uM, standard_median_intensities, on=['x', 'y', 'Indices'])

    # sort by standard concentration
    squeeze_standards['standard_median_intensities'] = squeeze_standards.apply(lambda row: np.array(row['standard_median_intensities'])[np.argsort(row['standard_concentration_uM'])].tolist(), axis=1)
    squeeze_standards['standard_concentration_uM'] = squeeze_standards.apply(lambda row: np.array(row['standard_concentration_uM'])[np.argsort(row['standard_concentration_uM'])].tolist(), axis=1)

    # remove concentrations if necessary
    if remove_concs is not None:
        # get positions of concs in lists stored in standard_concentration_uM column of standard_data
        conc_list = squeeze_standards['standard_concentration_uM'].iloc[0]
        positions = [i for i, x in enumerate(conc_list) if x in remove_concs]

        # remove concentrations from standard_data columns standard_concentration_uM and standard_median_intensities
        squeeze_standards['standard_concentration_uM'] = squeeze_standards['standard_concentration_uM'].apply(lambda x: [i for j, i in enumerate(x) if j not in positions])
        squeeze_standards['standard_median_intensities'] = squeeze_standards['standard_median_intensities'].apply(lambda x: [i for j, i in enumerate(x) if j not in positions])

    # get list of standard concentrations
    standard_concentrations = squeeze_standards.standard_concentration_uM.iloc[0]

    # get list containing the median intensity for each standard concentration
    all_chambers_median_intensities = []
    for n, i in enumerate(standard_concentrations):
        curr_median_intensity = (np.median([x[n] for x in squeeze_standards.standard_median_intensities]))
        all_chambers_median_intensities.append(curr_median_intensity)


    # fit isotherm to the median data
    if standard_type.upper() == 'PBP':
        # define pbp isotherm function amd vectorize
        def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
        v_isotherm = np.vectorize(isotherm)

        # define curve fit function
        # excluded high concentration
        # # if curve fit fails, return nan
        try:
            popt, pcov = optimize.curve_fit(isotherm, standard_concentrations, all_chambers_median_intensities, bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])) # A, KD, PS, I_0uMP_i
        except:
            popt = np.nan
            
    elif standard_type.upper() == 'LINEAR':
        
        # define linear function amd vectorize
        def linear(x, m, b): return m*x + b
        v_linear = np.vectorize(linear)

        # define curve fit function
        # if curve fit fails, return nan
        try:
            popt, pcov = optimize.curve_fit(linear, standard_concentrations, all_chambers_median_intensities, bounds=([-np.inf, 0], [np.inf, np.inf])) # m, b
        except:
            popt = np.nan


    # plot random subset of standard data
    rand_n = 500
    selected = np.random.randint(0, len(squeeze_standards), rand_n)
    for i in selected:
        # plot the relationship
        plt.scatter(x=squeeze_standards.standard_concentration_uM[i], y=squeeze_standards.standard_median_intensities[i], c='grey', alpha=0.1)

        # add vertical lines to indicate linear regime
        if linear_range is not None:
            plt.axvline(x=linear_range[0], c='red', linestyle='--', alpha=0.5)
            plt.axvline(x=linear_range[1], c='red', linestyle='--', alpha=0.5)
        # initialize plot
        ax = plt.gca()

        # add title and labels
        if standard_type.upper() == 'PBP':
            # plot pbp median standard curve
            t = np.linspace(0, max(standard_concentrations), 2000,)
            plt.plot(t, v_isotherm(t, *popt), 'b--', label='fit: A=%5.3f, KD=%5.3f, PS=%5.3f, I_0uMP_i=%5.3f' % tuple(popt))
            plt.title('%s \n Standard Curves \n %s %s uM' % (experiment_day, standard_type.upper(), pbp_conc))
            plt.xlabel('Phosphate Concentration (uM)')
        elif standard_type.upper() == 'LINEAR':
            # plot linear median standard curve
            t = np.linspace(0, max(standard_concentrations), 2000,)
            plt.plot(t, v_linear(t, *popt), 'b--', label='fit: m=%5.3f, b=%5.3f' % tuple(popt))
            plt.title('%s \n Standard Curves \n %s uM' % (experiment_day, standard_type.upper()))
            plt.xlabel('Standard Concentration (uM)')
        plt.ylabel('Median Chamber Intensity (RFU)')

    # add n to plot
    plt.text(0.9, 0.1, 'n (chambers)=%s' % rand_n, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, fontsize=14)

    # Manually add handles for scatter plots
    if standard_type.upper() == 'PBP':
        handles = [mpatches.Patch(color='grey', label='All-chamber Median Intensities'),
                    mpatches.Patch(color='b', label='Universal Median Fit: \n A=%5.3f, \n KD=%5.3f, \n PS=%5.3f, \n I_0uMP_i=%5.3f' % tuple(popt))
                ]
    if standard_type.upper() == 'LINEAR':
        handles = [mpatches.Patch(color='grey', label='All-chamber Median Intensities'),
                    mpatches.Patch(color='b', label='Universal Median Fit: \n m=%5.3f, \n b=%5.3f' % tuple(popt))
                ]
    
    if linear_range is not None:
        handles.append(mpatches.Patch(color='red', label='Linear Range'))
    
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    display(squeeze_standards.head(4))

    return squeeze_standards

# Calculate enzyme concentrations
def calculate_enzyme_concs(squeeze_kinetics, rfu_per_nM_E):
    # adjust the enzyme concentrations by the egfp slope
    squeeze_kinetics['EnzymeConc'] = squeeze_kinetics['summed_button_BGsub_Button_Quant']/rfu_per_nM_E # adjusted manually based on expected median concentration of ~40 nM from 230419 experiment

    # plot a histogram of enzyme conc for blank chambers
    blank_dat = squeeze_kinetics[squeeze_kinetics['MutantID'] == 'BLANK']
    plt.hist(blank_dat['EnzymeConc'], color='grey', alpha=0.3, linewidth=1.2, edgecolor='black', bins=50, label='Blank')

    # plot a histogram of enzyme conc for non-blank chambers
    non_blank_dat = squeeze_kinetics[squeeze_kinetics['MutantID'] != 'BLANK']
    plt.hist(non_blank_dat['EnzymeConc'], color='green', alpha=0.3, linewidth=1.2, edgecolor='green', bins=50, label='Printed')

    # adjust labels and plot
    plt.ylim(0, 400)
    plt.title('Enzyme Concentrations')
    plt.xlabel('Enzyme Concentration (nM)')
    plt.ylabel('Counts')
    plt.legend(title='Chamber')

    # adjust ratio of figure size to make it more square
    plt.gcf().set_size_inches(5, 5)

    plt.show()

    return squeeze_kinetics

# Perform standard curve fitting
def standard_curve_fit(squeeze_standards, standard_type, manual_concs=None):
    """Perform standard curve fitting
    Parameters:
    - squeeze_standards : (pandas dataframe) Standard data with standard concentrations serialized
    - standard_type : (str) Standard type, either 'pbp' or 'linear'
    Returns:
    - squeeze_standards : (pandas dataframe) Standard data with standard concentrations serialized and curve fit parameters added
    """

    # Perform standard curve fitting for pbp coupled reaction
    if standard_type.upper() == 'PBP':
        # define pbp isotherm function amd vectorize
        def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
        v_func = np.vectorize(isotherm)

        # define curve fit function
        # excluded high concentration
        def standard_curve_fit(df):

            # if curve fit fails, return nan
            try:
                popt, pcov = optimize.curve_fit(isotherm, df.standard_concentration_uM[:-1], df.standard_median_intensities[:-1], bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])) # A, KD, PS, I_0uMP_i
            except:
                popt = np.nan
                
            return popt
    
    # Perform standard curve fitting for fluorogenic substrate
    elif standard_type == 'linear':
        # define pbp isotherm function amd vectorize
        def linear(x, a, b): return a*x + b 
        v_func = np.vectorize(linear)

        # define curve fit function
        def standard_curve_fit(df):
                
                # if curve fit fails, return nan
                try:
                    popt, pcov = optimize.curve_fit(linear, df.standard_concentration_uM, df.standard_median_intensities, bounds=([0, 0], [np.inf, np.inf])) # a, b
                except:
                    popt = np.nan
                    
                return popt

    # fit all standard curves
    print('Performing curve fits...')
    print('Some curve fits may fail, this is expected and will be replaced with NaNs')
    squeeze_standards['standard_popt'] = squeeze_standards.parallel_apply(standard_curve_fit, axis=1)

    # interpolate points manually if necessary
    if manual_concs is not None:
        for i in manual_concs:
            for index, row in squeeze_standards.iterrows():
                # skip if concentration already in list
                if i in row['standard_concentration_uM']:
                    continue
               
                # if curve fit failed for a particular chamber, skip
                try:
                    # insert in sorted order
                    row['standard_concentration_uM'].insert(bisect.bisect_left(row['standard_concentration_uM'], i), i)
                    conc_index = row['standard_concentration_uM'].index(i)
                    # approx conc from curve fit
                    row['standard_median_intensities'].insert(conc_index, v_func(i, *row['standard_popt']))
                except:
                    pass

    ## Plot standard curves to check fit
    num_examples = 4
    examples = np.random.randint(0, len(squeeze_standards), num_examples)

    # plot subplots
    fig, axs = plt.subplots(1, num_examples, figsize=(12, 3), sharey=True)

    for k,v in dict(zip(range(num_examples), examples)).items():
        xdata, ydata = squeeze_standards.standard_concentration_uM[v], squeeze_standards.standard_median_intensities[v]
        chamber_idx = squeeze_standards.Indices[v]
        chamber_popt = squeeze_standards.standard_popt[v]

        # if curve fit failed for a particular chamber, skip
        try:
            # finally, define the interpolation
            interp_xdata = np.linspace(0, np.max(xdata), num=100, endpoint=False)

            axs[k].plot(xdata, v_func(xdata, *chamber_popt), 'r-', label='%s curve fit' % standard_type)
            axs[k].plot(interp_xdata, v_func(interp_xdata, *chamber_popt), 'g--', label='interpolation')
            axs[k].plot(xdata, ydata, 'bo', label='data')
            axs[k].set_xlabel('[Pi] (uM)')
            axs[k].set_title('Chamber: ' + chamber_idx)

            fig.suptitle(f'{num_examples} Standard Curves and {standard_type} Fits', y=1.1)

        except:
            pass
    
    # add y label to first subplot
    axs[0].set_ylabel('Intensity')

    # only plot legend for last subplot
    axs[num_examples-1].legend(loc='lower right', shadow=True)


    return squeeze_standards


# Merge standard and kinetics dataframes and calculate product concentrations by interpolation of standard curve
def merge_and_get_product_concs(squeeze_kinetics, squeeze_standards, standard_type):
    """Merge standard and kinetics dataframes and calculate product concentrations by interpolation of standard curve
    Parameters:
    - squeeze_kinetics : (pandas dataframe) Kinetics data with substrate concentrations serialized
    - squeeze_standards : (pandas dataframe) Standard data with standard concentrations serialized and curve fit parameters added
    Returns:
    - sq_merged : (pandas dataframe) Kinetics data with substrate concentrations serialized and product concentrations calculated
    """

    # merge standard and kinetics dataframes
    sq_merged = pd.merge(squeeze_kinetics, squeeze_standards, on=['x', 'y', 'Indices'])

    # define interpolation function
    def interpolate(df):
        """
        """
        # define xdata range for interpolation
        xdata = df.standard_concentration_uM
        
        # if using a pbp coupled assay, define the isotherm function
        if standard_type == 'pbp':
            # define isotherm function amd vectorize
            def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
            v_func = np.vectorize(isotherm)

        # if using a fluorogenic substrate, define the linear function
        elif standard_type == 'linear':
            # define linear fit equation and vectorize
            def linear(x, a, b): return a*x + b 
            v_func = np.vectorize(linear)
            
        ## Create interpolated look-up dictionary
        # The variable 'num' corresponds to the number of intervals in the interpolation. 
        # Increasing this number leads to a larger lookup table and better
        # extrapolations of product concs
        try:
            # create lookup dictionary from xdata range
            int_xdata = np.linspace(start=-np.mean(xdata), stop=np.max(xdata), num=3000, endpoint=False)
            d = dict(zip(v_func(int_xdata, *df.standard_popt), int_xdata))

            # estimate product concentration from lookup dictionary, using the lookup key closest to the median intensity
            product_concs = [ d.get(i, d[min(d.keys(), key=lambda k: abs(k - i))]) for i in df.kinetic_median_intensities]

            return product_concs

        except:
            print('Curve fit failed for chamber: ' + df.Indices)
            print(df.standard_popt)
            return np.nan
    
    # calculate product concentrations
    print('Calculating product concentrations...')
    print('(If interpolating, this will take longer than the curve fitting step)')
    sq_merged['kinetic_product_concentration_uM'] = sq_merged.parallel_apply(interpolate, axis=1)
    sq_merged.head(1)

    return sq_merged


# Create button objects slightly larger than the segmented chamber images
def create_expanded_button(description, button_style):
    return widgets.Button(description=description, button_style=button_style, layout=widgets.Layout(height='200px', width='200px'))

# Creat action to add chamber index to flagged set if button is clicked
def capture_button(chamberindex, flagged_set):
    def on_button_clicked(b, flagged_set=flagged_set):
        if b['new'] == True:
            flagged_set.add(chamberindex)
            b['new'] = False
        else:
            b['new'] = True
            flagged_set.remove(chamberindex)
    return on_button_clicked

# If a chamber is flagged, change the button color color and font weight
def create_toggle_button(label):
    return widgets.ToggleButton(
    value=False,
    description=label,
    disabled=False,
    layout=widgets.Layout(height='25px', width='120px', min_width='30px'),
    button_style='',
    tooltip='label',
    style={'font_weight': 'bold'}, 
    icon='')

# Convert image array to byte array for display in widget
def imagearray_to_bytearr(img_arr, format = 'png'):
    """Convert image to a byte array
    """
    rescaled_image = np.interp(x = img_arr, 
                               xp = (img_arr.min(), img_arr.max()), # xp is the range of the image array
                               fp = (0, img_arr.max()*30) # 10**6 is the maximum value that can be displayed in the widget; in general, increasing this number will increase the brightness of the image
                               ).astype(np.uint32)
    img_byte_buffer = io.BytesIO()
    pil_img = Image.fromarray(rescaled_image)
    pil_img.save(img_byte_buffer, format = format)
    img_byte_arr = img_byte_buffer.getvalue()
    return img_byte_arr

# Create image pane for display in widget
def make_image_pane(imagearr):
    """Initialize single image pane
    """
    return widgets.Image(
            value= imagearray_to_bytearr(imagearr),
            format='png',
            width='80px', height='80px', 
            min_width='80px', 
            max_height = '80px', max_width = '80px')

# Create button grid for display in widget
def make_button_grid(button_stamps_subset, flagged_set):
    """Initialize button grid
    """

    items1 = []
    for enzymeconc, indices, imagearr in button_stamps_subset:
        chamber_index = str(indices)+' '+str(round(enzymeconc, 1))+' nM'
        tb = create_toggle_button(chamber_index)
        
        record_index_name = str(indices[0])+','+str(indices[1])
        tb.observe(capture_button(record_index_name, flagged_set), 'value')
        hb = widgets.VBox([tb, make_image_pane(imagearr), widgets.Label('.\n.')], 
                          width='120px', 
                          height = '220px',
                          min_height='220px',
                          min_width='120px',
                          overflow_x = 'none')
        hb.layout.align_items = 'center'
        hb.layout.object_fit = 'contain'
        hb.layout.object_position = 'top center'
        items1.append(hb)

    box_layout = widgets.Layout(overflow='scroll',
                    border='3px solid black',
                    width='1200px',
                    height='600px',
                    flex_flow='row wrap',
                    min_width='200px',
                    display='flex')
    scrollbox = widgets.Box(items1, layout = box_layout)

    return scrollbox
 

# Invoking ipywidget functions above, create button array to manually cull chambers
def manual_culling(sq_merged, egfp_button_summary_image_path, NUM_ROWS, NUM_COLS, culling_export_directory):
    """Creates an array of interactive buttons to cull manually, rankng by summed
    eGFP intensity
    Parameters:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all chambers.
    - egfp_button_summary_image_path (string): Path to directory containing eGFP images for each chamber.
    - NUM_ROWS (int): Number of rows in the button grid.
    - NUM_COLS (int): Number of columns in the button grid.
    Returns:
    - flagged_set (set): Set of chamber indices to be culled.
    - button_grid (ipywidget): Widget containing button grid for manual culling.
    - button_stamps (list):
    """

    # Check export directory for a previous culling record
    files = os.listdir(culling_export_directory)

    culling_record_exists = False
    for file in files:
        if 'cull' in file and '.csv' in file:
            culling_filename = file
            culling_record_exists = True
            break

    # If culling record exists, load it and create a set of flagged chambers
    if culling_record_exists == True:

        # create culling filepath from culling filename and export directory
        culling_export_filepath = os.path.join(culling_export_directory, culling_filename)

        # load culling record
        df = pd.read_csv(culling_export_filepath)

        # locate values of Indices column in rows where egfp_manual_flag is True
        flagged_set = set(df.loc[df['egfp_manual_flag'] == True]['Indices'])

        # create button grid
        for i in flagged_set:
            capture_button(i, flagged_set=flagged_set)

        # Create an abbreviated filepath string to show two levels of parent directories above the culling record
        culling_export_filepath_split = culling_export_filepath.split('/')
        culling_filepath_abbrev = '.../' + '/'.join(culling_export_filepath_split[-4:])

        # Print
        print('Culling record found: %s. \nLoaded culling record.' % culling_filepath_abbrev)

        # get expression values from all chambers
        all_chambers_expression = sq_merged[[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first') # removes duplicate enzyme concentrations from the same chamber

        # get Indices from all chambers
        concs_Indices = all_chambers_expression.loc[all_chambers_expression.EnzymeConc > 0][['EnzymeConc','Indices']].sort_values('EnzymeConc', ascending = False).to_numpy().tolist() # sort by enzyme concentration, descending
        Indices_to_visualize = [(enzyme_conc, tuple([int(i) for i in i.split(',')])) for enzyme_conc, i in concs_Indices] # convert Indices to tuple of ints

        # create button array
        button_img_arr = np.asarray(Image.open(egfp_button_summary_image_path))
        height, width = button_img_arr.shape

        # define stamp dimensions
        stamp_height = int(height/NUM_ROWS)
        stamp_width = int(width/NUM_COLS)

        # create button stamps
        button_stamps = {}
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                button_stamps[(col+1, row+1)] = button_img_arr[row*stamp_height:row*stamp_height+stamp_height, col*stamp_width:col*stamp_width+stamp_width]

        # create button grid
        button_grid = None

    # If culling record does not exist, create a new one
    elif culling_record_exists == False:

        print('No culling record found in export directory. Creating button grid for manual culling...')

        # create empty set to store flagged chambers and make this set global
        flagged_set = set()

        # get expression values from all chambers
        all_chambers_expression = sq_merged[[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')

        # get Indices from all chambers
        concs_Indices = all_chambers_expression.loc[all_chambers_expression.EnzymeConc > 0][['EnzymeConc','Indices']].sort_values('EnzymeConc', ascending = False).to_numpy().tolist()
        Indices_to_visualize = [(enzyme_conc, tuple([int(i) for i in i.split(',')])) for enzyme_conc, i in concs_Indices]

        # create button array
        button_img_arr = np.asarray(Image.open(egfp_button_summary_image_path))
        height, width = button_img_arr.shape

        # define stamp dimensions
        stamp_height = int(height/NUM_ROWS)
        stamp_width = int(width/NUM_COLS)

        # create button stamps
        button_stamps = {}
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                button_stamps[(col+1, row+1)] = button_img_arr[row*stamp_height:row*stamp_height+stamp_height, col*stamp_width:col*stamp_width+stamp_width]

        # create subset of button stamps to visualize (based on enzyme concentration)
        button_stamps_subset = [(enzymeconc, key, button_stamps[key]) for enzymeconc, key in Indices_to_visualize]

        # create button grid
        button_grid = make_button_grid(button_stamps_subset=button_stamps_subset, flagged_set=flagged_set)

        print('Button grid object created. Please wait for your viewport to render the grid...')

    return button_grid, flagged_set, button_stamps


# Handle flagged chambers
def handle_flagged_chambers(sq_merged, flagged_set, culling_export_directory):
    """Checks for existing culling file, updates and exports record
    Parameters:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all 
                                    chambers, with flagged chambers removed.
    - flagged_set (set): Set of chamber indices that were flagged for culling.
    - culling_export_directory (str): Path to directory where culling record is stored.
    Returns:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all 
                                    chambers, with flagged chambers removed.
    """

    # Check export directory for a previous culling record
    files = os.listdir(culling_export_directory)

    # If culling record exists, load it and create a set of flagged chambers
    culling_record_exists = False # initialize culling record exists flag
    for file in files:
        # ignore hidden files
        if not file.startswith('.'):
            # if culling record exists, load it
            if 'cull' in file and '.csv' in file:
                culling_filename = file
                culling_record_exists = True

    # If culling record exists, load it and create a set of flagged chambers
    if culling_record_exists == True:
        culling_export_filepath = os.path.join(culling_export_directory, culling_filename)
        df = pd.read_csv(culling_export_filepath)
        # replace NaNs with False
        df['egfp_manual_flag'] = df['egfp_manual_flag'].fillna(False)
        flagged_set = set(df.loc[df['egfp_manual_flag'] == True]['Indices'])
        print('Found culling record for the following chambers: %s' % str(flagged_set))

        # add flagged chambers to sq_merged
        bool_flagged_set = set(sq_merged.loc[sq_merged['Indices'].isin(flagged_set)]['Indices'])
        sq_merged['egfp_manual_flag'] = sq_merged.Indices.apply(lambda i: i in bool_flagged_set)
        print('Added flagged chambers to sq_merged.')

    # If culling record does not exist, create a new one
    elif culling_record_exists == False:
        # reformat each item in flagged set to add leading zeros
        flagged_set = set(['%02d,%02d' % (int(i.split(',')[0]), int(i.split(',')[1])) for i in flagged_set])
        print('Creating new culling record for the flagged chambers: %s' % str(flagged_set))

        # create new column in sq_merged to store culling status based on flagged set
        for i in flagged_set:
            sq_merged.loc[sq_merged['Indices'] == i, 'egfp_manual_flag'] = True

        # if culled set is empty, add a column of False values
        if len(flagged_set) == 0:
            sq_merged['egfp_manual_flag'] = False
            
        # save culling record
        culling_export_filepath = os.path.join(culling_export_directory, 'manual_culling_record.csv')
        sq_merged.sort_values(['x', 'y',  'MutantID', 'substrate_conc_uM', 'egfp_manual_flag']).to_csv(culling_export_filepath, index=False)
        print('Saved culling record to %s' % culling_export_filepath)
    
    # replace nan in egfp_manual_flag column with False
    sq_merged['egfp_manual_flag'] = sq_merged['egfp_manual_flag'].fillna(False)

    return sq_merged


############################################################################################################
# INITIAL RATE FITTING
############################################################################################################

# define exponential function amd vectorize
def exponential(t, A, k, y0): 
    return A*(1-np.exp(-k*t))+y0
v_exponential = np.vectorize(exponential)

# fit first order exponential 
def fit_first_order_exponential_row(row, nfev_exp, pbp_conc=None):
    """Fit first order exponential to data from current row
    """
    try:
        # get x and y data
        xdata, ydata = row.time_s, row.kinetic_product_concentration_uM

        # if pbp_conc is not None, remove points where product concentration is greater than 80% of the PBP concentration
        if pbp_conc is not None:
            xdata = [ xdata[i] for i in range(len(xdata)) if ydata[i] < 0.8 * pbp_conc ]
            ydata = [ ydata[i] for i in range(len(ydata)) if ydata[i] < 0.8 * pbp_conc ]

        # perform fit with max nfev of 1000 iterations
        popt, pcov = optimize.curve_fit(exponential, xdata, ydata, bounds=([0, 0, -20], [np.inf, 0.05, 75]), max_nfev=nfev_exp) # A, k, y0

        # get R^2 of fit
        Rsq = np.corrcoef(ydata, v_exponential(xdata, *popt))[0,1]**2

        # finally, get the estimated kcat/KM from kobs
        enzyme_conc_uM, substrate_conc_uM, kobs, A = row.EnzymeConc/1000, row.substrate_conc_uM, popt[1], popt[0]

        # calculate kcat/KM
        if enzyme_conc_uM == 0: # if enzyme concentration is 0, kcat/KM is undefined
            kobs_kcat_KM = np.nan
        else: # otherwise, calculate kcat/KM
            kobs_kcat_KM = kobs/enzyme_conc_uM # in uM^-1 s^-1
            kobs_kcat_KM = kobs_kcat_KM * 10**6 # in M^-1 s^-1

    except:
        popt = [np.nan, np.nan, np.nan]
        Rsq = np.nan
        kobs_kcat_KM = np.nan


    return popt, Rsq, kobs_kcat_KM

# define the initial rate determination algorithm for pbp coupled reactions
def fit_initial_rate_row_pbp(row, pbp_conc):
    try:
        substrate_conc = row.substrate_conc_uM
        x, y = row.time_s, row.kinetic_product_concentration_uM
        first_third = int(len(x)*0.3)
        regime = 0

        # "regime 1": the substrate concentration is less than the concentration of PBP.
        if substrate_conc < pbp_conc: # "regime 1"
            regime = 1

            # if less than or equal to 30% of substrate has turned before aqusition of second point,
            # fit the first 30% of the data
            # if y[1] < 0.3*substrate_conc:
            x = x[0:first_third]
            y = y[0:first_third]

            # else, fit the first two points
            # else:
            #     x = x[0:1]
            #     y = y[0:1]

        # "regime 2": 30% of the substrate concentration is greater than 2/3 the concentration of PBP.
        elif 0.3*substrate_conc > (1/3)*pbp_conc: # regime "2"
            regime = 2

            # fit points within the linear regime of the standard curve, i.e. 2/3 of the PBP concentration
            x = [ x[i] for i in range(len(x)) if y[i] < (1/3)*pbp_conc ]
            y = [ y[i] for i in range(len(y)) if y[i] < (1/3)*pbp_conc ]

        # "regime 3": 30% of the substrate concentration is below 2/3 the concentration of PBP. 
        elif 0.3*substrate_conc < (2/3)*pbp_conc: # regime "3"
            regime = 3

            # fit all points in which the product concentration is less than 30% of the total substrate concentration
            x = [ x[i] for i in range(len(x)) if y[i] < 0.3*substrate_conc ]
            y = [ y[i] for i in range(len(y)) if y[i] < 0.3*substrate_conc ]


        # exclude points where product concentration is greater than 80% of the PBP concentration
        x = [ x[i] for i in range(len(x)) if y[i] < 0.8*pbp_conc ]
        y = [ y[i] for i in range(len(y)) if y[i] < 0.8*pbp_conc ]

        # now save number of points fit for later
        num_points_fit_vo = len(x)

        # reshape the arrays for least-squares regression
        x = np.array(x)
        y = np.array(y)

        # perform fit
        m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0] # stores initial rate and intercept, the first two objects in the array

        # set two point flag
        if len(x) == 2:
            two_point = True
        else:
            two_point = False
            
        return m, c, two_point, regime, num_points_fit_vo
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
def fit_initial_rate_row_linear(row):
    """Fit initial rate for a given row of sq_merged
    """
    try:
        x, y = row.time_s, row.kinetic_product_concentration_uM
        num_points_fit_vo = len(x)

        # fit only first 30% of the data
        first_third = int(len(x)*0.3)
        half = int(len(x)*0.5)
        x = x[0:half]
        y = y[0:half]
        # x = x[0:first_third]
        # y = y[0:first_third]
        num_points_fit_vo = len(x)
        
        # # if all points are fit, refit with first few points
        # if num_points_fit_vo == len(x):
        #     m, c = scp.stats.linregress(x[0:3], y[0:3])[0:2]
        #     num_points_fit_vo = 3

        m, c = scp.stats.linregress(x, y)[0:2]

        # two point fit, regime, and number of points fit
        two_point = False
        regime = None

        return m, c, two_point, regime, num_points_fit_vo
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan
        
# Get initial rates
def get_initial_rates(sq_merged, fit_type, pbp_conc, nfev_exp):
    """Apply initial rate fitting algorithm to dataframe
    Parameters:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all chambers.
    - pbp_conc (int): Concentration of PBP in uM.
    Returns:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all chambers, 
                                    with initial rates added.
    """

    # calculate the initial rates
    print('1) Fitting initial rates...')
    if fit_type == 'pbp':
        results = sq_merged.apply(fit_initial_rate_row_pbp, axis=1, pbp_conc=pbp_conc)
    else:
        results = sq_merged.apply(fit_initial_rate_row_linear, axis=1)
    print('Done fitting initial rates. Adding results to dataframe...\n')

    # now add the rates and intercepts to the dataframe
    sq_merged['initial_rate'] = results.apply(lambda x: x[0])
    sq_merged['initial_rate_intercept'] = results.apply(lambda x: x[1])
    sq_merged['two_point_fit'] = results.apply(lambda x: x[2])
    sq_merged['rate_fit_regime'] = results.apply(lambda x: x[3])
    sq_merged['num_points_fit_vo'] = results.apply(lambda x: x[4])

    # calculate the exponential fit parameters
    print('2) Fitting exponential parameters...')
    # use tqdm to display progress bar
    for index, row in tqdm(sq_merged.iterrows(), total=sq_merged.shape[0]):
        row_results = fit_first_order_exponential_row(row, nfev_exp, pbp_conc=pbp_conc)
        sq_merged.at[index, 'exponential_A'] = row_results[0][0]
        sq_merged.at[index, 'exponential_kobs'] = row_results[0][1]
        sq_merged.at[index, 'exponential_y0'] = row_results[0][2]
        sq_merged.at[index, 'exponential_R2'] = row_results[1]
        sq_merged.at[index, 'kobs_kcat_KM'] = row_results[2]

    # results = sq_merged.parallel_apply(fit_first_order_exponential_row, axis=1, pbp_conc=pbp_conc)
    print('Done fitting exponential parameters. Adding results to dataframe...\n')

    return sq_merged

# Plot progress curves
def plot_sample_progress_curves(sq_merged, pbp_conc=None, target_MutantID=None, target_chamber=None):
    """Plot progress curves for a random library member
    Arguments:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all chambers.
    - pbp_conc (int): Concentration of PBP in uM.
    - target_MutantID (str): Mutant ID to plot progress curves for (e.g. 'hsapiens2_wt')
    - target_chamber (int): Chamber index to plot progress curves for (e.g. '01,11')
    Returns:
    - None
    """

    print('Plotting progress curves for one random library member...')
    
    # set list of substrate concentrations
    concs = sorted(list(set(sq_merged['substrate_conc_uM'])))
    substrates = sorted(list(set(sq_merged['substrate'])))
    conc_d = dict(zip(range(len(concs)), concs))

    #### Target Mutant/Chamber ####
    # if target mutant and chamber are given, plot progress curves 
    # for that mutant and chamber
    if target_MutantID is not None and target_chamber is not None:
        check = np.random.choice(list(set(sq_merged['MutantID'])), 1)[0]
        my_df = sq_merged[sq_merged['MutantID'] == check]
    
    # if target mutant, plot progress curves for that mutant
    elif target_MutantID is not None and target_chamber is None:
        my_df = sq_merged[sq_merged['MutantID'] == target_MutantID]

    # if target chamber, plot progress curves for corresponding mutant
    elif target_MutantID is None and target_chamber is not None:
        check = sq_merged[sq_merged['Indices'] == target_chamber]['MutantID'].iloc[0]
        my_df = sq_merged[sq_merged['MutantID'] == check]

    # if nothing is given, plot progress curves for a random mutant
    elif target_MutantID is None and target_chamber is None:
        check = np.random.choice(list(set(sq_merged['MutantID'])), 1)[0]
        my_df = sq_merged[sq_merged['MutantID'] == check]

    #### Initialize figure ####
    # set number of columns by number of substrate concentrations
    if len(concs) > 1:
        ncols = len(concs)
    else:
        ncols = 2

    # create figure
    fig, axs = plt.subplots(ncols=ncols, nrows=2, figsize=(ncols*3, 6), sharey=True)

    # get list of chambers and color dictionary
    selected_chambers = list(set(my_df['Indices']))
    color_d = dict(zip(selected_chambers, plt.cm.tab20.colors))

    # get all product concs for the selected chambers
    all_product_concs = my_df['kinetic_product_concentration_uM'].tolist()
    all_product_concs = [item for sublist in all_product_concs for item in sublist]
    # display(all_product_concs)

    #### Plot progress curves ####
    # loop through the concentrations
    for k,v in conc_d.items():
        v_df = my_df[my_df['substrate_conc_uM'] == v]

        # loop through the chambers at this concentration
        for index, row in v_df.iterrows():
            try:
                times, product_concs = row['time_s'], row['kinetic_product_concentration_uM']

                # get initial rate and intercept
                m, b = row.initial_rate, row.initial_rate_intercept

                # get exponential fit parameters
                A, k_obs, y0 = row.exponential_A, row.exponential_kobs, row.exponential_y0
                num_points_fit_vo = int(row.num_points_fit_vo)

                # get substrate concentration, indices, and fit regime
                substrate = row.substrate
                indices = row.Indices
                regime = row.rate_fit_regime
                color = color_d[indices]

                ## plot progress curves with linear fit
                axs[0,k].scatter(times[0:num_points_fit_vo], product_concs[0:num_points_fit_vo], s=10, label=indices, color=color) # subset the data to num_points_fit_vo and plot as o's
                axs[0,k].scatter(times[num_points_fit_vo:], product_concs[num_points_fit_vo:], s=10, marker='x', color=color) # subset the data to num_points_fit_vo and plot as x's
                axs[0,k].plot([0, max(times)], [b, m*max(times)+b], '--', color=color) # plot the linear fit
                axs[0,k].set_title(str(v) + 'uM\n' , loc='left')
                axs[0,k].set_ylim(bottom=0, top=1.1*max(all_product_concs))

                # plot progress curves with exponential fit
                axs[1,k].scatter(times, product_concs, s=10, label=indices, color=color)
                axs[1,k].plot(times, v_exponential(times, A, k_obs, y0), '--', color=color)
                axs[1,k].set_ylim(bottom=0, top=1.1*max(all_product_concs))
                axs[1,k].set_xlabel('Time (s)')

            except:
                # get initial rate and intercept
                times, product_concs = row['time_s'], row['kinetic_product_concentration_uM']
                substrate = row.substrate

                # get exponential fit parameters
                A, k_obs, y0 = row.exponential_A, row.exponential_kobs, row.exponential_y0

                # get substrate concentration, indices, and fit regime
                substrate = row.substrate
                indices = row.Indices
                regime = row.rate_fit_regime
                color = color_d[indices]

                # plot progress curves with linear fit
                axs[0,k].scatter(times, product_concs, s=10, label=indices, color=color) # subset the data to num_points_fit_vo and plot as o's
                axs[0,k].set_title(str(v) + 'uM\n' + substrate, loc='left')
                axs[0,k].set_ylim(bottom=0, top=1.1*max(all_product_concs))

                # plot progress curves with exponential fit
                axs[1,k].scatter(times, product_concs, s=10, label=indices, color=color)
                axs[1,k].plot(times, v_exponential(times, A, k_obs, y0), '--', color=color)
                axs[1,k].set_ylim(bottom=0, top=1.1*max(all_product_concs))
                axs[1,k].set_xlabel('Time (s)')

        
        ## define coordinates for on top of the right corner of the plot
        coords = (1, 1.01)
        axs[0,k].text(coords[0], coords[1], 'Regime: ' + str(regime), transform=axs[0,k].transAxes, horizontalalignment='right', verticalalignment='bottom')
        axs[1,k].text(coords[0], coords[1], 'Exponential Fit', transform=axs[1,k].transAxes, horizontalalignment='right', verticalalignment='bottom')
    
    # add legend1 for points used in linear fit
    handles1 = [
        Line2D([0], [0], marker='o', color='none', label='Included',
                markerfacecolor='black', markersize=5),
        Line2D([0], [0], marker='x', color='none', label='Excluded',
                markerfacecolor='black', markersize=5),
    ]
    legend1 = axs[0,-1].legend(handles=handles1, bbox_to_anchor=(1.1, 1), loc='upper left', title='Points Used ')
    axs[0,-1].add_artist(legend1)

    # add legend2 for chambers
    handles2 = []
    for i in selected_chambers:
        handles2.append(Line2D([0], [0], marker='o', color='none', label=i,
                markerfacecolor=color_d[i], markersize=12))
    legend2 = axs[0,-1].legend(handles=handles2, bbox_to_anchor=(1.1, 0.6), loc='upper left', title='Chambers')
    axs[0,-1].add_artist(legend2)

    # only set the y label for the first plot
    for ax in axs.flat:
        ax.set_ylabel('Product Concentration (uM)')

    # add suptitle, tight layout, and show
    fig.suptitle("Library Member: " + check + '\n{}'.format(substrate), y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()



## =================================================================================================
# MICHAELIS-MENTEN FITTING
## =================================================================================================

# define michaelis menten equation
def mm_func(S, Km, Vmax): 
    return Vmax*S/(Km+S)
v_mm_func = np.vectorize(mm_func)


# fit michaelis-menten equation to the initial rates
def fit_michaelis_menten(sq_merged, exclude_concs=[], local_bg_ratio_threshold=2):
    """Fits the Michaelis-Menten equation to the initial rates of the progress curves.
    Parameters:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all chambers, 
                                    with initial rates calculated.
    - exclude_concs (list): List of substrate concentrations to exclude from the fit.
    Returns:
    - sq_merged (pandas dataframe): Contains kinetic and standard data for all chambers, 
                                    with initial rates calculated and Michaelis-Menten 
                                    fit parameters calculated.
    """

    ###################
    # DATAFRAME SETUP #
    ###################

    # if egfp_manual_flag column does not exist, create it
    if 'egfp_manual_flag' not in sq_merged.columns:
        sq_merged['egfp_manual_flag'] = np.nan

    # first, group by Indices to create initial rate and substrate conc series
    initial_rates = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['initial_rate'].apply(list).reset_index(name='initial_rates')
    num_points_fit_vo = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['num_points_fit_vo'].apply(list).reset_index(name='num_points_fit_vo')
    exponential_A = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['exponential_A'].apply(list).reset_index(name='exponential_A')
    exponential_kobs = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['exponential_kobs'].apply(list).reset_index(name='exponential_kobs')
    exponential_y0 = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['exponential_y0'].apply(list).reset_index(name='exponential_y0')
    exponential_R2 = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['exponential_R2'].apply(list).reset_index(name='exponential_R2')
    exponential_kcat_over_km = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['kobs_kcat_KM'].apply(list).reset_index(name='exponential_kcat_over_km')
    substrate_concs = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['substrate_conc_uM'].apply(list).reset_index(name='substrate_concs')
    if 'local_bg_ratio_at_substrate_conc' in sq_merged.columns:
        local_bg_ratios = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['local_bg_ratio_at_substrate_conc'].apply(list).reset_index(name='local_bg_ratios')

    # merge all of the series into one dataframe
    squeeze_mm = pd.merge(substrate_concs, initial_rates, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    squeeze_mm = pd.merge(squeeze_mm, num_points_fit_vo, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    squeeze_mm = pd.merge(squeeze_mm, exponential_A, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    squeeze_mm = pd.merge(squeeze_mm, exponential_kobs, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    squeeze_mm = pd.merge(squeeze_mm, exponential_y0, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    squeeze_mm = pd.merge(squeeze_mm, exponential_R2, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    squeeze_mm = pd.merge(squeeze_mm, exponential_kcat_over_km, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])
    if 'local_bg_ratio_at_substrate_conc' in sq_merged.columns:
        squeeze_mm = pd.merge(squeeze_mm, local_bg_ratios, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])

    # get the substrate concentration list from the first row
    all_substrate_concs = squeeze_mm['substrate_concs'].iloc[0]

    #################################
    # FIT MICHAELIS-MENTEN EQUATION #
    #################################

    # define curve fit function
    def fit_mm_curve(df, exclude_concs=exclude_concs):
        """
        Fits the Michaelis-Menten equation to the initial rates of the progress curves. This 
        function excludes substrate concentrations if they match the following criteria:
            1. The user specifies to exclude them
            2. The initial rate is negative
            3. The initial rate is approximately non-increasing from the previous substrate concentration

        Parameters:
        df (pandas dataframe): Contains kinetic and standard data for all chambers,
                                with initial rates calculated.
        exclude_concs (list): List of substrate concentrations to exclude from the fit.

        Returns:
        df (pandas dataframe): Contains kinetic and standard data for all chambers,
                                with initial rates calculated and Michaelis-Menten
                                fit parameters calculated.
        """

        # define data
        xdata = df.substrate_concs
        ydata = df.initial_rates
        enzyme_conc = df.EnzymeConc
        num_points_fit_vo = df.num_points_fit_vo
        local_bg_ratios = df.local_bg_ratios

        # create list of substrate concentrations to exclude from fit in this chamber
        # note: cannot overwrite exclude_concs variable because it is a global variable!
        chamber_exclude_concs = []
        if (exclude_concs != None) and (len(exclude_concs) > 0):
            chamber_exclude_concs = chamber_exclude_concs + exclude_concs

        ydata_final = copy.copy(ydata)
        xdata_final = copy.copy(xdata)

        # # exclude substrate concentrations from fit if the local_bg_ratio is below threshold
        # if 'local_bg_ratio_at_substrate_conc' in sq_merged.columns:
        #     for i, ratio in enumerate(local_bg_ratios):
        #         if ratio < local_bg_ratio_threshold:
        #             # append to list of substrate concentrations to exclude
        #             chamber_exclude_concs.append(xdata_final[i])

        # exclude substrate concentrations from fit if the initial rate is negative
        for i, rate in enumerate(ydata_final):
            if i != 0: # don't exclude the first substrate concentration since this can sometimes be near zero
                if rate < 0:
                    # append to list of substrate concentrations to exclude
                    chamber_exclude_concs.append(xdata_final[i])

        # exclude substrate concs from fit if slope is zero
        for i, rate in enumerate(ydata_final):
            if i != 0:
                if rate == 0:
                    # append to list of substrate concentrations to exclude
                    chamber_exclude_concs.append(xdata_final[i])

        # # exclude substrate concentrations from fit if the initial rate is approximately non-increasing
        # for i, rate in enumerate(ydata_final):
        #     if i != 0:
        #         if rate < ydata_final[i-1]*0.9:
        #             # append to list of substrate concentrations to exclude
        #             chamber_exclude_concs.append(xdata_final[i])
                
        #         # # for rates that remain low after dropping, exclude them
        #         # if rate < 




        # now remove excluded substrate concentrations from the fit
        chamber_exclude_concs = list(set(chamber_exclude_concs)) # remove duplicates
        if chamber_exclude_concs != None:
            for conc in chamber_exclude_concs:
                if conc in xdata_final:
                    # remove from lists without changing order
                    ydata_final.pop(xdata_final.index(conc))
                    xdata_final.remove(conc)

        # perform curve fit
        try:
            # define KM bounds
            KM_min = min(xdata_final)*0.2 # set lower bound to 0.5 times the minimum substrate concentration
            KM_max = max(xdata_final)*2 # set upper bound to 1.5 times the maximum substrate concentration

            # fit curve and get parameters
            mm_params, pcov = optimize.curve_fit(mm_func, xdata=xdata_final, ydata=ydata_final, bounds=([KM_min, 0], [KM_max, np.inf])) # Km, Vmax
        except:
            mm_params = [np.nan, np.nan]

        # store variables
        KM_fit = mm_params[0] # at this point, this is in uM
        vmax_fit = mm_params[1] # at this point, in uM per second
        if enzyme_conc != 0:
            kcat_fit = vmax_fit/(enzyme_conc/1000) # at this point, enzyme conc is in nM, so convert to uM; kcat is in s^-1
        else:
            kcat_fit = np.nan

        # convert to kcat/KM in M^-1 s^-1
        MM_kcat_over_KM_fit = 10**6 * (kcat_fit/KM_fit) # at this point, kcat in s^-1 and KM in uM, so multiply by 1000000 to give s^-1 * M^-1 
        
        # calculate r^2 value
        if len(xdata_final) > 1:
            r2 = np.corrcoef(ydata_final, v_mm_func(xdata_final, *[KM_fit, vmax_fit]))[0, 1]**2 # calculate r^2 value
        else:
            r2 = np.nan

        # correct infinite MM values to 0
        if np.isinf(MM_kcat_over_KM_fit):
            MM_kcat_over_KM_fit = 0
        
        if np.isinf(kcat_fit):
            kcat_fit = 0

        if np.isinf(KM_fit):
            KM_fit = 0

        # check if exclude concs is empty
        if len(chamber_exclude_concs) == 0:
            chamber_exclude_concs = str([])
        else:
            chamber_exclude_concs = str(chamber_exclude_concs)

        return KM_fit, vmax_fit, kcat_fit, MM_kcat_over_KM_fit, float(r2), chamber_exclude_concs

    # apply function and store fit parameters
    print('Fitting Michaelis-Menten curves...')
    for index, row in tqdm(squeeze_mm.iterrows(), total=squeeze_mm.shape[0]):
        KM_fit, vmax_fit, kcat_fit, MM_kcat_over_KM_fit, r2, exclude_concs = fit_mm_curve(row)
        squeeze_mm.at[index, 'KM_fit'] = KM_fit
        squeeze_mm.at[index, 'vmax_fit'] = vmax_fit
        squeeze_mm.at[index, 'kcat_fit'] = kcat_fit
        squeeze_mm.at[index, 'MM_kcat_over_KM_fit'] = MM_kcat_over_KM_fit
        squeeze_mm.at[index, 'MM_kcat_over_KM_fit_R2'] = r2
        squeeze_mm.at[index, 'exclude_concs'] = exclude_concs

    print('Done fitting Michaelis-Menten curves.')


    ##########################################
    # PLOT EXAMPLES OF MICHAELIS-MENTEN FITS #
    ##########################################

    # select example chambers to plot
    print('Plotting examples...')
    num_examples = 4
    examples_df = squeeze_mm[squeeze_mm['MutantID'] != 'BLANK'].sample(num_examples)
    examples_df = examples_df.reset_index()

    # plot subplots
    fig, axs = plt.subplots(1, num_examples, figsize=(16, 5), sharey=True)

    # initialize maximum vmax for y-axis scaling
    max_vmax = 0

    # iterate 
    for index, row in examples_df.iterrows():
        xdata = row.substrate_concs
        ydata = row.initial_rates
        vmax_fit = row.vmax_fit
        KM_fit = row.KM_fit
        # literal eval to convert string to list
        if (row.exclude_concs == '[]') or (row.exclude_concs == []):
            exclude_concs = []
        else:
            exclude_concs = ast.literal_eval(row.exclude_concs)

        # if vmax is greater than current max, update max
        if vmax_fit > max_vmax:
            max_vmax = vmax_fit

        # plot data and curve fit
        t = np.arange(0, float(max(all_substrate_concs)), float(max(all_substrate_concs))/500)
        axs[index].plot(t, v_mm_func(t, *[KM_fit, vmax_fit]), 'g--')
        axs[index].plot(xdata, ydata, 'bo')
        axs[index].axhline(vmax_fit)
        axs[index].set_xlabel('Substrate Concentration (uM)')

        # plot excluded concentrations as grey circles
        xdata_excluded = [x for x in xdata if x in exclude_concs]
        ydata_excluded = [ydata[xdata.index(x)] for x in xdata if x in exclude_concs]
        axs[index].plot(xdata_excluded, ydata_excluded, 'o', color='grey')

        # define title
        name = row.MutantID 

        # if name is too long, split it into two lines
        if len(row.MutantID) > 25:
            # find underscore closest to middle of string
            underscore_indices = [i for i, letter in enumerate(name) if letter == '_']
            middle_index = int(len(name)/2)
            closest_index = min(underscore_indices, key=lambda x:abs(x-middle_index))

            # replace underscore with newline
            name = name[:closest_index] + '\n' + name[closest_index+1:]
        else:
            name = row.MutantID

        # add title with concentration rounded to 2 decimal places
        axs[index].set_title('(' + row.Indices + ')' + ' ' + name + '\n' + 'Enz Conc: ' + str(round(row.EnzymeConc, 2)) + ' nM')

        # define legend text, which includes fit parameters and R2 (kcat/KM is in scientific notation)
        # if kcat/KM is greater than 10**6, convert to scientific notation
        if row.MM_kcat_over_KM_fit > 10**6:
            kcat_over_KM_sci = '%.2E' % Decimal(row.MM_kcat_over_KM_fit)
        else:
            kcat_over_KM_sci = str(round(row.MM_kcat_over_KM_fit, 2))

        legend_text = '$K_M = $' + str(round(KM_fit, 2)) + ' uM' + '\n' + '$k_{cat} = $' + str(round(row.kcat_fit, 2)) + ' $s^{-1}$' + '\n' + '$k_{cat}/K_M = $' + kcat_over_KM_sci + ' $M^{-1}s^{-1}$' + '\n' + '$R^2 = $' + str(round(row.MM_kcat_over_KM_fit_R2, 2))
        
        # add legend text to legend
        handles = [Line2D([0], [0], color='g', linestyle='--')]
        axs[index].legend(handles=handles, labels=[legend_text, 'Data'], fancybox=True, shadow=True)    

    # set y-axis limit and plot layout
    axs[0].set_ylabel('Initial Rate (uM/s)') # only label y-axis for leftmost plot
    plt.ylim(0, max_vmax*2.5) # set y-axis limit
    plt.tight_layout()

    # create column for number of points to fit for each chamber
    squeeze_mm['MM_points_to_fit'] = squeeze_mm.apply(lambda row: (len(row.initial_rates) - len(ast.literal_eval(row.exclude_concs))), axis=1)

    ####################################
    # REFORMAT EXPONENTIAL FIT COLUMNS #
    ####################################

    # add "_list" to end of exponential parameter colomn names
    for col in squeeze_mm.columns:
        if 'exponential' in col:
            squeeze_mm = squeeze_mm.rename(columns={col: col + '_list'})                                                                                                                                                                                                                                                                                                                                                                                                                                                              

    return squeeze_mm

# get local background indices
def get_local_bg_indices(x,y,device_rows):
    """Get indices of local background chambers for a given chamber.

    Arguments:
    x (int): x coordinate of chamber
    y (int): y coordinate of chamber
    device_rows (int): number of rows in device

    Returns:
    local_bg_indices (list): list of tuples containing x and y coordinates of local background chambers
    """
    
    # initialize list of indices
    local_bg_indices = []

    # chamber is at the top of a column
    if y == 1:
        local_bg_indices.append((x, y+1))
    
    # chamber is at bottom of column
    elif y == device_rows: 
        local_bg_indices.append((x, y - 1))
    
    # chamber is at bottom of column
    else: 
        local_bg_indices.append((x, y - 1))
        local_bg_indices.append((x, y + 1))

    return local_bg_indices


# ==========================================================================================
# CALCULATE LOCAL BACKGROUND RATIO
# ==========================================================================================

# calculate local background ratio for every chamber
def calculate_local_bg_ratio(squeeze_mm, sq_merged, target_substrate_conc, device_rows):
    """Calculate local background ratio for every chamber.

    Parameters:
    squeeze_mm (pandas dataframe): dataframe containing squeeze data
    sq_merged (pandas dataframe): dataframe containing merged squeeze data
    device_rows (int): number of rows in device
    exclude_concs (list): list of concentrations to exclude from local background calculation

    Returns:
    squeeze_mm (pandas dataframe): dataframe containing squeeze data with local background ratio added
    """

    ##################################
    # STORE LOCAL BACKGROUND INDICES #
    ##################################

    # for every chamber, get local background indices
    local_bg_dict = {}
    for index, row in tqdm(squeeze_mm.iterrows(), total=len(squeeze_mm), desc='Storing local background indices...'):
        x, y = row.x, row.y
        lbg_idxs = get_local_bg_indices(x,y,device_rows)
        local_bg_dict[(x,y)] = lbg_idxs


    #####################################
    # CALCULATE LOCAL BACKGROUND RATIOS #
    #####################################
    
    # for every chamber in sq_merged, get local background rates
    for S in tqdm(set(sq_merged['substrate_conc_uM']), desc='Calculating local background rates...'):
        subset_merged = sq_merged[sq_merged['substrate_conc_uM'] == S]
        subset_merged['index_tuple'] = subset_merged.apply(lambda row: (row.x, row.y), axis=1)
        for index, row in tqdm(subset_merged.iterrows(), total=len(subset_merged), desc='Calculating local background rates for ' + str(S) + ' uM...'):
            x, y = row.x, row.y
            chamber_rate = row.initial_rate

            # get local (above and below) chamber neighbors
            lbg_idxs = local_bg_dict[(x,y)]

            # get rates for each of neighboring chambers
            lbg_rates = []

            # for every local bg chamber, get rate
            for tup in lbg_idxs: # for each local bg chamber...
                
                # get rate if chamber is in local_bg_df, else set to 0
                if tup in subset_merged['index_tuple'].tolist():
                    rate = subset_merged.loc[subset_merged['index_tuple'] == tup, 'initial_rate'].iloc[0]
                else:
                    rate = 0.00001

                # account for zero-slope rates
                if rate != 0:
                    lbg_rates.append(rate)
                else:
                    lbg_rates.append(0.01)
                
            # remove negative rates from list
            lbg_rates = [rate for rate in lbg_rates if rate > 0]

            # get max of local rates and calculate ratio
            if len(lbg_rates) > 1: # if there are multiple local bg rates, take the max
                lbg_rate = max(lbg_rates)
            elif len(lbg_rates) == 1: # if there is only one local bg rate, take that rate
                lbg_rate = lbg_rates[0]
            else: # if there are no local bg rates, set to 0.01
                lbg_rate = 0.01
            lbg_ratio = chamber_rate/lbg_rate

            # # if rates are near-zero, set ratio to 0.
            # # this prevents infinite ratios
            # if chamber_rate < 0.001:
            #     lbg_ratio = 0

            # account for negative ratios
            if lbg_ratio < 0:
                lbg_ratio = 0

            # store ratio in new column
            sq_merged.loc[index, 'local_bg_ratio'] = lbg_ratio


    ####################################
    # ADD LOCAL BG RATIO TO SQUEEZE_MM #
    ####################################
    
    # remove local_bg_ratio column if it already exists
    for col in squeeze_mm: 
        if col == 'local_bg_ratio':
            # drop column if it already exists
            squeeze_mm = squeeze_mm.drop(columns=['local_bg_ratio'])
    
    # get max local bg ratio for each chamber
    max_lbg_df = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['local_bg_ratio'].max().reset_index(name='local_bg_ratio')

    # merge max local bg ratio with squeeze_mm
    squeeze_mm = pd.merge(squeeze_mm, max_lbg_df, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])

    return squeeze_mm, sq_merged



# ==========================================================================================
# ADD EXPONENTIAL FIT PARAMETERS TO SQUEEZE_MM
# ==========================================================================================

# add exponential fit parameters to squeeze_mm
def add_exponential_fit_parameters(squeeze_mm, sq_merged, target_substrate_conc=None):
    if target_substrate_conc != None:
        print('Calculating exponential fit parameters for ' + str(target_substrate_conc) + ' uM...')
        for index, row in tqdm(squeeze_mm.iterrows(), total=len(squeeze_mm)):
            # get enzyme conc, substrate conc, kobs, and exp R^2
            substrate_concs = row['substrate_concs']
            exp_kobs = row['exponential_kobs_list']
            exp_r2 = row['exponential_R2_list']
            exp_kcat_over_KM = row['exponential_kcat_over_km_list']
            chamber_indices = row['Indices']

            # get index of target substrate conc
            target_substrate_index = substrate_concs.index(target_substrate_conc)

            # get kobs and kcat/KM at this index
            exp_kobs = exp_kobs[target_substrate_index]
            exp_kcat_over_KM = exp_kcat_over_KM[target_substrate_index]

            # create new columns
            squeeze_mm.loc[index, 'kobs_kcat_over_KM'] = exp_kcat_over_KM 

    else:
        print('Calculating exponential fit parameters for optimal substrate conc...')

        for index, row in tqdm(squeeze_mm.iterrows(), total=len(squeeze_mm)):
            # get enzyme conc, substrate conc, kobs, and exp R^2
            substrate_concs = row['substrate_concs']
            exp_kobs = row['exponential_kobs_list']
            exp_r2 = row['exponential_R2_list']
            exp_kcat_over_KM = row['exponential_kcat_over_km_list']
            chamber_indices = row['Indices']

            # 1) get dict of substrate concs and local bg ratios from sq_merged
            lbg_dict = dict(zip(substrate_concs, [0]*len(substrate_concs)))
            
            ### for every substrate conc, get local bg ratio and store in dict ###
            for i, conc in enumerate(substrate_concs):
                subset_df = sq_merged[(sq_merged['Indices'] == chamber_indices) & (sq_merged['substrate_conc_uM'] == conc)]
                if len(subset_df) > 0:
                    lbg_dict[conc] = subset_df['local_bg_ratio'].iloc[0]
            

            ### get lowest substrate conc where lbg is > 2, if it exists ###
            if all(np.isnan(exp_r2)) == False: # if the exp_r2 list is not all nan
                if min(lbg_dict.values()) > 2: # if lowest lbg is greater than 2
                    min_substrate_conc = min([conc for conc in substrate_concs if lbg_dict[conc] > 2])
                    min_substrate_index = substrate_concs.index(min_substrate_conc)
                else:
                    min_substrate_index = 0

                # if r2 of this index is < 0.9, set kobs and r2 to next conc up where r2 > 0.9
                while (exp_r2[min_substrate_index] < 0.9) or (np.isnan(exp_r2[min_substrate_index])):
                    min_substrate_index += 1
                    if min_substrate_index == len(exp_r2):
                        min_substrate_index -= 1
                        break

                # get kobs and kcat/KM at this index
                exp_kobs = exp_kobs[min_substrate_index]
                exp_kcat_over_KM = exp_kcat_over_KM[min_substrate_index]

            else:
                exp_kobs = np.nan
                exp_kcat_over_KM = np.nan


            # create new columns
            squeeze_mm.loc[index, 'kobs_kcat_over_KM'] = exp_kcat_over_KM

    return squeeze_mm, sq_merged


# ==========================================================================================
# PLOTTING FUNCTIONS
# ==========================================================================================

# plot progress curves in PDF output file
def plot_progress_curve(df, conc, ax, kwargs_for_scatter, kwargs_for_line, pbp_conc, substrate_name, fit_descriptors=False, exclude_concs=[]):
    """
    Inputs:
        - concs: current substrate concentrations
        - df: current chamber df
        - ax: position of plot
        - fit_descriptors: boolean to add descriptors for two-point fit, mm fit exclusion, and initial rate fit regime
    """

    # create a df of substrate concentrations
    conc_df = df[df['substrate_conc_uM'] == conc]

    # initialize list for all product concentrations
    all_product_concs = []

    # iterate through each substrate conentration
    for index, row in conc_df.iterrows():

        # get times and product concentrations
        times, product_concs = np.array(row['time_s']), row['kinetic_product_concentration_uM']
        all_product_concs.append(product_concs)

        # get initial rate, intercept, fit regime, and two-point fit boolean
        vi = row['initial_rate']
        intercept = row['initial_rate_intercept']
        regime = row['rate_fit_regime']
        two_point_fit = row['two_point_fit']

        # get exponential fit parameters
        A = row.exponential_A
        k_obs = row.exponential_kobs
        y0 = row.exponential_y0
        kobs_kcat_over_KM = row.kobs_kcat_KM
        exp_R2 = row.exponential_R2
        
        # if not nan
        if np.isnan(row.num_points_fit_vo) == False:
            num_points_fit_vo = int(row.num_points_fit_vo)
        else:
            num_points_fit_vo = 0

        # plot data for the current chamber if it is not NaN
        if type(product_concs) == list:

            # plot points
            ax.scatter(times[:num_points_fit_vo], product_concs[:num_points_fit_vo], marker='o', **kwargs_for_scatter) # plot points fit as circles
            
            # plot points excluded from fit as x's
            ax.scatter(times[num_points_fit_vo:], product_concs[num_points_fit_vo:], marker='x', **kwargs_for_scatter)
            
            # plot initial rate line
            ax.plot(times, (times*vi) + intercept, **kwargs_for_line) # plot initial rate line

            # plot exponential fit
            ax.plot(times, v_exponential(times, A, k_obs, y0), '--', alpha=0.5)

            # add text for fit descriptors
            if fit_descriptors==True:

                # set descriptor font size, spacing and starting y position
                descriptor_fontsize = 7

                # get number of substrate concs in experiment
                num_concs = len(df['substrate_conc_uM'].unique())
                
                # if 
                if num_concs > 5:
                    descriptor_y = -0.45
                    descriptor_vspacing = 0.15
                else:
                    descriptor_y = -0.35
                    descriptor_vspacing = 0.1

                # add linear title with underlined text
                ax.text(0, descriptor_y, "Lin Fit Regime: " + str(regime), transform=ax.transAxes, fontsize=descriptor_fontsize)

                # add two-point fit
                if two_point_fit == True:
                    ax.text(0, descriptor_y-descriptor_vspacing*1, "Two-point: " + str(two_point_fit), color='red', transform=ax.transAxes, fontsize=descriptor_fontsize)
                else:
                    ax.text(0, descriptor_y-descriptor_vspacing*1, "Two-point: " + str(two_point_fit), transform=ax.transAxes, fontsize=descriptor_fontsize)
            
                # add initial rate text
                ax.text(0, descriptor_y-descriptor_vspacing*2, "$v_i$: " + str(round(vi, 5)), transform=ax.transAxes, fontsize=descriptor_fontsize)

                # add exponential fit text
                ax.text(0, descriptor_y-descriptor_vspacing*3, "$k_{obs}$: " + str(round(k_obs, 3)), transform=ax.transAxes, fontsize=descriptor_fontsize)
                ax.text(0, descriptor_y-descriptor_vspacing*4, "$A$: " + str(round(A, 3)), transform=ax.transAxes, fontsize=descriptor_fontsize)
                ax.text(0, descriptor_y-descriptor_vspacing*5, "$R^2$: " + str(round(exp_R2, 3)), transform=ax.transAxes, fontsize=descriptor_fontsize)
                # show kcat/KM in scientific notation
                ax.text(0, descriptor_y-descriptor_vspacing*6, "Exp $kcat/KM$: " + str('{:.2e}'.format(kobs_kcat_over_KM)), transform=ax.transAxes, fontsize=descriptor_fontsize)

            # add title
            title_string = str(conc) + ' uM ' + substrate_name
            # insert \n to break up long titles
            if len(title_string) > 10:
                # split title string into two lines
                title_string = title_string.split('uM')
                title_string = title_string[0] + 'uM\n' + title_string[1]
            ax.set_title(title_string)
            ax.set_box_aspect(1)



# plot heatmap in PDF output file
def heatmap(data, ax=None, norm=None,
            cbar_kw=None, cbarlabel="", display_cbar=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    # if ax is None:
    #     ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, aspect='equal', **kwargs)

    # Create colorbar
    if display_cbar == True:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        cbar = ax.figure.colorbar(im, ax=ax, norm=norm, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# ==========================================================================================
# COMPLETE SUMMARY PLOTTING FUNCTION FOR MICHAELIS-MENTEN EXPERIMENTS
# ==========================================================================================

# main function to plot progress curves, heatmap, and chamber descriptors for experiment
def plot_chip_summary(squeeze_mm, standard_type, sq_merged, squeeze_standards, filter_dictionary, exclude_2point_MM, squeeze_kinetics, button_stamps, device_columns, device_rows, export_path_root, 
                      experimental_day, experiment_name, substrate, pbp_conc=None, starting_chamber=None, exclude_concs=[], MM_fit_r2_threshold=0.9):

    # create export directory
    newpath = export_path_root + '/PDF/pages/'

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ## defining figure ============================================================
    # initialize figure
    fig = plt.figure(figsize=(15, 10)) # size of print area of A4 page is 7 by 9.5

    # get today's date as MMM DD, YYYY
    from datetime import datetime
    today = datetime.today().strftime('%b %d, %Y')

    # increase the spacing between the subplots
    fig.subplots_adjust(wspace=1.5, hspace=1)
    fig.suptitle(' '.join([experimental_day, experiment_name, substrate, 'Summary']) + '\nGenerated on ' + today, fontsize=16, y=1)

    ## defining subplots ============================================================
    ax_conc_heatmap = plt.subplot2grid((5, 8), (0, 0), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_kinetic_heatmap = plt.subplot2grid((5, 8), (0, 2), rowspan=2, colspan=2) # kinetic heatmap
    ax_egfp_hist = plt.subplot2grid((5, 8), (0, 4), rowspan=2, colspan=2) # eGFP histogram
    ax_filter_table = plt.subplot2grid((5, 8), (2, 0), rowspan=1, colspan=2) # table of filter parameters
    ax_replicate_scatter_MM_KM = plt.subplot2grid((5, 8), (3, 0), rowspan=2, colspan=2) # replicate scatter of KM
    ax_replicate_scatter_MM_kcat = plt.subplot2grid((5, 8), (3, 2), rowspan=2, colspan=2) # replicate scatter of kcat
    ax_replicate_scatter_MM_kcatKM = plt.subplot2grid((5, 8), (3, 4), rowspan=2, colspan=2) # replicate scatter of kcat/KM
    ax_replicate_scatter_exp_kcatKM = plt.subplot2grid((5, 8), (3, 6), rowspan=2, colspan=2) # replicate scatter of kcat/KM from kobs fit


    ############################################
    ########### Heatmap plotting ###############
    ############################################

    # fill NaN
    grid = squeeze_mm.fillna(0)

    # plot enzyme concentration heatmap ============================================================
    grid_EC = grid.pivot(columns='x', index='y', values='EnzymeConc')
    grid_EC = grid_EC[grid_EC.columns[::-1]] # flip horizontally
    grid_EC = np.array(grid_EC)
    display_cbar=True # only display cbar on last grid
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    im, cbar = heatmap(grid_EC,
                        cmap="plasma", 
                        cbarlabel="Enzyme Conc (nM)", 
                        display_cbar=display_cbar,
                        ax=ax_conc_heatmap,
                        norm=norm
                        )
    ax_conc_heatmap.set_xticks([])
    ax_conc_heatmap.set_yticks([])
    ax_conc_heatmap.set_title('Enzyme Concentration \n by chamber')

    # plot enz LBG (shrink large values to 6) ============================================================
    max_lbg = 4
    grid_lbg = grid.pivot(columns='x', index='y', values='local_bg_ratio')
    grid_lbg = grid_lbg[grid_lbg.columns[::-1]] # flip horizontally
    grid_lbg = np.array(grid_lbg)
    display_cbar = True # only display cbar on last grid
    grid_lbg[grid_lbg > max_lbg] = max_lbg

    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    im, cbar = heatmap(grid_lbg,
                        cmap="viridis", 
                        cbarlabel="Local BG Ratio", 
                        display_cbar=display_cbar,
                        norm=norm,
                        ax=ax_kinetic_heatmap,
                        )
    
    # set cbar ticks and labels
    cbar_ticks = np.linspace(0, max_lbg, max_lbg+1)
    cbar_tick_labels = [ str(int(x)) for x in cbar_ticks[:-1]] + [">"+str(max_lbg)]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_tick_labels)

    ax_kinetic_heatmap.set_xticks([])
    ax_kinetic_heatmap.set_yticks([])
    ax_kinetic_heatmap.set_title('Local BG Ratio \n by chamber')


    ## Histogram plotting ============================================================
    # remove blank chambers
    all_blanks = sq_merged[sq_merged['MutantID'] == 'BLANK'][[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')
    all_library = sq_merged[sq_merged['MutantID'] != 'BLANK'][[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')

    # plot histograms, with blank in light grey and library in light green. Add outlines
    ax_egfp_hist.hist(all_library['EnzymeConc'], bins=20, alpha=0.5, label='Library', color='green', edgecolor='black', linewidth=1.2)
    ax_egfp_hist.hist(all_blanks['EnzymeConc'], bins=20, alpha=0.5, label='Blank', color='grey', edgecolor='black', linewidth=1.2)

    # set plot 
    ax_egfp_hist.set_xlabel('eGFP Concentration (uM)')
    ax_egfp_hist.set_ylabel('Count')
    ax_egfp_hist.set_title('eGFP concentration \n in blank chambers')


    ############################################
    ######### Filter Table plotting ############
    ############################################
    # create filter table, including a label for each filter
    filter_table = pd.DataFrame.from_dict(filter_dictionary, orient='index', columns=['Value'])
    filter_table['Filter'] = filter_table.index
    filter_table = filter_table.reset_index(drop=True)
    filter_table = filter_table[['Filter', 'Value']]
    filter_table = filter_table.sort_values(by=['Value'], ascending=False)

    # plot filter table
    ax_filter_table.axis('off')
    ax_filter_table.axis('tight')
    ax_filter_table.table(cellText=filter_table.values, colLabels=filter_table.columns, loc='center')


    ############################################
    ######## Replicate Scatter Plots ###########
    ############################################
    # Filter out blanks, low Enz, low local background ratio, and egfp flag
    sq_merged_filtered = squeeze_mm[(squeeze_mm['MutantID'] != 'BLANK') 
                                     & (squeeze_mm['EnzymeConc'] > filter_dictionary['EnzymeConc'])
                                     & (squeeze_mm['local_bg_ratio'] > filter_dictionary['local_bg_ratio'])
                                     & (squeeze_mm['egfp_manual_flag'] == 0)
                                     & (squeeze_mm['MM_kcat_over_KM_fit_R2'] > MM_fit_r2_threshold)
                                     ]
    
    # remove MM fits with less than 3 points if filter is set to True
    if exclude_2point_MM == True:
        sq_merged_filtered = sq_merged_filtered[(squeeze_mm['MM_points_to_fit'] > 2)]
    
    # # only keep the two replicates from each MutantID with the highest MM_kcat_over_KM_fit_R2
    # sq_merged_filtered = sq_merged_filtered.sort_values(by=['MM_kcat_over_KM_fit_R2'], ascending=False)
    # sq_merged_filtered = sq_merged_filtered.groupby('MutantID').head(2)
    
    # Group by 'MutantID' and filter out those with less than two 'kcat' values
    filtered_groups_kcat = sq_merged_filtered[['MutantID', 'kcat_fit', 'Indices']].groupby('MutantID').filter(lambda x: len(x) >= 2).sort_values(by='MutantID')
    filtered_groups_KM = sq_merged_filtered[['MutantID', 'KM_fit', 'Indices']].groupby('MutantID').filter(lambda x: len(x) >= 2).sort_values(by='MutantID')
    filtered_groups_MM_kcatKM = sq_merged_filtered[['MutantID', 'MM_kcat_over_KM_fit', 'Indices']].groupby('MutantID').filter(lambda x: len(x) >= 2).sort_values(by='MutantID')
    filtered_groups_exp_kcatKM = sq_merged_filtered[['MutantID', 'kobs_kcat_over_KM', 'Indices']].groupby('MutantID').filter(lambda x: len(x) >= 2).sort_values(by='MutantID')

    # Create a helper column for replicate enumeration
    filtered_groups_kcat['replicate'] = filtered_groups_kcat.groupby('MutantID').cumcount() + 1
    filtered_groups_KM['replicate'] = filtered_groups_KM.groupby('MutantID').cumcount() + 1
    filtered_groups_MM_kcatKM['replicate'] = filtered_groups_MM_kcatKM.groupby('MutantID').cumcount() + 1
    filtered_groups_exp_kcatKM['replicate'] = filtered_groups_exp_kcatKM.groupby('MutantID').cumcount() + 1

    # Keep only the first two replicates for each MutantID
    filtered_groups_kcat = filtered_groups_kcat[filtered_groups_kcat['replicate'] <= 2]
    filtered_groups_KM = filtered_groups_KM[filtered_groups_KM['replicate'] <= 2]
    filtered_groups_MM_kcatKM = filtered_groups_MM_kcatKM[filtered_groups_MM_kcatKM['replicate'] <= 2]
    filtered_groups_exp_kcatKM = filtered_groups_exp_kcatKM[filtered_groups_exp_kcatKM['replicate'] <= 2]

    # Pivot the table to spread the replicates into separate columns
    pivot_df_kcat = filtered_groups_kcat.pivot(index='MutantID', columns=['replicate'], values=['Indices', 'kcat_fit']).reset_index()
    pivot_df_KM = filtered_groups_KM.pivot(index='MutantID', columns=['replicate'], values=['Indices', 'KM_fit']).reset_index()
    pivot_df_MM_kcatKM = filtered_groups_MM_kcatKM.pivot(index='MutantID', columns=['replicate'], values=['Indices', 'MM_kcat_over_KM_fit']).reset_index()
    pivot_df_exp_kcatKM = filtered_groups_exp_kcatKM.pivot(index='MutantID', columns=['replicate'], values=['Indices', 'kobs_kcat_over_KM']).reset_index()
    
    # remove multi-index if not empty
    if len(pivot_df_kcat) > 0:
        pivot_df_kcat.columns = ['MutantID', 'indices_1', 'indices_2', 'replicate_1', 'replicate_2']
        pivot_df_KM.columns = ['MutantID', 'indices_1', 'indices_2', 'replicate_1', 'replicate_2']
        pivot_df_MM_kcatKM.columns = ['MutantID', 'indices_1', 'indices_2', 'replicate_1', 'replicate_2']
        pivot_df_exp_kcatKM.columns = ['MutantID', 'indices_1', 'indices_2', 'replicate_1', 'replicate_2']

        # KM ============================================================
        ax_replicate_scatter_MM_KM.scatter(pivot_df_KM['replicate_1'], pivot_df_KM['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_MM_KM.plot([pivot_df_KM['replicate_1'].min(), pivot_df_KM['replicate_1'].max()], [pivot_df_KM['replicate_1'].min(), pivot_df_KM['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_MM_KM.set_xlabel('Replicate 1 ($uM$)')
        ax_replicate_scatter_MM_KM.set_ylabel('Replicate 2 ($uM$)')
        ax_replicate_scatter_MM_KM.set_title('$K_M$ Replicates')
        ax_replicate_scatter_MM_KM.set_xscale('log')
        ax_replicate_scatter_MM_KM.set_yscale('log')

        # kcat ============================================================
        distance_max = 0.8
        ax_replicate_scatter_MM_kcat.scatter(pivot_df_kcat['replicate_1'], pivot_df_kcat['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_MM_kcat.plot([pivot_df_kcat['replicate_1'].min(), pivot_df_kcat['replicate_1'].max()], [pivot_df_kcat['replicate_1'].min(), pivot_df_kcat['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_MM_kcat.set_xlabel('Replicate 1 ($s^{-1}$)')
        ax_replicate_scatter_MM_kcat.set_ylabel('Replicate 2 ($s^{-1}$)')
        ax_replicate_scatter_MM_kcat.set_title('$k_{cat}$ Replicates')
        ax_replicate_scatter_MM_kcat.set_xscale('log')
        ax_replicate_scatter_MM_kcat.set_yscale('log')

        # label outliers on this scatter plot
        labels = []
        texts = []
        for index, row in pivot_df_kcat.iterrows():
            x = row['replicate_1']
            y = row['replicate_2']
            mutant_id = '1:({})\n2:({})'.format(row['indices_1'], row['indices_2'])
            distance = 1 - min(x,y)/max(x,y)
            if distance > distance_max:
                outlier_label = mutant_id
                labels.append(outlier_label)
                # Adjust initial text placement to be closer to the point
                text = ax_replicate_scatter_MM_kcat.text(x, y, outlier_label, fontsize=8, ha='right', va='bottom')
                texts.append(text)

        if texts:
            adjust_text(texts,
                        arrowprops=dict(arrowstyle="->", color='gray', lw=1, shrinkA=5, shrinkB=5),  # Adjusted arrow length
                        ax=ax_replicate_scatter_MM_kcat,
                        force_points=0.2,  # Increased force
                        force_text=0.2,  # Increased force
                        expand_points=(1, 1),  # Increased area
                        expand_text=(1, 1),  # Increased area
                        autoalign='y',
                        avoid_points=True,  # Avoid overlap with points
                        add_objects=[ax_replicate_scatter_MM_kcat.get_lines()[0], # Include y=x line in avoidance  
                                     ],  
                        ) 
    
    
        # Michaelis-Menten kcat/KM ============================================================
        distance_max = 0.8
        ax_replicate_scatter_MM_kcatKM.scatter(pivot_df_MM_kcatKM['replicate_1'], pivot_df_MM_kcatKM['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_MM_kcatKM.plot([pivot_df_MM_kcatKM['replicate_1'].min(), pivot_df_MM_kcatKM['replicate_1'].max()], [pivot_df_MM_kcatKM['replicate_1'].min(), pivot_df_MM_kcatKM['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_MM_kcatKM.set_xlabel('Replicate 1 ($M^{-1}s^{-1}$)')
        ax_replicate_scatter_MM_kcatKM.set_ylabel('Replicate 2 ($M^{-1}s^{-1}$)')
        ax_replicate_scatter_MM_kcatKM.set_title('MM $k_{cat}/K_M$ Replicates\nOutliers:>%s%%' % str(int(distance_max * 100)))
        ax_replicate_scatter_MM_kcatKM.set_xscale('log')
        ax_replicate_scatter_MM_kcatKM.set_yscale('log')

        # label outliers on this scatter plot
        labels = []
        texts = []
        for index, row in pivot_df_MM_kcatKM.iterrows():
            x = row['replicate_1']
            y = row['replicate_2']
            mutant_id = '1:({})\n2:({})'.format(row['indices_1'], row['indices_2'])
            distance = 1 - min(x,y)/max(x,y)
            if distance > distance_max:
                outlier_label = mutant_id
                labels.append(outlier_label)
                # Improved initial text placement and alignment
                text = ax_replicate_scatter_MM_kcatKM.text(x, y, outlier_label, fontsize=8, ha='right', va='top')
                texts.append(text)

        if texts:
            adjust_text(texts,
                        arrowprops=dict(arrowstyle="->", color='gray', lw=1, shrinkA=5, shrinkB=5),  # Adjusted arrow length
                        ax=ax_replicate_scatter_MM_kcatKM,
                        force_points=0.2,  # Increased force
                        force_text=0.2,  # Increased force
                        expand_points=(1, 1),  # Increased area
                        expand_text=(1, 1),  # Increased area
                        autoalign='y',
                        avoid_points=True,  # Avoid overlap with points
                        add_objects=[ax_replicate_scatter_MM_kcatKM.get_lines()[0]],  # Include y=x line in avoidance
                        # only_move='y', # Restrict movement to y-axis if suitable
                        ) 

        # # Exponential kcat/KM ============================================================
        # ax_replicate_scatter_exp_kcatKM.scatter(pivot_df_exp_kcatKM['replicate_1'], pivot_df_exp_kcatKM['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        # ax_replicate_scatter_exp_kcatKM.plot([pivot_df_exp_kcatKM['replicate_1'].min(), pivot_df_exp_kcatKM['replicate_1'].max()], [pivot_df_exp_kcatKM['replicate_1'].min(), pivot_df_exp_kcatKM['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        # ax_replicate_scatter_exp_kcatKM.set_xlabel('Replicate 1 ($M^{-1}s^{-1}$)')
        # ax_replicate_scatter_exp_kcatKM.set_ylabel('Replicate 2 ($M^{-1}s^{-1}$)')
        # ax_replicate_scatter_exp_kcatKM.set_title('Exponential $k_{cat}/K_M$ Replicates\nOutliers:>%s%%' % str(int(distance_max * 100)))
        # ax_replicate_scatter_exp_kcatKM.set_xscale('log')
        # ax_replicate_scatter_exp_kcatKM.set_yscale('log')
        
        # # label outliers on this scatter plot
        # labels = []
        # texts = []
        # for index, row in pivot_df_exp_kcatKM.iterrows():
        #     # get x and y values
        #     x = row['replicate_1']
        #     y = row['replicate_2']

        #     # get mutant id
        #     mutant_id = row['MutantID']

        #     # get distance from y=x line
        #     distance = 1 - min(x,y)/max(x,y)

        #     # if distance is greater than 100, label the point
        #     if distance > distance_max:
        #         outlier_label = mutant_id.split('_')[0]
        #         labels.append(outlier_label)
        #         texts.append(ax_replicate_scatter_exp_kcatKM.text(x, y, outlier_label, fontsize=8))
        # # adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), ax=ax_replicate_scatter_exp_kcatKM, 
        # #             force_points=0.1, force_text=0.1, expand_points=(2, 2), expand_text=(1.2, 1.2))


    ############################################
    ############## Save figure #################
    ############################################

    plt.savefig(newpath + '00_Chip_Summary.pdf')
    plt.show()

    # close figure
    plt.close()

    ## Initialize v_isothermal ============================================================
    def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
    v_isotherm = np.vectorize(isotherm)


    ############################################
    ######### Chamber-wise plotting ############
    ############################################

    # initialize starting chamber
    start_x = 1
    start_y = 1

    # if user specifies starting chamber, start from there
    if starting_chamber is not None:
        start_x = starting_chamber[0]
        start_y = starting_chamber[1]

    # initialize tqdm
    with tqdm(total = (device_columns*device_rows) - ((start_x-1)*device_rows + (start_y-1))) as pbar:

        # Plot Chamber-wise Summaries
        for x in range(start_x, device_columns + 1, 1):
            for y in range(start_y, device_rows + 1, 1):
                
                ## initialize data dfs ============================================================
                export_kinetic_df = sq_merged[sq_merged['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                export_standards_df = squeeze_standards[squeeze_standards['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                export_mm_df = squeeze_mm[squeeze_mm['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                
                
                ## defining figure ============================================================
                # initialize figure
                fig = plt.figure(figsize=(10, 12))

                # if there are more than 4 substrate concs, decrease height spacing
                if len(set(export_kinetic_df['substrate_conc_uM'])) > 4:
                    pdf_grid_hspacing = 0
                else:
                    pdf_grid_hspacing = 0.75

                # increase the spacing between the subplots
                fig.subplots_adjust(wspace=0.5, hspace=pdf_grid_hspacing)

                ## defining subplots ============================================================
                ax_image = plt.subplot2grid((10, 12), (0, 0), colspan=2) # chamber image
                ax_image.set_rasterization_zorder(0)
                
                # define concentration series
                concs = set(export_kinetic_df['substrate_conc_uM'])

                # initialize progress curve subplot2grid objects
                for n, conc in enumerate(sorted(concs)):
                    if n == 0:
                        globals()['ax_progress_curve_' + str(n)] = plt.subplot2grid((5, len(concs)), (1, n), colspan=1, rowspan=1) # progress curves
                        globals()['ax_progress_curve_' + str(n)].set_rasterization_zorder(0)
                    else:
                        globals()['ax_progress_curve_' + str(n)] = plt.subplot2grid((5, len(concs)), (1, n), colspan=1, rowspan=1, sharey=ax_progress_curve_0) # progress curves
                        globals()['ax_progress_curve_' + str(n)].set_rasterization_zorder(0)

                # initialize progress curve zoom subplot2grid objects (in row below progress curves)
                for n, conc in enumerate(sorted(concs)):
                    if n == 0:
                        globals()['ax_progress_curve_zoom_' + str(n)] = plt.subplot2grid((5, len(concs)), (2, n), colspan=1, rowspan=1)
                        globals()['ax_progress_curve_zoom_' + str(n)].set_rasterization_zorder(0)
                    else:
                        globals()['ax_progress_curve_zoom_' + str(n)] = plt.subplot2grid((5, len(concs)), (2, n), colspan=1, rowspan=1, sharey=ax_progress_curve_zoom_0) 
                        globals()['ax_progress_curve_zoom_' + str(n)].set_rasterization_zorder(0)
            
                # define table, mm curve, and pbp std curve subplot2grid objects
                ax_table = plt.subplot2grid((8, 10), (0, 3), colspan=5) # table
                ax_mm_curve = plt.subplot2grid((8, 10), (6, 0), colspan=3, rowspan=3)
                ax_std_curve = plt.subplot2grid((8, 10), (6, 7), colspan=3, rowspan=3)
                ax_table.set_rasterization_zorder(0)
                ax_mm_curve.set_rasterization_zorder(0)
                ax_std_curve.set_rasterization_zorder(0)


                ## image plotting ============================================================
                img_idx = (x, y)
                ax_image.imshow(button_stamps[img_idx], cmap='gray', vmin=0, vmax=10**4.3)


                ## std curve fit  ============================================================
                # if export_standards_df is empty, skip this step
                if export_standards_df.empty == False:
                    xdata, ydata = export_standards_df.standard_concentration_uM.values[0], export_standards_df.standard_median_intensities.values[0]
                    chamber_popt = export_standards_df.standard_popt.values[0]
                else:
                    xdata, ydata = np.nan, np.nan
                    chamber_popt = np.nan

                # define the interpolation
                interp_xdata = np.linspace(-np.min(xdata), np.max(xdata), num=100, endpoint=False)

                # if standard type is PBP, plot
                if standard_type.upper() == 'PBP':
                    if type(chamber_popt) != float: # if fit is successful
                        ax_std_curve.plot(interp_xdata, v_isotherm(interp_xdata, *chamber_popt), color='g', linestyle='dashed', label='Isotherm Fit')
                    ax_std_curve.scatter(xdata, ydata, color='b', s=50, alpha=0.4, label='data')
                    ax_std_curve.set_ylabel('Intensity')
                    ax_std_curve.set_xlabel('Pi Concentration (uM)')
                    ax_std_curve.set_title('PBP Standard Curve')
                    ax_std_curve.legend(loc='lower right', shadow=True, title='PBP Conc = %s uM' % pbp_conc)

                # if standard type is linear, plot
                elif standard_type == 'linear':
                    if type(chamber_popt) != float:
                        def linear(x, a, b): return a*x + b 
                        v_linear = np.vectorize(linear)
                        ax_std_curve.plot(xdata, v_linear(xdata, *chamber_popt), 'r-', label='Linear Fit')
                        ax_std_curve.plot(interp_xdata, v_linear(interp_xdata, *chamber_popt), color='g', linestyle='dashed', label='interpolation')
                    ax_std_curve.scatter(xdata, ydata, color='b', s=50, alpha=0.4, label='data')
                    ax_std_curve.set_ylabel('Intensity')
                    ax_std_curve.set_xlabel('Product Concentration (uM)')
                    ax_std_curve.set_title('Standard Curve')
                    ax_std_curve.legend(loc='lower right', shadow=True)

                ############################################################################################################
                ## Michaelis-Menten curve plotting ============================================================
                ############################################################################################################
                # if df is not empty, plot
                if export_mm_df.empty == False:
                    xdata = export_mm_df.iloc[0].substrate_concs
                    ydata = export_mm_df.iloc[0].initial_rates
                    MutantID = export_mm_df.iloc[0].MutantID
                    Indices = export_mm_df.iloc[0].Indices
                    KM_fit = export_mm_df.iloc[0].KM_fit
                    vmax_fit = export_mm_df.iloc[0].vmax_fit
                    r2 = export_mm_df.iloc[0].MM_kcat_over_KM_fit_R2
                    mm_excluded_concs = export_mm_df.iloc[0].exclude_concs

                    # subset xdata for points that were excluded from fit, based on mm_excluded_concs
                    if len(mm_excluded_concs) == 0:
                        mm_excluded_concs = []
                    else:
                        mm_excluded_concs = ast.literal_eval(mm_excluded_concs)
                    excluded_xdata = [x for x in xdata if x in mm_excluded_concs]
                    excluded_ydata = [y for x, y in zip(xdata, ydata) if x in mm_excluded_concs]
                    included_xdata = [x for x in xdata if x not in mm_excluded_concs]
                    included_ydata = [y for x, y in zip(xdata, ydata) if x not in mm_excluded_concs]

                    # plot points and curve fit
                    t = np.arange(0, max(xdata), 0.2) # x range for plotting
                    fit_ydata = v_mm_func(t, *[KM_fit, vmax_fit]) # get y values from fit function
                    ax_mm_curve.plot(included_xdata, included_ydata, 'o', color='b', label='Fit Data') # plot included points
                    ax_mm_curve.plot(excluded_xdata, excluded_ydata, 'x', color='grey', label='Excluded (n=%s)' % len(excluded_xdata)) # plot excluded points
                    ax_mm_curve.axhline(vmax_fit, label='$v_{max}$') # plot vmax line
                    ax_mm_curve.plot(t, fit_ydata, 'g--', label='MM Fit \n$R^2$: %.3f' % r2) # plot michaelis-menten curve fit

                    # add labels, title, and legend
                    ax_mm_curve.set_ylabel('$v_i$')
                    ax_mm_curve.set_xlabel('AcP Concentration (uM)')
                    ax_mm_curve.set_title('Michaelis-Menten Plot')
                    if max(ydata) > 0:
                        miny = -min([y for y in ydata if y > 0])/20 # set min to 1/20 of lowest y value at or above 0
                        maxy = max(ydata)*1.2
                        ax_mm_curve.set_ylim([miny, maxy]) # set y axis limits, with some buffer
                    ax_mm_curve.set_xlim([-max(xdata)/20, max(xdata)*1.2]) # set x axis limits, with some buffer
                    ax_mm_curve.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                                       shadow=True, fancybox=True, title='Michaelis-Menten Fit')

                else:
                    xdata = [0,0,0,0]
                    ydata = [0,0,0,0]
                    MutantID = np.nan
                    Indices = np.nan
                    KM_fit = np.nan
                    vmax_fit = np.nan


                ############################################################################################################
                ## progress curve plotting ============================================================
                ############################################################################################################
                # find local background indices and store data in new df
                local_background_indices = get_local_bg_indices(x=x, y=y, device_rows=device_rows) # stores indices of local background chambers in a list
                local_background_df = pd.DataFrame([])
                for i in local_background_indices:
                    data = sq_merged[sq_merged['Indices'] == f"{i[0]:02d}" + ',' + f"{i[1]:02d}"]
                    
                    # add data to local_background_df
                    local_background_df = pd.concat([local_background_df, data])

                ### plot first row of curves ###
                for ax, conc in enumerate(sorted(concs)):
                    plot_progress_curve(export_kinetic_df, conc=conc, ax=globals()['ax_progress_curve_' + str(ax)] , fit_descriptors=False, kwargs_for_scatter={"s": 15, "c": 'blue'}, kwargs_for_line={"c": 'blue'}, pbp_conc=pbp_conc, substrate_name=substrate)
                    plot_progress_curve(local_background_df, conc=conc, ax=globals()['ax_progress_curve_' + str(ax)], kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'}, pbp_conc=pbp_conc, fit_descriptors=False, substrate_name=substrate)

                    # set ylabel on first (left-most) plot
                    if ax == 0:
                        globals()['ax_progress_curve_' + str(ax)].set_ylabel('Product\nConc (uM)')
                    else:
                        plt.setp(globals()['ax_progress_curve_' + str(ax)].get_yticklabels(), visible=False)

                    # set xlabels on all plots
                    globals()['ax_progress_curve_' + str(ax)].set_xlabel('Time (s)')
                    
                # set ylims
                # get all product concentrations
                all_product_concs = []
                for index, row in export_kinetic_df.iterrows():
                    all_product_concs.append(row['kinetic_product_concentration_uM'])

                # flatten list of all product concentrations
                if len(all_product_concs) == 0:
                    all_product_concs = [0]
                elif type(all_product_concs[0]) == list:
                    all_product_concs = [item for sublist in all_product_concs for item in sublist]
                elif type(all_product_concs[0]) == float:
                    all_product_concs = [0]

                # now set ylims for each ax
                for ax, conc in enumerate(sorted(concs)):
                    globals()['ax_progress_curve_' + str(ax)].set_ylim(min(all_product_concs), max(all_product_concs)*1.1)

                ### add second row of progress curves, zoomed in ###
                # set max_x for zoomed in plots
                max_x = 700 # seconds

                # plot curves
                for ax, conc in enumerate(sorted(concs)):
                    plot_progress_curve(export_kinetic_df, conc=conc, ax=globals()['ax_progress_curve_zoom_' + str(ax)] , fit_descriptors=True, kwargs_for_scatter={"s": 15, "c": 'blue'}, kwargs_for_line={"c": 'blue'}, pbp_conc=pbp_conc, substrate_name=substrate)
                    plot_progress_curve(local_background_df, conc=conc, ax=globals()['ax_progress_curve_zoom_' + str(ax)], kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'}, pbp_conc=pbp_conc, fit_descriptors=False, substrate_name=substrate)

                    # # set xlabels on all plots
                    # globals()['ax_progress_curve_zoom_' + str(ax)].set_xlabel('Time (s)')

                    # set ylabel on first (left-most) plot
                    if ax == 0:
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_ylabel('Product\nConc (uM)')
                    else:
                        plt.setp(globals()['ax_progress_curve_zoom_' + str(ax)].get_yticklabels(), visible=False)

                    # remove title
                    if ax == 0:
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_title('')
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_title('Zoomed at\n$t_{max}$ = %s s' % max_x, fontsize=8, loc='left')
                    else:
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_title('')


                # set xlim and ylim
                for ax, conc in enumerate(sorted(concs)):
                    globals()['ax_progress_curve_zoom_' + str(ax)].set_xlim(-30, max_x)

                ## set y upper limit to pbp_conc
                # get index of closest timepoint to max_x from export_kinetic_df
                if export_kinetic_df.empty == False:
                    time_list = export_kinetic_df['time_s'].iloc[0]
                    max_x_index = min(range(len(time_list)), key=lambda i: abs(time_list[i]-max_x))

                    # get max product concentration before max_x
                    all_product_concs_zoom = []
                    for index, row in export_kinetic_df.iterrows():
                        if type(row['kinetic_product_concentration_uM']) == list:
                            all_product_concs_zoom.append(row['kinetic_product_concentration_uM'][:max_x_index])
                        else:
                            all_product_concs_zoom.append(row['kinetic_product_concentration_uM'])
                            # remove nans
                            all_product_concs_zoom = [x for x in all_product_concs_zoom if str(x) != 'nan']
                else:
                    all_product_concs_zoom = []

                # if all_product_concs_zoom is not empty, set ylim
                if len(all_product_concs_zoom) != 0:
                    # flatten list of all product concentrations
                    all_product_concs_zoom = [item for sublist in all_product_concs_zoom for item in sublist]

                    for ax, conc in enumerate(sorted(concs)):
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_ylim(min(all_product_concs), 1.2*max(all_product_concs_zoom))
                else:
                    print('all_product_concs_zoom is empty')
                    


                ## table plotting ============================================================
                if export_mm_df.empty == False:
                    table_df = export_mm_df[['Indices', 'EnzymeConc', 'egfp_manual_flag', 'local_bg_ratio', 'kcat_fit', 'KM_fit', 'MM_kcat_over_KM_fit', 'kobs_kcat_over_KM']]
                    table_df.apply(pd.to_numeric, errors='ignore', downcast='float')
                    table_df = table_df.round(decimals=2)
                    table_df['FilteringOutcome'] = 'Passed' # filtering outcome column
                    filtering_outcome = 'Passed' # default filtering outcome

                    # if kcat/KM is greater than 10^6, convert to scientific notation
                    if table_df['MM_kcat_over_KM_fit'].max() > 10**6:
                        table_df['MM_kcat_over_KM_fit'] = table_df['MM_kcat_over_KM_fit'].apply(lambda x: '%.2E' % x)

                    params_table = ax_table.table(cellText=table_df.values, loc='center', colLabels=['Idx', '[E] (nM)', 'eGFP Flag', 'LBGR', '$k_{cat}$', '$K_M$ (uM)', 'MM $k_{cat}/K_M$ ($M^{-1} s^{-1}$)', '$k_{obs}$ $k_{cat}/K_M$ ($M^{-1} s^{-1}$)', 'Filtering'])
                    params_table.auto_set_font_size(False)
                    params_table.set_fontsize(8)
                    params_table.scale(0.1, 1.5)
                    params_table.auto_set_column_width(col=list(range(len(table_df.columns))))
                    ax_table.set_title(export_kinetic_df.iloc[0].MutantID)
                    ax_table.set_axis_off()
                    ax_table.set_frame_on(False)

                    ## table filtering
                    # if value is below the filter threshold from the filter_dict, set the corresponding column color to light red
                    for key, value in filter_dictionary.items():
                        for row in range(len(table_df)):
                            if (type(value) == int) or (type(value) == float):
                                if table_df.iloc[row][key] < value:
                                    params_table[(row+1, table_df.columns.get_loc(key))].set_facecolor('#ffcccc')
                                    filtering_outcome = 'Failed'
                            else:
                                if table_df.iloc[row][key] == value:
                                    params_table[(row+1, table_df.columns.get_loc(key))].set_facecolor('#ffcccc')
                                    filtering_outcome = 'Failed'

                    # update cell in params table with filtering outcome and change color
                    if filtering_outcome == 'Failed':
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].get_text().set_text(filtering_outcome)
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].set_facecolor('#ffcccc')
                    else:
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].get_text().set_text(filtering_outcome)
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].set_facecolor('#ccffcc')

                # update progress bar
                pbar.update(1)
                pbar.set_description("Processing %s" % str((x, y)))

                # set facecolor to white
                fig.patch.set_facecolor('white')

                # save figure
                fig.savefig(newpath + str(Indices) + '.pdf', dpi=200)

                # close figure
                plt.close(fig)

        pbar.set_description("Export complete")


# ==========================================================================================
# COMPLETE SUMMARY PLOTTING FUNCTION FOR INHIBITION EXPERIMENTS
# ==========================================================================================

# main function to plot progress curves, heatmap, and chamber descriptors for experiment
def plot_chip_summary_inhibition(Inhibitor, standard_type, sq_merged, squeeze_standards, filter_dictionary, squeeze_kinetics, button_stamps, device_columns, device_rows, export_path_root, 
                      experimental_day, experiment_name, substrate, pbp_conc=None, starting_chamber=None, exclude_concs=[]):

    # create export directory
    newpath = export_path_root + '/PDF/pages/'

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ## defining figure ============================================================
    # initialize figure
    fig = plt.figure(figsize=(15, 10)) # size of print area of A4 page is 7 by 9.5

    # increase the spacing between the subplots
    fig.subplots_adjust(wspace=1.5, hspace=1)
    fig.suptitle(' '.join([experimental_day, experiment_name, substrate, 'Summary']))

    ## defining subplots ============================================================
    ax_conc_heatmap = plt.subplot2grid((5, 8), (0, 0), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_kinetic_heatmap = plt.subplot2grid((5, 8), (0, 2), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_egfp_hist = plt.subplot2grid((5, 8), (0, 4), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_filter_table = plt.subplot2grid((5, 8), (2, 0), rowspan=1, colspan=2) # enzyme concentration heatmap
    ax_replicate_scatter_Hillslope = plt.subplot2grid((5, 8), (3, 0), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_replicate_scatter_ic50 = plt.subplot2grid((5, 8), (3, 2), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_replicate_scatter_Hillbottom = plt.subplot2grid((5, 8), (3, 4), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_replicate_scatter_Hilltop = plt.subplot2grid((5, 8), (3, 6), rowspan=2, colspan=2, sharex=ax_replicate_scatter_Hillbottom, sharey=ax_replicate_scatter_Hillbottom) # enzyme concentration heatmap

    ## Heatmap plotting ============================================================
    # fill NaN
    grid = sq_merged.fillna(0).drop_duplicates(subset=['Indices'], keep='first')

    # plot enz conc
    grid_EC = grid.pivot('x', 'y', 'EnzymeConc').T
    grid_EC = grid_EC[grid_EC.columns[::-1]] # flip horizontally
    grid_EC = np.array(grid_EC)
    display_cbar=True # only display cbar on last grid
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    im, cbar = heatmap(grid_EC,
                        cmap="plasma", 
                        cbarlabel="Enzyme Conc (nM)", 
                        display_cbar=display_cbar,
                        ax=ax_conc_heatmap,
                        norm=norm
                        )
    ax_conc_heatmap.set_xticks([])
    ax_conc_heatmap.set_yticks([])
    ax_conc_heatmap.set_title('Enzyme Concentration \n by chamber')

    # plot enz LBG (shrink large values to 6)
    max_lbg = 4
    grid_lbg = grid.pivot('x', 'y', 'local_bg_ratio').T
    grid_lbg = grid_lbg[grid_lbg.columns[::-1]] # flip horizontally
    grid_lbg = np.array(grid_lbg)
    display_cbar = True # only display cbar on last grid
    grid_lbg[grid_lbg > max_lbg] = max_lbg

    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    im, cbar = heatmap(grid_lbg,
                        cmap="viridis", 
                        cbarlabel="Local BG Ratio", 
                        display_cbar=display_cbar,
                        norm=norm,
                        ax=ax_kinetic_heatmap,
                        )
    
    # get colorbar tick labels
    cbar_ticks = cbar.get_ticks()
    cbar_tick_labels = []
    for tick in cbar_ticks:
        if tick == max_lbg:
            cbar_tick_labels.append('>' + str(max_lbg))
        else:
            cbar_tick_labels.append(str(int(tick)))

    # set colorbar tick labels
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_tick_labels)

    ax_kinetic_heatmap.set_xticks([])
    ax_kinetic_heatmap.set_yticks([])
    ax_kinetic_heatmap.set_title('Local BG Ratio \n by chamber')


    ## Histogram plotting ============================================================
    # remove blank chambers
    all_blanks = sq_merged[sq_merged['MutantID'] == 'BLANK'][[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')
    all_library = sq_merged[sq_merged['MutantID'] != 'BLANK'][[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')

    # plot histograms, with blank in light grey and library in light green. Add outlines
    ax_egfp_hist.hist(all_library['EnzymeConc'], bins=20, alpha=0.5, label='Library', color='green', edgecolor='black', linewidth=1.2)
    ax_egfp_hist.hist(all_blanks['EnzymeConc'], bins=20, alpha=0.5, label='Blank', color='grey', edgecolor='black', linewidth=1.2)

    # set plot 
    ax_egfp_hist.set_xlabel('eGFP Concentration (uM)')
    ax_egfp_hist.set_ylabel('Count')
    ax_egfp_hist.set_title('eGFP concentration \n in blank chambers')


    ## Filter Table plotting ============================================================
    # create filter table, including a label for each filter
    filter_table = pd.DataFrame.from_dict(filter_dictionary, orient='index', columns=['Value'])
    filter_table['Filter'] = filter_table.index
    filter_table = filter_table.reset_index(drop=True)
    filter_table = filter_table[['Filter', 'Value']]
    filter_table = filter_table.sort_values(by=['Value'], ascending=False)

    # plot filter table
    ax_filter_table.axis('off')
    ax_filter_table.axis('tight')
    ax_filter_table.table(cellText=filter_table.values, colLabels=filter_table.columns, loc='center')


    ## Plot replicate scatter plots ============================================================
    # Filter out blanks, low Enz, low local background ratio, and egfp flag
    sq_merged_filtered = sq_merged[(sq_merged['MutantID'] != 'BLANK') & 
                                     (sq_merged['EnzymeConc'] > filter_dictionary['EnzymeConc']) & 
                                     (sq_merged['local_bg_ratio'] > filter_dictionary['local_bg_ratio'])
                                     ]
    
    # only keep the two replicates from each MutantID with the highest MM_kcat_over_KM_fit_R2
    # sq_merged_filtered = sq_merged_filtered.sort_values(by=['MM_kcat_over_KM_fit_R2'], ascending=False)
    sq_merged_filtered = sq_merged_filtered.groupby('MutantID').head(2)
    sq_merged_filtered = sq_merged_filtered.drop_duplicates(subset=['MutantID', 'Indices'], keep='first')

    # generate pivot table for replicate scatter plots
    sq_merged_filtered_ic50 = sq_merged_filtered.pivot(index='MutantID', columns='Indices', values='HillCurve_ic50').reset_index()
    sq_merged_filtered_Hillslope = sq_merged_filtered.pivot(index='MutantID', columns='Indices', values='HillCurve_hill_slope').reset_index()
    sq_merged_filtered_Hillbottom = sq_merged_filtered.pivot(index='MutantID', columns='Indices', values='HillCurve_bottom').reset_index()
    sq_merged_filtered_Hilltop = sq_merged_filtered.pivot(index='MutantID', columns='Indices', values='HillCurve_top').reset_index()
    sq_merged_filtered_ic50.columns.name = None # remove the names of the columns
    sq_merged_filtered_Hillslope.columns.name = None # remove the names of the columns
    sq_merged_filtered_Hillbottom.columns.name = None # remove the names of the columns
    sq_merged_filtered_Hilltop.columns.name = None # remove the names of the columns

    # compress pivot table to only include two replicates
    compressed_ic50 = sq_merged_filtered_ic50.set_index('MutantID').stack().groupby(level=0).apply(lambda x: pd.Series(x.values)).unstack().reset_index()
    compressed_Hillslope = sq_merged_filtered_Hillslope.set_index('MutantID').stack().groupby(level=0).apply(lambda x: pd.Series(x.values)).unstack().reset_index()
    compressed_Hillbottom = sq_merged_filtered_Hillbottom.set_index('MutantID').stack().groupby(level=0).apply(lambda x: pd.Series(x.values)).unstack().reset_index()
    compressed_Hilltop = sq_merged_filtered_Hilltop.set_index('MutantID').stack().groupby(level=0).apply(lambda x: pd.Series(x.values)).unstack().reset_index()

    # rename columns
    new_columns_ic50 = ['MutantID'] + [f'replicate_{i+1}' for i in range(len(compressed_ic50.columns)-1)] # Create new column names
    new_columns_Hillslope = ['MutantID'] + [f'replicate_{i+1}' for i in range(len(compressed_Hillslope.columns)-1)] # Create new column names
    new_columns_Hillbottom = ['MutantID'] + [f'replicate_{i+1}' for i in range(len(compressed_Hillbottom.columns)-1)] # Create new column names
    new_columns_Hilltop = ['MutantID'] + [f'replicate_{i+1}' for i in range(len(compressed_Hilltop.columns)-1)] # Create new column names

    # rename columns
    compressed_ic50.columns = new_columns_ic50 # Rename the columns
    compressed_Hillslope.columns = new_columns_Hillslope # Rename the columns
    compressed_Hillbottom.columns = new_columns_Hillbottom # Rename the columns
    compressed_Hilltop.columns = new_columns_Hilltop # Rename the columns
    
    # only keep the first four replicates of each measurement for plotting
    keep_reps = 4
    compressed_ic50 = compressed_ic50.iloc[:, :keep_reps+1] # remove all columns after the fourth replicate
    compressed_Hillslope = compressed_Hillslope.iloc[:, :keep_reps+1] # remove all columns after the fourth replicate
    compressed_Hillbottom = compressed_Hillbottom.iloc[:, :keep_reps+1] # remove all columns after the fourth replicate
    compressed_Hilltop = compressed_Hilltop.iloc[:, :keep_reps+1] # remove all columns after the fourth replicate

    try:
        ax_replicate_scatter_ic50.scatter(compressed_ic50['replicate_1'], compressed_ic50['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_ic50.plot([compressed_ic50['replicate_1'].min(), compressed_ic50['replicate_1'].max()], [compressed_ic50['replicate_1'].min(), compressed_ic50['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_ic50.set_xlabel('Replicate 1 ($s^{-1}$)')
        ax_replicate_scatter_ic50.set_ylabel('Replicate 2 ($s^{-1}$)')
        ax_replicate_scatter_ic50.set_title('$k_{cat}$ Replicates')
        ax_replicate_scatter_ic50.set_xscale('log')
        ax_replicate_scatter_ic50.set_yscale('log')
    except:
        pass
    
    try:
        ax_replicate_scatter_Hillslope.scatter(compressed_Hillslope['replicate_1'], compressed_Hillslope['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_Hillslope.plot([compressed_Hillslope['replicate_1'].min(), compressed_Hillslope['replicate_1'].max()], [compressed_Hillslope['replicate_1'].min(), compressed_Hillslope['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_Hillslope.set_xlabel('Replicate 1 ($uM$)')
        ax_replicate_scatter_Hillslope.set_ylabel('Replicate 2 ($uM$)')
        ax_replicate_scatter_Hillslope.set_title('$K_M$ Replicates')
        ax_replicate_scatter_Hillslope.set_xscale('log')
        ax_replicate_scatter_Hillslope.set_yscale('log')
    except:
        pass    

    try:
        ax_replicate_scatter_Hillbottom.scatter(compressed_Hillbottom['replicate_1'], compressed_Hillbottom['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_Hillbottom.plot([compressed_Hillbottom['replicate_1'].min(), compressed_Hillbottom['replicate_1'].max()], [compressed_Hillbottom['replicate_1'].min(), compressed_Hillbottom['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_Hillbottom.set_xlabel('Replicate 1 ($M^{-1}s^{-1}$)')
        ax_replicate_scatter_Hillbottom.set_ylabel('Replicate 2 ($M^{-1}s^{-1}$)')
        ax_replicate_scatter_Hillbottom.set_title('MM $k_{cat}/K_M$ Replicates')
        ax_replicate_scatter_Hillbottom.set_xscale('log')
        ax_replicate_scatter_Hillbottom.set_yscale('log')
    except:
        pass

    try:
        ax_replicate_scatter_Hilltop.scatter(compressed_Hilltop['replicate_1'], compressed_Hilltop['replicate_2'], color='blue', label='Replicate 1 vs. Replicate 2')
        ax_replicate_scatter_Hilltop.plot([compressed_Hilltop['replicate_1'].min(), compressed_Hilltop['replicate_1'].max()], [compressed_Hilltop['replicate_1'].min(), compressed_Hilltop['replicate_1'].max()], color='red', linestyle='--', label='y=x')
        ax_replicate_scatter_Hilltop.set_xlabel('Replicate 1 ($M^{-1}s^{-1}$)')
        ax_replicate_scatter_Hilltop.set_ylabel('Replicate 2 ($M^{-1}s^{-1}$)')
        ax_replicate_scatter_Hilltop.set_title('Exponential $k_{cat}/K_M$ Replicates')
        ax_replicate_scatter_Hilltop.set_xscale('log')
        ax_replicate_scatter_Hilltop.set_yscale('log')
    except:
        pass

    plt.savefig(newpath + '00_Chip_Summary.pdf')
    plt.show()

    ## Initialize v_isothermal ============================================================
    def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
    v_isotherm = np.vectorize(isotherm)

    # initialize starting chamber
    start_x = 1
    start_y = 1

    # if user specifies starting chamber, start from there
    if starting_chamber is not None:
        start_x = starting_chamber[0]
        start_y = starting_chamber[1]

    # initialize tqdm
    with tqdm(total = (device_columns*device_rows) - ((start_x-1)*device_rows + (start_y-1))) as pbar:

        # Plot Chamber-wise Summaries
        for x in range(start_x, device_columns + 1, 1):
            for y in range(start_y, device_rows + 1, 1):

                # Define Indices
                Indices = f"{x:02d}" + ',' + f"{y:02d}"
                
                ## initialize data dfs ============================================================
                export_kinetic_df = sq_merged[sq_merged['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                export_standards_df = squeeze_standards[squeeze_standards['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                export_sq_merged = sq_merged[sq_merged['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                # export_mm_df = squeeze_mm[squeeze_mm['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                
                ## defining figure ============================================================
                # initialize figure
                fig = plt.figure(figsize=(10, 12))

                # increase the spacing between the subplots
                fig.subplots_adjust(wspace=0.5, hspace=0.7)

                ## defining subplots ============================================================
                ax_image = plt.subplot2grid((10, 12), (0, 0), colspan=2) # chamber image
                
                # define concentration series
                concs = set(export_kinetic_df['substrate_conc_uM'])

                # initialize progress curve subplot2grid objects
                for n, conc in enumerate(sorted(concs)):
                    if n == 0:
                        globals()['ax_progress_curve_' + str(n)] = plt.subplot2grid((5, len(concs)), (1, n), colspan=1, rowspan=1) # progress curves
                    else:
                        globals()['ax_progress_curve_' + str(n)] = plt.subplot2grid((5, len(concs)), (1, n), colspan=1, rowspan=1, sharey=ax_progress_curve_0) # progress curves

                # initialize progress curve zoom subplot2grid objects (in row below progress curves)
                for n, conc in enumerate(sorted(concs)):
                    if n == 0:
                        globals()['ax_progress_curve_zoom_' + str(n)] = plt.subplot2grid((5, len(concs)), (2, n), colspan=1, rowspan=1)
                    else:
                        globals()['ax_progress_curve_zoom_' + str(n)] = plt.subplot2grid((5, len(concs)), (2, n), colspan=1, rowspan=1, sharey=ax_progress_curve_zoom_0) 
            
                # define table, mm curve, and pbp std curve subplot2grid objects
                ax_table = plt.subplot2grid((8, 10), (0, 3), colspan=5) # table
                ax_mm_curve = plt.subplot2grid((8, 10), (6, 0), colspan=3, rowspan=3)
                ax_std_curve = plt.subplot2grid((8, 10), (6, 7), colspan=3, rowspan=3)


                ## image plotting ============================================================
                img_idx = (x, y)
                ax_image.imshow(button_stamps[img_idx], cmap='gray', vmin=0, vmax=10**4.5)


                ## std curve fit  ============================================================
                # if export_standards_df is empty, skip this step
                if export_standards_df.empty == False:
                    xdata, ydata = export_standards_df.standard_concentration_uM.values[0], export_standards_df.standard_median_intensities.values[0]
                    chamber_popt = export_standards_df.standard_popt.values[0]
                else:
                    xdata, ydata = np.nan, np.nan
                    chamber_popt = np.nan

                # define the interpolation
                interp_xdata = np.linspace(-np.min(xdata), np.max(xdata), num=100, endpoint=False)

                # if standard type is PBP, plot
                if standard_type.upper() == 'PBP':
                    if type(chamber_popt) != float: # if fit is successful
                        ax_std_curve.plot(interp_xdata, v_isotherm(interp_xdata, *chamber_popt), color='g', linestyle='dashed', label='Isotherm Fit')
                    ax_std_curve.scatter(xdata, ydata, color='b', s=50, alpha=0.4, label='data')
                    ax_std_curve.set_ylabel('Intensity')
                    ax_std_curve.set_xlabel('Pi Concentration (uM)')
                    ax_std_curve.legend(loc='lower right', shadow=True, title='PBP Conc = %s uM' % pbp_conc)

                # if standard type is linear, plot
                elif standard_type == 'linear':
                    if type(chamber_popt) != float:
                        def linear(x, a, b): return a*x + b 
                        v_linear = np.vectorize(linear)
                        ax_std_curve.plot(xdata, v_linear(xdata, *chamber_popt), 'r-', label='Linear Fit')
                        ax_std_curve.plot(interp_xdata, v_linear(interp_xdata, *chamber_popt), color='g', linestyle='dashed', label='interpolation')
                    ax_std_curve.scatter(xdata, ydata, color='b', s=50, alpha=0.4, label='data')
                    ax_std_curve.set_ylabel('Intensity')
                    ax_std_curve.set_xlabel('Product Concentration (uM)')
                    ax_std_curve.legend(loc='lower right', shadow=True, title='Standard Curve')


 ## Hill curve plotting ============================================================
                # if df is not empty, plot
                if export_sq_merged.empty == False:
                    xdata = export_sq_merged['substrate_conc_uM'].tolist()
                    ydata = export_sq_merged['initial_rate'].tolist()
                    MutantID = export_sq_merged['MutantID'].tolist()[0]
                    Indices = export_sq_merged['Indices'].tolist()[0]
                    HillCurve_top = export_sq_merged['HillCurve_top'].tolist()[0]
                    HillCurve_bottom = export_sq_merged['HillCurve_bottom'].tolist()[0]
                    HillCurve_ic50 = export_sq_merged['HillCurve_ic50'].tolist()[0]
                    HillCurve_hill_slope = export_sq_merged['HillCurve_hill_slope'].tolist()[0]

                    # define Hill function
                    def Hill_func(X, Bottom, Top, IC50, HillSlope):
                        """
                        X = substrate concentration
                        Bottom = minimum value
                        Top = maximum value
                        IC50 = half-maximum value
                        HillSlope = steepness of curve
                        """
                        return Bottom + ((Top - Bottom) / (1 + 10**((np.log10(X) - np.log10(IC50)) * HillSlope)))


                    # plot points and curve fit
                    t = np.arange(0, max(xdata), 0.2)
                    fit_ydata = Hill_func(t, *[HillCurve_bottom, HillCurve_top, HillCurve_ic50, HillCurve_hill_slope])
                    ax_mm_curve.plot(xdata, ydata, 'o', color='b', label='Fit Data')
                    ax_mm_curve.plot(t, fit_ydata, 'g--', label='Hill Fit')
                    ax_mm_curve.axhline(HillCurve_top, label='Top')
                    ax_mm_curve.axhline(HillCurve_bottom, label='Bottom')
                    ax_mm_curve.axvline(HillCurve_ic50, label='IC50')
                    ax_mm_curve.plot(t, Hill_func(t, *[HillCurve_bottom, HillCurve_top, HillCurve_ic50, HillCurve_hill_slope]), 'g--', label='Hill Fit')

                    # add end zero condition as horizontal line
                    end_slope = export_sq_merged[export_sq_merged['substrate_conc_uM'] == -0.1]['initial_rate'].tolist()[0]
                    ax_mm_curve.axhline(end_slope, label='End Zero Condition', color='r')

                    ax_mm_curve.set_xscale('log')
                    ax_mm_curve.set_ylabel('Initial Rate')
                    ax_mm_curve.set_xlabel('%s Concentration (nM)' % Inhibitor)
                    ax_mm_curve.set_title('Hill Curve Fit for\n%s %s' % (MutantID, Indices))
                    ax_mm_curve.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='R2 = %s' % round(export_sq_merged['HillCurve_r_squared'].tolist()[0], 2))

                    # set limits
                    miny, maxy = ax_mm_curve.get_ylim()
                    ax_mm_curve.set_ylim(0, maxy*1.1)



                ## progress curve plotting ============================================================
                # find local background indices and store data in new df
                local_background_indices = get_local_bg_indices(x=x, y=y, device_rows=device_rows) # stores indices of local background chambers in a list
                local_background_df = pd.DataFrame([])
                for i in local_background_indices:
                    data = sq_merged[sq_merged['Indices'] == f"{i[0]:02d}" + ',' + f"{i[1]:02d}"]
                    local_background_df = local_background_df.append(data)

                # plot curves
                for ax, conc in enumerate(sorted(concs)):
                    plot_progress_curve(export_kinetic_df, conc=conc, ax=globals()['ax_progress_curve_' + str(ax)] , fit_descriptors=False, kwargs_for_scatter={"s": 15, "c": 'blue'}, kwargs_for_line={"c": 'blue'}, pbp_conc=pbp_conc, substrate_name=substrate)
                    plot_progress_curve(local_background_df, conc=conc, ax=globals()['ax_progress_curve_' + str(ax)], kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'}, pbp_conc=pbp_conc, fit_descriptors=False, substrate_name=substrate)

                    # set ylabel on first (left-most) plot
                    if ax == 0:
                        globals()['ax_progress_curve_' + str(ax)].set_ylabel('Product Conc (uM)')
                    else:
                        plt.setp(globals()['ax_progress_curve_' + str(ax)].get_yticklabels(), visible=False)
                    
                # set ylims
                # get all product concentrations
                all_product_concs = []
                for index, row in export_kinetic_df.iterrows():
                    all_product_concs.append(row['kinetic_product_concentration_uM'])

                # flatten list of all product concentrations
                if len(all_product_concs) == 0:
                    all_product_concs = [0]
                elif type(all_product_concs[0]) == list:
                    all_product_concs = [item for sublist in all_product_concs for item in sublist]
                elif type(all_product_concs[0]) == float:
                    all_product_concs = [0]

                # now set ylims for each ax
                for ax, conc in enumerate(sorted(concs)):
                    globals()['ax_progress_curve_' + str(ax)].set_ylim(min(all_product_concs), max(all_product_concs)*1.1)

                ## add second row of progress curves, zoomed in
                # set max_x for zoomed in plot
                max_x = 500 # seconds

                # plot curves
                for ax, conc in enumerate(sorted(concs)):
                    plot_progress_curve(export_kinetic_df, conc=conc, ax=globals()['ax_progress_curve_zoom_' + str(ax)] , fit_descriptors=True, kwargs_for_scatter={"s": 15, "c": 'blue'}, kwargs_for_line={"c": 'blue'}, pbp_conc=pbp_conc, substrate_name=substrate)
                    plot_progress_curve(local_background_df, conc=conc, ax=globals()['ax_progress_curve_zoom_' + str(ax)], kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'}, pbp_conc=pbp_conc, fit_descriptors=False, substrate_name=substrate)

                    # set ylabel on first (left-most) plot
                    if ax == 0:
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_ylabel('Product Conc (uM)')
                    else:
                        plt.setp(globals()['ax_progress_curve_zoom_' + str(ax)].get_yticklabels(), visible=False)

                    # remove title
                    if ax == 0:
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_title('Zoomed at %s seconds' % max_x, fontsize=8, loc='left')
                    else:
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_title('')


                # set xlim and ylim
                for ax, conc in enumerate(sorted(concs)):
                    globals()['ax_progress_curve_zoom_' + str(ax)].set_xlim(-30, max_x)

                # set y upper limit to pbp_conc
                time_list = export_kinetic_df['time_s'].iloc[0]
                max_x_index = min(range(len(time_list)), key=lambda i: abs(time_list[i]-max_x))

                # get max product concentration before max_x
                all_product_concs_zoom = []
                for index, row in export_kinetic_df.iterrows():
                    if type(row['kinetic_product_concentration_uM']) == list:
                        all_product_concs_zoom.append(row['kinetic_product_concentration_uM'][:max_x_index])
                    else:
                        all_product_concs_zoom.append(row['kinetic_product_concentration_uM'])
                        # remove nans
                        all_product_concs_zoom = [x for x in all_product_concs_zoom if str(x) != 'nan']

                # if all_product_concs_zoom is not empty, set ylim
                if len(all_product_concs_zoom) != 0:
                    # flatten list of all product concentrations
                    all_product_concs_zoom = [item for sublist in all_product_concs_zoom for item in sublist]

                    for ax, conc in enumerate(sorted(concs)):
                        globals()['ax_progress_curve_zoom_' + str(ax)].set_ylim(min(all_product_concs), 1.2*max(all_product_concs_zoom))
                else:
                    print('all_product_concs_zoom is empty')
                    


                ## table plotting ============================================================
                if export_sq_merged.empty == False:
                    table_df = export_sq_merged[['Indices', 'EnzymeConc', 'egfp_manual_flag', 'local_bg_ratio', 'HillCurve_bottom', 'HillCurve_top', 'HillCurve_ic50', 'HillCurve_hill_slope']].drop_duplicates(subset=['Indices'], keep='first')
                    table_df.apply(pd.to_numeric, errors='ignore', downcast='float')
                    table_df = table_df.round(decimals=2)
                    table_df['FilteringOutcome'] = 'Passed' # filtering outcome column
                    filtering_outcome = 'Passed' # default filtering outcome

                    # # if kcat/KM is greater than 10^6, convert to scientific notation
                    # if table_df['MM_kcat_over_KM_fit'].max() > 10**6:
                    #     table_df['MM_kcat_over_KM_fit'] = table_df['MM_kcat_over_KM_fit'].apply(lambda x: '%.2E' % x)

                    params_table = ax_table.table(cellText=table_df.values, loc='center', colLabels=['Idx', '[E] (nM)', 'eGFP Flag', 'LBGR', 'Hill Bottom', 'Hill Top', 'Hill IC50', 'Hill Hill Slope', 'Filtering Outcome'], colColours=['#f2f2f2']*9)
                    params_table.auto_set_font_size(False)
                    params_table.set_fontsize(8)
                    params_table.scale(0.1, 1.5)
                    params_table.auto_set_column_width(col=list(range(len(table_df.columns))))
                    ax_table.set_title(export_kinetic_df.iloc[0].MutantID)
                    ax_table.set_axis_off()
                    ax_table.set_frame_on(False)

                    ## table filtering
                    # if value is below the filter threshold from the filter_dict, set the corresponding column color to light red
                    for key, value in filter_dictionary.items():
                        for row in range(len(table_df)):
                            if (type(value) == int) or (type(value) == float):
                                if table_df.iloc[row][key] < value:
                                    params_table[(row+1, table_df.columns.get_loc(key))].set_facecolor('#ffcccc')
                                    filtering_outcome = 'Failed'
                            else:
                                if table_df.iloc[row][key] == value:
                                    params_table[(row+1, table_df.columns.get_loc(key))].set_facecolor('#ffcccc')
                                    filtering_outcome = 'Failed'

                    # update cell in params table with filtering outcome and change color
                    if filtering_outcome == 'Failed':
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].get_text().set_text(filtering_outcome)
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].set_facecolor('#ffcccc')
                    else:
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].get_text().set_text(filtering_outcome)
                        params_table[(row+1, table_df.columns.get_loc('FilteringOutcome'))].set_facecolor('#ccffcc')

                # update progress bar
                pbar.update(1)
                pbar.set_description("Processing %s" % str((x, y)))

                plt.savefig(newpath + str(Indices) + '.pdf')
                plt.close('all')

        pbar.set_description("Export complete")

# ====================================================================================================
# MERGE PDFS
# ====================================================================================================

# merge all pdfs into one
def merge_pdfs(export_path_root, substrate, experimental_day):
    """
    Merges all pdfs in the export_path_root/PDF/pages/ directory into one pdf file.
    
    Parameters:
    export_path_root (str): path to the directory containing the PDF directory
    substrate (str): name of the substrate
    experimental_day (str): date of the experiment

    Returns:
    None
    """

    # add trailing slash to export_path_root if not present
    if 'PDF' not in export_path_root:
        mypath = export_path_root + '/PDF/'
    elif export_path_root[-1] != '/':
        mypath = export_path_root + '/'

    # store all pdf files in a list
    all_files = [f for f in listdir(mypath + 'pages/') if isfile(join(mypath + 'pages/', f))]

    # remove files beginning with '._'
    all_files = [f for f in all_files if not f.startswith('._')]

    # remove nan files
    all_files = [f for f in all_files if not f.startswith('nan')]

    print('Found %d pdfs.' % len(all_files))

    import time
    # get length of all_files
    print('Merging %d pdfs...' % len(all_files))
    time.sleep(1)

    # initialize merger
    merger = PyPDF2.PdfMerger()

    # remove summary from all_files for sorting
    summary = [f for f in all_files if f.startswith('00')]
    all_files = [f for f in all_files if not f.startswith('00')]
    # sort all files based on indices of the form: 01,01
    all_files = sorted(all_files, key=lambda x: int(x.split('.')[0].split(',')[0])*100 + int(x.split('.')[0].split(',')[1]))
    all_files = summary + all_files
    print('First three files: %s' % all_files[:3])

    # merge pdfs with tqdm progress bar
    for pdf in tqdm(all_files, desc='Merging PDFs'):
        # merge pdf
        try:
            merger.append(mypath + 'pages/' + pdf)
        except:
            print('Could not merge %s' % pdf)

    # write merged pdf to file
    print('Writing merged pdf to file...')
    merged_filename = mypath + 'merged_' + substrate + '_' + experimental_day + '.pdf'
    merger.write(merged_filename)
    print('Done.')

    # close merger
    merger.close()

    return merged_filename


# ====================================================================================================
# EXPORT DATA TO CSV
# ====================================================================================================

# export data to csv
def export_data(sq_merged, squeeze_mm, export_path_root, experimental_day, setup, device, substrate, experiment_name):
    """ Format and export data to csv. 
    Description
    -----------
    Adds additional columns with information about the experiment. Exports to csv.

    Parameters
    ----------
    sq_merged (pandas.DataFrame): Dataframe containing all data from the squeeze analysis.
    squeeze_mm (pandas.DataFrame): Dataframe containing all data from the Michaelis-Menten curve fits.
    export_path_root (str): Path to the folder where the csv will be exported.
    experimental_day (str): Experimental day.
    setup (str): Microscopy setup.
    device (str): Device number on manifold.
    substrate (str): Substrate name.
    experiment_name (str): Experiment name.

    Returns
    -------
    None
    """
    # add additional columns to squeeze_mm
    if 'num_points_fit_vo' in squeeze_mm.columns:
        squeeze_mm['num_points_fit_vo_list'] = squeeze_mm['num_points_fit_vo']
        squeeze_mm.drop(columns=['num_points_fit_vo'], inplace=True)

    # generate export dataframe
    export_df = pd.merge(squeeze_mm, sq_merged, on=list(np.intersect1d(sq_merged.columns, squeeze_mm.columns)))

    # add additional columns to export dataframe
    export_df['GlobalExperimentIndex'] = '_'.join([experimental_day, setup, device, substrate])
    export_df['Experiment'] = experiment_name

    # export to csv
    # export_df.to_csv(export_path_root + '/' + experimental_day + '_' + substrate + device + '_workup.csv')
    export_df.to_csv(export_path_root + '/' + '_'.join([experimental_day, device, substrate]) + '_workup.csv')

    # return filepath
    # return export_path_root + '/' + experimental_day + '_' + substrate + '_workup.csv'
    return export_path_root + '/' + '_'.join([experimental_day, device, substrate]) + '_workup.csv'