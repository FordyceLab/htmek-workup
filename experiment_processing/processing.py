# Author: Micah Olivas
# Date: 2023-3-31
# Description: This script contains functions for processing the data from the image processing script. It is designed to be imported into the analysis notebook.



# Standard imports
import pandas as pd
import seaborn as sns
import numpy as np
import re

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
from os import listdir
from os.path import isfile, join




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


# Create button objects slightly larger than the segmented chamber images
def create_expanded_button(description, button_style):
    return widgets.Button(description=description, button_style=button_style, layout=widgets.Layout(height='200px', width='200px'))

# Creat action to add chamber index to flagged set if button is clicked
def capture_button(chamberindex):
    def on_button_clicked(b):
        global flagged_set
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
    """
    The minimum and maximum values np.interp

    """
    rescaled_image = np.interp(x = img_arr, 
                               xp = (img_arr.min(), img_arr.max()), 
                               fp = (0, 10**6.5)
                               ).astype(np.uint32)
    img_byte_buffer = io.BytesIO()
    pil_img = Image.fromarray(rescaled_image)
    pil_img.save(img_byte_buffer, format = format)
    img_byte_arr = img_byte_buffer.getvalue()
    return img_byte_arr

# Create image pane for display in widget
def make_image_pane(imagearr):
    return widgets.Image(
            value= imagearray_to_bytearr(imagearr),
            format='png',
            width='80px', height='80px', 
            min_width='80px', 
            max_height = '80px', max_width = '80px')

# Create button grid for display in widget
def make_button_grid(button_stamps_subset):
    items1 = []
    for enzymeconc, indices, imagearr in button_stamps_subset:
        
        chamber_index = str(indices)+' '+str(round(enzymeconc, 1))+' nM'
        tb = create_toggle_button(chamber_index)
        
        record_index_name = str(indices[0])+','+str(indices[1])
        tb.observe(capture_button(record_index_name), 'value')
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
    """
    Parameters
    ----------
    sq_merged : pandas dataframe
        Contains kinetic and standard data for all chambers.
    egfp_button_summary_image_path : string
        Points to directory of eGFP button summary image form segmentation.
    NUM_ROWS : int
        Number of rows in device.
    NUM_COLS : int
        Number of columns in device.
    culling_export_directory : string
        Points to directory where culling record will be saved.

    Returns
    -------
    button_grid : ipywidgets grid object
        Interactive grid of buttons with eGFP images for each chamber.
    flagged_set : set
        Set of chamber indices that were flagged for culling.
    button_stamps: list
        List of tuples containing button stamp details.
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
        flagged_set = set(df.loc[df['egfp_manual_flag'] == True]['Indices'])

        # create button grid
        for i in flagged_set:
            capture_button(i)

        # Create an abbreviated filepath string to show two levels of parent directories above the culling record
        culling_export_filepath_split = culling_export_filepath.split('/')
        culling_filepath_abbrev = '.../' + '/'.join(culling_export_filepath_split[-4:])

        # Print
        print('Culling record found: %s. \nLoaded culling record.' % culling_filepath_abbrev)

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

        # create button grid
        button_grid = None

    # If culling record does not exist, create a new one
    elif culling_record_exists == False:

        print('No culling record found in export directory. Creating button grid for manual culling...')

        # create empty set to store flagged chambers
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
        button_grid = make_button_grid(button_stamps_subset=button_stamps_subset)

        print('Button grid object created. Please wait for your viewport to render the grid...')

    return button_grid, flagged_set, button_stamps


# Handle flagged chambers
def handle_flagged_chambers(sq_merged, flagged_set, culling_export_directory):
    """
    Parameters
    ----------
    sq_merged : pandas dataframe
        Contains kinetic and standard data for all chambers.
    flagged_set : set
        Contains chamber indices of flagged chambers.
    culling_export_directory : string
        Points to directory where culling record will be saved.

    Returns
    -------
    sq_merged : pandas dataframe
        Contains kinetic and standard data for all chambers, with flagged chambers removed.
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
        culling_export_filepath = os.path.join(culling_export_directory, culling_filename)
        df = pd.read_csv(culling_export_filepath)
        flagged_set = set(df.loc[df['egfp_manual_flag'] == True]['Indices'])

    elif culling_record_exists == False:
        print('Creating new culling record.')
        culling_export_filepath = os.path.join(culling_export_directory, 'manual_culling_record.csv')
        sq_merged.sort_values(['x', 'y',  'MutantID', 'substrate_conc_uM', 'egfp_manual_flag']).to_csv(culling_export_filepath, index=False)
        print('Saved culling record to %s' % culling_export_filepath)

    # create new column in sq_merged to store culling status based on flagged set
    bool_flagged_set = set(df.loc[df['egfp_manual_flag'] == True]['Indices'])
    sq_merged['egfp_manual_flag'] = sq_merged.Indices.apply(lambda i: i in bool_flagged_set)

    return sq_merged


def get_initial_rates(sq_merged, pbp_conc = 30):
    """
    Parameters
    ----------
    sq_merged : pandas dataframe
        Contains kinetic and standard data for all chambers.
    pbp_conc : float, optional
        The concentration of Phosphate Binding Protein (PBP) in uM. The default is 30.
    
    Returns
    -------
    sq_merged : pandas
        Contains kinetic and standard data for all chambers, with initial rates calculated.
    """

    # define the least-squares algorithm
    def fit_initial_rate_row(row, pbp_conc=pbp_conc):

        substrate_conc = row.substrate_conc_uM
        x, y = row.time_s, row.kinetic_product_concentration_uM
        full_x, full_y = x, y 
        first_third = int(len(x)*0.3)
        regime = 0

        if substrate_conc < pbp_conc: # "regime 1"
            regime = 1
            if y[1] >= 0.3*substrate_conc: # fit two points, flag as a two point fit
                x = x[1:3]
                y = y[1:3]
            else: # fit first 30% of the series
                x = x[1:first_third]
                y = y[1:first_third]

        # "regime 2": 30% of the substrate concentration is greater than 2/3 the concentration of PBP.
        elif 0.3*substrate_conc > (2/3)*pbp_conc: # regime "2"
            regime = 2
            x = x[1:first_third]
            y = y[1:first_third]

        # "regime 3": 30% of the substrate concentration is below 2/3 the concentration of PBP. 
        elif 0.3*substrate_conc < (2/3)*pbp_conc: # regime "3"
            regime = 3
            x = x[1:first_third]
            y = y[1:first_third]
            # y = [ i for i in y if i < 0.3 * substrate_conc ] # fit points only if less than 30% of substrate conc
            # x = x[:len(y)]

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
            
        return m, c, two_point, regime

    # next, store the slope in a new column
    print('Fitting initial rates...')
    results = sq_merged.apply(fit_initial_rate_row, axis=1)
    print('Done fitting initial rates. Adding results to dataframe...')

    # now add the rates and intercepts to the dataframe
    sq_merged['initial_rate'] = results.apply(lambda x: x[0])
    sq_merged['initial_rate_intercept'] = results.apply(lambda x: x[1])
    sq_merged['two_point_fit'] = results.apply(lambda x: x[2])
    sq_merged['rate_fit_regime'] = results.apply(lambda x: x[3])


    # Plot several initial rates
    print('Plotting progress curves for one random mutant...')
    # set list of substrate concentrations
    concs = sorted(list(set(sq_merged['substrate_conc_uM'])))
    conc_d = dict(zip(range(len(concs)), concs))

    # sample a random mutants and subset the progress curve dictionary
    check = np.random.choice(list(set(sq_merged['MutantID'])), 1)[0]
    my_df = sq_merged[sq_merged['MutantID'] == check]
    my_df

    fig, axs = plt.subplots(ncols=5, figsize=(10, 3), sharey=True)

    for k,v in conc_d.items():
        v_df = my_df[my_df['substrate_conc_uM'] == v]

        for index, row in v_df.iterrows():
            times, product_concs = row['time_s'], row['kinetic_product_concentration_uM']
            m, b = row.initial_rate, row.initial_rate_intercept
            indices = row.Indices

            axs[k].scatter(times, product_concs, s=10)
            axs[k].plot([0, max(times)], [b, m*max(times)+b], '--')
            axs[k].set_title(str(v) + 'uM AcP')
            axs[k].set_xlim(0, 2000)
            axs[k].set_ylim(0, pbp_conc)
            axs[k].set_ylim(0, pbp_conc)
            axs[k].set_xlabel('Time (s)')

    # only set the y label for the first plot
    axs[0].set_ylabel('Product Concentration (uM)')

    fig.suptitle("Library Member: " + check, y=1.1)
    plt.show()

    return sq_merged


# define michaelis menten equation
def mm_func(S, Km, Vmax): 
    return Vmax*S/(Km+S)
v_mm_func = np.vectorize(mm_func)


# fit michaelis-menten equation to the initial rates
def fit_michaelis_menten(sq_merged, exclude_concs):
    """
    Parameters
    ----------
    sq_merged : pandas dataframe
        Contains kinetic and standard data for all chambers, with initial rates calculated.

    Returns
    -------
    sq_merged : pandas dataframe
        Contains kinetic and standard data for all chambers, with initial rates calculated and Michaelis-Menten fit parameters calculated.
    """

    # =================================================================================================
    # DATAFRAME SETUP

    # first, group by Indices to create initial rate and substrate conc series
    initial_rates = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['initial_rate'].apply(list).reset_index(name='initial_rates')
    substrate_concs = sq_merged.groupby(['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])['substrate_conc_uM'].apply(list).reset_index(name='substrate_concs')
    squeeze_mm = pd.merge(substrate_concs, initial_rates, on=['x', 'y', 'Indices', 'MutantID', 'substrate', 'EnzymeConc', 'egfp_manual_flag'])

    # get the substrate concentration list from the first row
    all_substrate_concs = squeeze_mm['substrate_concs'].iloc[0]

    # now, add zero to the beginning of each list of substrate concentrations and initial rates if there is no zero substrate condition
    if np.min(all_substrate_concs) != 0:
        squeeze_mm['substrate_concs'] = squeeze_mm['substrate_concs'].apply(lambda x: [0] + x)
        squeeze_mm['initial_rates'] = squeeze_mm['initial_rates'].apply(lambda x: [0] + x)
    
    # =================================================================================================
    # FIT MICHAELIS-MENTEN EQUATION

    # define curve fit function
    def fit_mm_curve(df, exclude_concs=exclude_concs):

        # store data
        xdata = df.substrate_concs
        ydata = df.initial_rates
        enzyme_conc = df.EnzymeConc
        name = df.MutantID

        # now, exclude substrate concentrations from fit if needed
        for conc in exclude_concs:
            if conc in xdata:
                conc_index = xdata.index(conc)

                del xdata[conc_index]
                del ydata[conc_index]

        ydata_final = ydata
        xdata_final = xdata
        
        # # IN DEVELOPMENT: remove non-approximately increasing rates
        # final_index = 0
        # for index, rate in enumerate(ydata):
        #     if index > 0:
        #         if ydata[index] > (ydata[index-1] * 0.8) :
        #             # trunc_ydata = ydata[:index+1]
        #             final_index += 1
        #         else:
        #             final_index += 0

        # xdata_final = xdata[:final_index]
        # ydata_final = ydata[:final_index]

        # if '1A1_' in name:
        #     xdata_final = xdata[:3]
        #     ydata_final = ydata[:3]
            

        # perform curve fit
        mm_params, pcov = optimize.curve_fit(mm_func, xdata=xdata_final, ydata=ydata_final, bounds=([0, 0], [np.inf, np.inf]))

        # store variables
        KM_fit = mm_params[0] # at this point, this is in uM
        vmax_fit = mm_params[1] # at this point, in uM per second
        kcat_fit = vmax_fit/(enzyme_conc/1000) # at this point, enzyme conc is in nM, so convert to uM; kcat is in s^-1
        kcat_over_KM_fit = 10**6 * kcat_fit/KM_fit # at this point, kcat in s^-1 and KM in uM, so multiply by 1000000 to give s^-1 * M^-1 

        return KM_fit, vmax_fit, kcat_fit, kcat_over_KM_fit, xdata_final, ydata_final

    # apply function and store fit parameters
    print('Fitting Michaelis-Menten curves...')
    results = squeeze_mm.parallel_apply(fit_mm_curve, axis=1)
    print('Done fitting Michaelis-Menten curves.')

    squeeze_mm['KM_fit'] = results.apply(lambda x: x[0])
    squeeze_mm['vmax_fit'] = results.apply(lambda x: x[1])
    squeeze_mm['kcat_fit'] = results.apply(lambda x: x[2])
    squeeze_mm['kcat_over_KM_fit'] = results.apply(lambda x: x[3])
    squeeze_mm['substrate_concs'] = results.apply(lambda x: x[4])
    squeeze_mm['initial_rates'] = results.apply(lambda x: x[5])


    # ==========================================================================================
    # PLOT EXAMPLES
    # select example chambers to plot
    print('Plotting examples...')
    num_examples = 4
    examples_df = squeeze_mm[squeeze_mm['MutantID'] != 'BLANK'].sample(num_examples)
    examples_df = examples_df.reset_index()

    # plot subplots
    fig, axs = plt.subplots(1, num_examples, figsize=(20, 3), sharey=True)

    # iterate 
    for index, row in examples_df.iterrows():
        xdata = row.substrate_concs
        ydata = row.initial_rates
        enzyme_conc = row.EnzymeConc
        vmax_fit = row.vmax_fit
        KM_fit = row.KM_fit

        # plot
        t = np.arange(0, float(max(all_substrate_concs)), float(max(all_substrate_concs))/500)
        axs[index].plot(t, v_mm_func(t, *[KM_fit, vmax_fit]), 'g--')
        axs[index].plot(xdata, ydata, 'bo', label='data')
        axs[index].axhline(vmax_fit)
        axs[index].set_ylabel('Initial Rate')
        axs[index].set_xlabel('[AcP]')
        axs[index].set_ylim(0, 0.1)

        # add title with concentration rounded to 2 decimal places
        axs[index].set_title(row.MutantID + '\n' + 'Conc:' + str(row.EnzymeConc) + ' nM')
        axs[index].legend(loc='upper left', fancybox=True, shadow=True)

    return squeeze_mm

# get local background indices
def get_local_bg_indices(x,y,device_rows):
    """Get indices of local background chambers for a given chamber.

    Parameters
    ----------
    x : int
        x coordinate of chamber
    y : int
        y coordinate of chamber

    Returns
    -------
    local_bg_indices : list
        list of tuples containing x and y coordinates of local background chambers
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

# calculate local background ratio for every chamber
def calculate_local_bg_ratio(squeeze_mm, sq_merged, device_rows, exclude_concs=[]):
    """Calculate local background ratio for every chamber.

    Parameters
    ----------
    squeeze_mm : pandas dataframe
        dataframe containing squeeze data
    sq_merged : pandas dataframe
        dataframe containing squeeze data

    Returns
    -------
    squeeze_mm : pandas dataframe
        dataframe containing squeeze data
    """

    # get set of substrate concs
    substrate_concs = set(sq_merged.substrate_conc_uM)

    # get highest allowed substrate concentration
    max_acp_conc = max(substrate_concs ^ set(exclude_concs)) # gets maximum value from disjoint of the two sets

    # save this as a df
    local_bg_df = sq_merged[sq_merged['substrate_conc_uM'] == max_acp_conc]
    local_bg_df['index_tuple'] = local_bg_df.apply(lambda row: (row.x, row.y), axis=1)
    local_bg_df = local_bg_df.reset_index()  # make sure indexes pair with number of rows
    local_bg_df['initial_rate'] = local_bg_df['initial_rate'].round(5)

    for index, row in local_bg_df.iterrows():

        # store chamber details
        chamber_rate = row.initial_rate
        x = row.x
        y = row.y

        # get local (above and below) chamber neighbors
        lbg_idxs = get_local_bg_indices(x,y,device_rows)

        # get rates for each of neighboring chambers
        lbg_rates = []

        for tup in lbg_idxs:
            rate = local_bg_df.loc[local_bg_df['index_tuple'] == tup, 'initial_rate'].iloc[0]
            
            # account for zero-slope rates
            if rate != 0:
                lbg_rates.append(rate)
            else:
                lbg_rates.append(0)
                

        # get average of lbg_rates and calculate ratio
        lbg_avg = np.mean(lbg_rates)
        
        if lbg_avg != 0:
            lbg_ratio = chamber_rate/lbg_avg
        else:
            lbg_ratio = 0

        # store ratio in new column
        local_bg_df.loc[index, 'local_bg_ratio'] = lbg_ratio


    # add column into sq_merged
    sq_merged = pd.merge(sq_merged, local_bg_df[["Indices", "local_bg_ratio"]], on="Indices", how="left")
    squeeze_mm = pd.merge(squeeze_mm, local_bg_df[["Indices", "local_bg_ratio"]], on="Indices", how="right")

    return squeeze_mm, sq_merged


# ==========================================================================================
# PDF OUTPUT
# ==========================================================================================

# plot progress curves in PDF output file
def plot_progress_curve(df, conc, ax, kwargs_for_scatter, kwargs_for_line, fit_descriptors=False, exclude_concs=[]):
    """
    Inputs:
        - concs: current substrate concentrations
        - df: current chamber df
        - ax: position of plot
        - fit_descriptors: boolean to add descriptors for two-point fit, mm fit exclusion, and initial rate fit regime
    """

    # create a df of substrate concentrations
    conc_df = df[df['substrate_conc_uM'] == conc]


    for index, row in conc_df.iterrows():

        times, product_concs = np.array(row['time_s']), row['kinetic_product_concentration_uM']
        vi = row['initial_rate']
        intercept = row['initial_rate_intercept']
        regime = row['rate_fit_regime']
        two_point_fit = row['two_point_fit']

        # plot data for the current chamber
        ax.scatter(times, product_concs, **kwargs_for_scatter) # plot progress curve
        ax.plot(times, (times*vi) + intercept, **kwargs_for_line) # plot initial rate line
        # ax.set_xticklabels([]) # remove tick labels

        if fit_descriptors==True:

            # add regime text
            ax.text(0, -0.45, "Fit Regime: " + str(regime), transform=ax.transAxes) # Here, transAxes applies a transform to the axes to ensure that spacing isn't off between plots

            # add two-point fit
            if two_point_fit == True:
                ax.text(0, -0.6, "Two-point: " + str(two_point_fit), color='red', transform=ax.transAxes) # Here, transAxes applies a transform to the axes to ensure that spacing isn't off between plots
            else:
                ax.text(0, -0.6, "Two-point: " + str(two_point_fit), transform=ax.transAxes) # Here, transAxes applies a transform to the axes to ensure that spacing isn't off between plots
            
            # add MM exclusion text
            if conc in exclude_concs:
                ax.text(0, -0.75, "MM-fit: Held out", transform=ax.transAxes) # Here, transAxes applies a transform to the axes to ensure that spacing isn't off between plots

            # add initial rate text
            ax.text(0, -0.9, "$V_i$: " + str(round(vi, 5)), transform=ax.transAxes) # Here, transAxes applies a transform to the axes to ensure that spacing isn't off between plots

        ax.set_title(str(conc) + 'uM AcP')
        ax.set_box_aspect(1)
        # ax.set_ylim([0, max(product_concs)*1.2])
        ax.set_ylim([0, 30])


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


# main function to plot progress curves, heatmap, and chamber descriptors for experiment
def plot_chip_summary(squeeze_mm, sq_merged, squeeze_standards, squeeze_kinetics, button_stamps, device_columns, device_rows, export_path_root, experimental_day, experiment_name, exclude_concs=[]):

    # create export directory
    newpath = export_path_root + '/PDF/pages/'

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    ## defining figure ============================================================
    # initialize figure
    fig = plt.figure(figsize=(10, 10)) # size of print area of A4 page is 7 by 9.5

    # increase the spacing between the subplots
    fig.subplots_adjust(wspace=4, hspace=0.7)
    fig.suptitle(' '.join([experimental_day, experiment_name, 'Summary']))

    ## defining subplots ============================================================
    ax_conc_heatmap = plt.subplot2grid((5, 6), (0, 0), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_kinetic_heatmap = plt.subplot2grid((5, 6), (0, 2), rowspan=2, colspan=2) # enzyme concentration heatmap
    ax_egfp_hist = plt.subplot2grid((5, 6), (0, 4), rowspan=2, colspan=2) # enzyme concentration heatmap


    ## Heatmap plotting ============================================================
    # fill NaN
    grid = squeeze_mm.fillna(0)

    # plot enz conc
    grid_EC = grid.pivot('x', 'y', 'EnzymeConc').T
    grid_EC = grid_EC[grid_EC.columns[::-1]] # flip horizontally
    grid_EC = np.array(grid_EC)
    display_cbar=True # only display cbar on last grid
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    im, cbar = heatmap(grid_EC,
                        cmap="plasma", 
                        cbarlabel="Enzyme Conc (uM)", 
                        display_cbar=display_cbar,
                        ax=ax_conc_heatmap,
                        norm=norm
                        )
    ax_conc_heatmap.set_xticks([])
    ax_conc_heatmap.set_yticks([])
    ax_conc_heatmap.set_title('Enzyme Concentration \n by chamber')

    # plot enz LBG
    grid_kinetic = grid.pivot('x', 'y', 'local_bg_ratio').T
    grid_kinetic = grid_kinetic[grid_kinetic.columns[::-1]] # flip horizontally
    grid_kinetic = np.array(grid_kinetic)
    display_cbar = True # only display cbar on last grid
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    im, cbar = heatmap(grid_kinetic,
                        cmap="viridis", 
                        cbarlabel="Local BG Ratio", 
                        display_cbar=display_cbar,
                        norm=norm,
                        ax=ax_kinetic_heatmap,
                        )
    ax_kinetic_heatmap.set_xticks([])
    ax_kinetic_heatmap.set_yticks([])
    ax_kinetic_heatmap.set_title('Local BG Ratio \n by chamber')


    ## Histogram plotting ============================================================

    # remove blank chambers
    all_blanks = sq_merged[sq_merged['MutantID'] == 'BLANK'][[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')
    all_library = sq_merged[sq_merged['MutantID'] != 'BLANK'][[ "Indices", 'MutantID', 'EnzymeConc']].drop_duplicates(subset=['Indices'], keep='first')

    # plot histograms
    n, bins, patches = plt.hist(all_blanks['EnzymeConc'], stacked=True, bins=30, facecolor='grey', alpha=0.3, edgecolor='black')
    n, bins, patches = plt.hist(all_library['EnzymeConc'], stacked=True, bins=30, facecolor='green', alpha=0.3, edgecolor='green')

    # set plot 
    ax_egfp_hist.set_xlabel('eGFP Concentration (uM)')
    ax_egfp_hist.set_ylabel('Count')
    ax_egfp_hist.set_title('eGFP concentration \n in blank chambers')

    plt.savefig(newpath + '00_Chip_Summary.pdf')
    plt.show()

    ## Initialize v_isothermal ============================================================
    def isotherm(P_i, A, KD, PS, I_0uMP_i): return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I_0uMP_i
    v_isotherm = np.vectorize(isotherm)


    # initialize tqdm
    with tqdm(total = device_columns * device_rows) as pbar:

        # Plot Chamber-wise Summaries
        for x in range(1, 33):
            for y in range(1, 57, 1):
                
                ## initialize data df ============================================================
                export_kinetic_df = sq_merged[sq_merged['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                export_standards_df = squeeze_standards[squeeze_standards['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                export_mm_df = squeeze_mm[squeeze_mm['Indices'] == f"{x:02d}" + ',' + f"{y:02d}"]
                
                ## defining figure ============================================================
                # initialize figure
                fig = plt.figure(figsize=(10, 10)) # size of print area of A4 page is 7 by 9.5

                # increase the spacing between the subplots
                fig.subplots_adjust(wspace=0.3, hspace=0.7)

                ## defining subplots ============================================================
                ax_image = plt.subplot2grid((5, 5), (0, 0), colspan=1) # chamber image
                ax_first_progress_curve = plt.subplot2grid((5, 6), (1, 1), colspan=1) # progress curves
                ax_second_progress_curve = plt.subplot2grid((5, 6), (1, 2), colspan=1, sharey=ax_first_progress_curve) # progress curves
                ax_third_progress_curve = plt.subplot2grid((5, 6), (1, 3), colspan=1, sharey=ax_first_progress_curve) # progress curves
                ax_fourth_progress_curve = plt.subplot2grid((5, 6), (1, 4), colspan=1, sharey=ax_first_progress_curve) # progress curves
                ax_fifth_progress_curve = plt.subplot2grid((5, 6), (1, 5), colspan=1, sharey=ax_first_progress_curve) # progress curves
                ax_table = plt.subplot2grid((5, 6), (0, 1), colspan=5) # table
                ax_mm_curve = plt.subplot2grid((5, 6), (3, 0), rowspan=2, colspan=2) # chamber image
                ax_pbp_std_curve = plt.subplot2grid((5, 6), (3, 4), rowspan=2, colspan=2) # chamber image


                ## image plotting ============================================================
                img_idx = (x, y)
                ax_image.imshow(button_stamps[img_idx], cmap='gray', vmin=0, vmax=np.max(button_stamps[img_idx]))


                ## PBP std curve fit  ============================================================
                xdata, ydata = export_standards_df.standard_concentration_uM.values[0], export_standards_df.standard_median_intensities.values[0]
                chamber_popt = export_standards_df.standard_popt.values[0]

                # define the interpolation
                interp_xdata = np.linspace(-np.min(xdata), np.max(xdata), num=100, endpoint=False)

                # plot
                ax_pbp_std_curve.plot(xdata, v_isotherm(xdata, *chamber_popt), 'r-', label='Isotherm Fit')
                ax_pbp_std_curve.plot(interp_xdata, v_isotherm(interp_xdata, *chamber_popt), color='g', linestyle='dashed', label='interpolation')
                ax_pbp_std_curve.scatter(xdata, ydata, color='b', s=50, alpha=0.4, label='data')
                ax_pbp_std_curve.set_ylabel('Intensity')
                ax_pbp_std_curve.set_xlabel('Pi Concentration (uM)')
                ax_pbp_std_curve.legend(loc='lower right', shadow=True)
                ax_pbp_std_curve.set_title('PBP Standard Curve')


                ## Michaelis-Menten curve plotting ============================================================
                xdata = export_mm_df.iloc[0].substrate_concs
                ydata = export_mm_df.iloc[0].initial_rates
                MutantID = export_mm_df.iloc[0].MutantID
                Indices = export_mm_df.iloc[0].Indices
                KM_fit = export_mm_df.iloc[0].KM_fit
                vmax_fit = export_mm_df.iloc[0].vmax_fit

                # plot
                t = np.arange(0, 100, 0.2)
                fit_ydata = v_mm_func(t, *[KM_fit, vmax_fit])
                ax_mm_curve.plot(t, fit_ydata, 'g--', label='Fit') 
                ax_mm_curve.plot(xdata, ydata, 'bo', label='data')
                ax_mm_curve.axhline(vmax_fit, label='$v_{max}$')
                # ax_mm_curve.axvline(KM_fit, ymin=0, ymax=(vmax_fit/.2))
                ax_mm_curve.set_ylabel('$v_i$')
                ax_mm_curve.set_xlabel('AcP Concentration (uM)')
                ax_mm_curve.set_ylim([0, max(ydata)*1.2])
                ax_mm_curve.set_xlim([-max(set(squeeze_kinetics['substrate_conc_uM']))/20, max(set(squeeze_kinetics['substrate_conc_uM']))])
                ax_mm_curve.legend(loc='lower right', fancybox=True, shadow=True)
                ax_mm_curve.set_title('Michaelis-Menten Fit')


                ## progress curve plotting ============================================================
                concs = set(export_kinetic_df['substrate_conc_uM'])
                
                # find local background indices and store data in new df
                local_background_indices = get_local_bg_indices(x=x, y=y, device_rows=device_rows) # stores indices of local background chambers in a list
                local_background_df = pd.DataFrame([])
                for i in local_background_indices:
                    data = sq_merged[sq_merged['Indices'] == f"{i[0]:02d}" + ',' + f"{i[1]:02d}"]
                    local_background_df = local_background_df.append(data)

                # plot curves
                plot_progress_curve(export_kinetic_df, 10, ax_first_progress_curve, fit_descriptors=True, kwargs_for_scatter={"s": 10, "c": 'blue'}, kwargs_for_line={"c": 'blue'})
                plot_progress_curve(local_background_df, 10, ax_first_progress_curve, kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'})

                plot_progress_curve(export_kinetic_df, 25, ax_second_progress_curve, fit_descriptors=True, kwargs_for_scatter={"s": 10, "c": 'blue'}, kwargs_for_line={"c": 'blue'})
                plot_progress_curve(local_background_df, 25, ax_second_progress_curve, kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'})

                plot_progress_curve(export_kinetic_df, 50, ax_third_progress_curve, fit_descriptors=True, kwargs_for_scatter={"s": 10, "c": 'blue'}, kwargs_for_line={"c": 'blue'})
                plot_progress_curve(local_background_df, 50, ax_third_progress_curve, kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'})

                plot_progress_curve(export_kinetic_df, 75, ax_fourth_progress_curve, fit_descriptors=True, kwargs_for_scatter={"s": 10, "c": 'blue'}, kwargs_for_line={"c": 'blue'})
                plot_progress_curve(local_background_df, 75, ax_fourth_progress_curve, kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'})
                
                plot_progress_curve(export_kinetic_df, 100, ax_fifth_progress_curve, fit_descriptors=True, kwargs_for_scatter={"s": 10, "c": 'blue'}, kwargs_for_line={"c": 'blue'})
                plot_progress_curve(local_background_df, 100, ax_fifth_progress_curve, kwargs_for_scatter={"s": 5, "c": 'red', 'alpha': 0.5}, kwargs_for_line={"c": 'red', 'alpha': 0.5, 'linestyle': 'dashed'})


                ## table plotting ============================================================
                table_df = export_mm_df[['Indices', 'EnzymeConc', 'egfp_manual_flag', 'local_bg_ratio', 'kcat_fit', 'KM_fit', 'kcat_over_KM_fit']]
                table_df.apply(pd.to_numeric, errors='ignore', downcast='float')
                table_df = table_df.round(decimals=3)

                table = ax_table.table(cellText=table_df.values, loc='center', colLabels=['Indices', '[E] (nM)', 'eGFP Flag', 'Local BG Ratio', '$k_{cat}$', '$K_M$', '$k_{cat}/K_M$ ($M^{-1} s^{-1}$)'])
                table.auto_set_font_size(True)
                table.scale(0.3, 2)
                table.auto_set_column_width(col=list(range(len(table_df.columns))))
                ax_table.set_title(export_kinetic_df.iloc[0].MutantID)
                ax_table.set_axis_off()
                ax_table.set_frame_on(False)

                # update progress bar
                pbar.update(1)
                pbar.set_description("Processing %s" % str((x, y)))

                plt.savefig(newpath + str(Indices) + '.pdf')
                plt.close('all')

        pbar.set_description("Export complete")


# merge all pdfs into one
def merge_pdfs(export_path_root):

    # set path
    mypath = export_path_root + '/PDF/'

    # store all pdf files in a list
    all_files = sorted([mypath + 'pages/' + f for f in listdir(mypath + 'pages/') if isfile(join(mypath + 'pages/', f))])

    # remove files from all_files that are not pdfs containing two indices in the filename
    for file in all_files:
        if not file.endswith('.pdf'):
            all_files.remove(file)
        elif not re.search(r'\d{2},\d{2}', file):
            all_files.remove(file)

    # initialize merger
    merger = PyPDF2.PdfFileMerger()

    # merge pdfs with tqdm progress bar
    for pdf in tqdm(all_files, desc='Merging PDFs'):
        merger.append(pdf)

    # write merged pdf to file
    print('Writing merged pdf to file...')
    merger.write(mypath + 'merged.pdf')
    print('Done.')

    # close merger
    merger.close()


# ====================================================================================================
# EXPORT DATA TO CSV
# ====================================================================================================

# export data to csv
def export_data(sq_merged, squeeze_mm, export_df, export_path_root, experimental_day, setup, device, substrate, experiment_name):
    """ Export data to csv.

    Parameters
    ----------
    export_df : pandas.DataFrame
        Dataframe containing all data to be exported.

    export_path_root : str
        Path to export directory.

    experimental_day : str
        Experimental day.

    setup : str
        Microscopy setup.

    device : str
        Device number corresponding to the manifold used on the microscopy setup.

    substrate : str
        Substrate used in the experiment.

    experiment_name : str
        Description of the experiment.

    Returns
    -------
    None

    """
    # define export df
    export_df = pd.merge(squeeze_mm, sq_merged, on=list(np.intersect1d(sq_merged.columns, squeeze_mm.columns)))

    # add remaining data
    export_df['GlobalExperimentIndex'] = '_'.join([experimental_day, setup, device, substrate])
    export_df['Experiment'] = experiment_name

    # export to csv
    export_df.to_csv(export_path_root + '/' + experimental_day + '_' + substrate + '_workup.csv')