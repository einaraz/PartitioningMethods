import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Partitioning import Partitioning
from glob import glob
from pprint import pprint

"""
    Describes how to implement CEC, MREA, and FVS to partition high-frequency eddy-covariance data into flux components
    Implements a simple quality control (data screening and cleaning) and pre-processing (rotation, density corrections, detrending, etc)
    This code does not include data cleaning based on sensor flags. These flags change depending on the type of sonic/gas-analyzer used. Ideally, 
        removal of bad periods based on these flags should be included in this script. The quality control implemented here is only capable of
        detecting outliers and unphysical values, thus missing periods of bad data that can be easily identified by flags.


    A description of the necessary data input and options is given below. Please read carefully and modify according to your case.
    

        listfiles: string. 
                   Path where all data (text) files are located; any extension is valid, but data should be separated by comma

        usecols: list of integers. 
                 Numbers of the columns (starting from zero) containing the input data. It does not need to be increasing order, but the
                    order must match the list 'names' shown below. In addition, columns representing additional variables that are not needed by the script
                    can be skipped by ommiting their respective numbers in usecols and name in names (see below)

        names: list of strings
               These are the names of the variables following the order in usecols. These do not need to match the header in the files (the original
               header will be ignored)
               The default list is [  'date',  'u',  'v', 'w', 'Ts',  'co2', 'h2o',  'Tair', 'P' ]. Note that all these variables are required with exception of Tair
               Names and units must be as described below:
                    'date': time of sampling   (-)
                    'u': x-component velocity  (m/s)  
                    'v': y-component velocity  (m/s)
                    'w': x-component velocity  (m/s)
                    'Ts': sonic temperature    (Celcius)
                    'co2': co2 concentration   (mg/m3)   
                    'h2o': h2o concentration   (g/m3)    
                    'Tair': air temperatture   (Celcius) -- not required and can be omitted; Ts will be used to compute air temperature by the code
                    'P': pressure              (kPa)
                If co2 and h2o are in different units, they first need to be converted before any pre-processing is applied.
                If data is already pre-processed, additional inputs should be given ( ex., 'co2_p', 'h2o_p', 'w_p', .. ) 
                The description of all variables can be found in Partitioning.py

        skiprows: list of integers.
                  Contains the number of the initial rows to skip (starting from zero). All lines containing text, including header, must be skipped.
                  The default files (in RawData30min/) only contain text in the first row (header); thus, by default skiprows = [0].

        NAN: list of strings/numbers. 
            How missing values are represented in the data files. If missing values in your data files are represented differently, add to this list
            By default, the following strings, when present in the text file, are considered missing data ["NAN",-9999,'-9999', '#NA', 'NULL']

   # ----------------------------------------------------------------------------------------------------

   Additional information for site and processing details are required in the following dictionaries

        'siteDetails'
            hi: float;
                canopy height (m)
            zi: float 
                measurement height (m)
            freq: float 
                  sampling frequency (Hz)
            length: integer
                    length in minutes of each data file. Most commonly, 30 minutes blocks are used, but other sizes are accepted by the code
                    The example files accompanying this code are 30min blocks measured at 20Hz (36000 points each file)
            PreProcessing: boolean,
                           True: assumes that raw data are passed; data are pre-processed before partioning
                           False: assumes that data is already pre-processed and that the relevant variables, including turbulent fluctuations, are available

        'processing_args'
            density_correction: boolean
                           True: corrects data for external fluctuations (if open-gas analyzer is used)
                           False: does not correct for external fluctuations (closed or enclosed-gas analyzer)
            fluctuations: string 
                          Defines how turbulent fluctuations are computed. Two options are available
                               BA: block average
                               LD: linear detrending
            maxGapsInterpolate: integer 
                                Number of consecutive gaps to be interpolated in the high-frequency time series after quality control eliminated bad data
            RemainingData: float (0,100). 
                           The percentage of data that should be available (ignoring missing values) in order to proceed with partioning
            steadyness: boolean (optional)
                        If a statistic to check stationarity is computed.
                           True: computes statistic indicating if the time series is steady
                           False: does not compute statistic
                        This statistic is intended to help the user decide on the quality of the time series. 
                        The description of the method can be found in Partioning.py
            saveprocessed: boolean
                        True: saves processed data (after quality control and pre-processing) to folder ProcessedData/
                              Note that saving high-frequency processed data slowers the overal performance of the code
                              

    This script outputs a csv file with the estimates for all methods 

    part_file: string,
               Name of the text file that will store the final results
"""      

# *************************************************************************************************
# ******* INPUTS (NEED MODIFICATION) **************************************************************
listfiles = glob("RawData30min/*.csv")
usecols   = [        0,      1,      2,     3,      4,        6,       8,         11,         12     ]
names     = [   'date',    'u',    'v',   'w',   'Ts',    'co2',   'h2o',     'Tair',        'P'     ]  # * note that Tair could be omitted
skiprows  = [ 0 ]                                    
NAN       = ["NAN",-9999,'-9999', '#NA', 'NULL']     

siteDetails = {            'hi': 2.5, 
                           'zi': 4.0,
                         'freq': 20,
                       'length': 30,
                'PreProcessing': True}

processing_args = {'density_correction': True, # if True, density corrections are implemented during pre-processing
                         'fluctuations': 'LD', 
                   'maxGapsInterpolate': 5   , # Intervals of up to 5 missing values are filled by linear interpolation
                        'RemainingData': 95  , # Only proceed with partioning if 95% of initial data is available after pre-processing
                           'steadyness': True,
                        'saveprocessed': False}

part_file       = "ResultsPart.csv"

# ******* No modifications are needed below this line *********************************************
# *************************************************************************************************

list_dates   = []  
part_results = { 'ET': [],
                 'Fc': [],
                 'Ecec': [], 'Tcec': [], 'Pcec': [],  'Rcec': [], 'status_cec': [],
                 'Erea': [], 'Trea': [], 'Prea': [],  'Rrea': [], 'status_rea': [],
                 'Efvs': [], 'Tfvs': [], 'Pfvs': [],  'Rfvs': [], 'status_fvs': [],
                 'W_const_ppm'  :   [], 
                 'W_const_ratio': [], 
                 'W_linear'     : [], 
                 'W_sqrt'       : [], 
                 'W_opt'        : []
               } 

for n,filei in enumerate(sorted(listfiles)):
    print("\n-------------------------------")
    print("Reading file %d/%d, file : %s"%(n+1,len(listfiles), filei))
   
    # Read csv file and store data in a dataframe
    df       = pd.read_csv( filei, header=None, index_col=0, usecols=usecols, names = names, na_values=NAN, skiprows=skiprows)
    df.index = pd.to_datetime(df.index)

    if n == 0:
        print("\nPlease check units for first file (won't be asked again)")
        M = df.median()
        for _var, unit in [ ('u', 'm/s'), ('v', 'm/s'), ('w', 'm/s'), ('Ts', 'Celcius'), ('co2', 'mg/m3'), ('h2o', 'g/m3') ]:
            print( "      Median %s: %.3f %s"%(_var, M[_var], unit))
        input("\n Press any key to continue ....")
    
    # Create object
    part = Partitioning(     hi = siteDetails['hi'], 
                             zi = siteDetails['zi'], 
                           freq = siteDetails['freq'],
                         length = siteDetails['length'],
                             df = df, 
                         PreProcessing = siteDetails['PreProcessing'], 
                         argsQC = processing_args )
    
    # % of valid data after quality control and interpolation
    print("      Valid data: %.1f %%"%part.valid_data) 
    if part.valid_data < processing_args['RemainingData']:
        print("    Less than %s of the total data is available. Skipping file.")
        del df, part
        continue

    data_begin = df.index[0].strftime("%Y-%m-%d %H:%M" )

    if processing_args['saveprocessed']:
        part.data.to_csv("ProcessedData/processed-%s.csv"%data_begin.replace(" ", "_"))

    """
    Applying partitioning methods ---------------
    """
    list_dates.append( data_begin )

    # CEC method
    part.partCEC(H=0)
    part_results['Ecec'].append(part.fluxesCEC['E'])
    part_results['Tcec'].append(part.fluxesCEC['T'])
    part_results['Pcec'].append(part.fluxesCEC['P'])
    part_results['Rcec'].append(part.fluxesCEC['R'])
    part_results['status_cec'].append(part.fluxesCEC['status'])
    
    # total fluxes
    part_results['ET'].append(part.fluxesCEC['ET'])
    part_results['Fc'].append(part.fluxesCEC['Fc'])

    # MREA method
    part.partREA(H=0.0)
    part_results['Erea'].append(part.fluxesREA['E'])
    part_results['Trea'].append(part.fluxesREA['T'])
    part_results['Prea'].append(part.fluxesREA['P'])
    part_results['Rrea'].append(part.fluxesREA['R'])
    part_results['status_rea'].append(part.fluxesREA['status'])

    # Compute water-use efficiency
    # All models are implemented: 'const_ppm', "const_ratio", "linear", "sqrt", "opt"
    part.WaterUseEfficiency(ppath='C3')

    part_results['W_const_ppm'].append(part.wue['const_ppm'])
    part_results['W_const_ratio'].append(part.wue['const_ratio'])
    part_results['W_linear'].append(part.wue['linear'])
    part_results['W_sqrt'].append(part.wue['sqrt'])
    part_results['W_opt'].append(part.wue['opt'])

    # FVS method - must enter one water-use efficiency in kg_co2/kg_h2o
    part.partFVS( W = part.wue['const_ppm'])
    part_results['Efvs'].append(part.fluxesFVS['E'])
    part_results['Tfvs'].append(part.fluxesFVS['T'])
    part_results['Pfvs'].append(part.fluxesFVS['P'])
    part_results['Rfvs'].append(part.fluxesFVS['R'])
    part_results['status_fvs'].append(part.fluxesFVS['status'])
    
    # alternatively, compute FVS from all options of W
    # for _w in part.wue.keys():
    #     part.partFVS( W = part.wue[_w])
    #  

    plt.scatter( part.data['h2o'], part.data['co2'])
    plt.show()
    plt.close()
    del df, part

part_results = pd.DataFrame(part_results, index = list_dates)
part_results.to_csv( part_file)