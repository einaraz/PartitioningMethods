import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from partitioning import Partitioning
from glob import glob
from pprint import pprint

# Location of the files with the raw data as a text file
listfiles   = glob("RawData30min/*.csv")

# Information about the measurement site and data including units: needs to be modified to your case
siteDetails = { "hi": 2.5 ,  # Canopy mean height in meters
                "zi": 4.0 ,  # EC measurement height in meters
              "freq": 20  ,  # EC measurement frequency in Hz
            "length": 30  ,  # length of data file in minutes
     "PreProcessing": True}  # If True, input is raw data and pre-processing is applied before partitioning (e.g. density corrections and fluctuations are computed)

# Information about the data processing
# All options have default values assuming that raw data is provided and measured by an open-path gas analyzer and a 3D sonic anemometer
processing_args = {
    "density_correction": True,  # If True, density corrections are implemented during pre-processing (depends on type of gas analyzer used)
    "fluctuations": "LD",        # If "LD", linear detrending is applied to the data. BA (block averaging) and FL (filter low freqencies) are also available
    "filtercut": 5,              # Cutoff timescale to filter low frequencies (in minutes). Needed when FL is selected as fluctuation method
    "maxGapsInterpolate": 5,     # Intervals of up to 5 missing values are filled by linear interpolation
    "RemainingData": 95,         # Only proceed with partioning if 95% of initial data is available after pre-processing
    "steadyness": False,          # Compute statistic to check stationarity (will not delete data based on this test, only print the results)
    "saveprocessed": False,      # If True, save the intermediate processed data including all corrections and fluctuations
    # For closed-path gas analyzers, set "density_correction" to False and use the following to correct the time lag
    #"time_lag_correction": False, # If True, a time lag correction is applied to the CO2 and H2O time series relative to the W time series
    #"max_lag_seconds": 5,         # Maximum time lag in seconds to consider for cross correlation analyses
    #"saveplotlag": False,         # If True, saves a plot of the cross-correlation function between the CO2 and H2O time series with respect to the W time series
    #'type_lag': 'positive',       # Specifies the type of lag to consider ('negative', 'positive', or 'both')
}

dates = []
part_results = {}

for n, filei in enumerate(sorted(listfiles)):
    print("Reading file %d/%d, file : %s" % (n + 1, len(listfiles), filei))

    # Ensure that the header of the file constains the following variables: "date","u","v","w","Ts","co2","h2o","Tair","P"
    # Before processing the data, ensure that the units are correct (convert if necessary)
    # "date": [yyyy-mm-dd HH:MM:SS]
    #  "u","v","w": [m/s]
    #         "Ts": [oC]
    #        "co2": [mg_CO2/m3]
    #        "h2o": [g_h2o/m3]
    #       "Tair": [oC]
    #          "P": [kPa]
    # Missing values should be coded as NaN
    df = pd.read_csv(
        filei,
        header=None,
        index_col=0,
        usecols=[0, 1, 2, 3, 4, 6, 8, 11, 12],
        names=["date","u","v","w","Ts","co2","h2o","Tair","P"],
        na_values=["NAN", -9999, "-9999", "#NA", "NULL"],
        skiprows=[0] )
    df.index = pd.to_datetime(df.index)

    # Create a partitioning object
    part = Partitioning(
                hi=siteDetails["hi"],
                zi=siteDetails["zi"],
                freq=siteDetails["freq"],
                length=siteDetails["length"],
                df=df,
                PreProcessing=siteDetails["PreProcessing"],
                argsQC=processing_args)

    # Plot time series of fluctuations
    #fig, ax = plt.subplots(4, 1, figsize=(12, 6))
    #ax[0].plot(part.data.index, part.data["co2_p"],)
    #ax[1].plot(part.data.index, part.data["h2o_p"],)
    #ax[2].plot(part.data.index, part.data["w_p"],  )
    #ax[3].plot(part.data.index, part.data["Ts_p"], )
    #ax[0].set_ylabel("CO2")
    #ax[1].set_ylabel("H2O")
    #ax[2].set_ylabel("w")
    #ax[3].set_ylabel("Ts")
    #plt.show()

    if processing_args["saveprocessed"]:
        part.data.to_csv("ProcessedData/processed-%s.csv" % (n+1) )

    """
    Applying partitioning methods ---------------
    """
    dates.append(df.index[0])
    
    # ------------------------------------
    # Total fluxes and statistics
    part.TurbulentStats()
    # print(part.turbstats)
    
    # Save the results to a dictionary (use .magnitude to get the value and .units to get the units)
    if "ustar" not in part_results.keys():
        for key in part.turbstats.keys():
            part_results[key] = []
    for key in part.turbstats.keys():
        part_results[key].append(part.turbstats[key].magnitude)
    
    # If steadyness is True, results can also be acessed here
    # print(part.FokenStatTest())
    
    # ------------------------------------
    # CEC method
    part.partCEC()
    # print(part.fluxesCEC)
    
    # Save the results in a dictionary (use .magnitude to get the value and .units to get the units)
    if "Ecec" not in part_results.keys():
        for key in ['Ecec', 'Tcec', 'Pcec', 'Rcec', 'statuscec']:
            part_results[key] = []
            
    for key in ['Ecec', 'Tcec', 'Pcec', 'Rcec']:
        part_results[key].append(part.fluxesCEC[key].magnitude)
    part_results["statuscec"].append(part.fluxesCEC["statuscec"])

    # MREA method ------------------------
    part.partREA()
    #print(part.fluxesREA)
    
    # Save the results in a dictionary (use .magnitude to get the value and .units to get the units)
    if "Emrea" not in part_results.keys():
        for key in ['Emrea', 'Tmrea', 'Pmrea', 'Rmrea', 'statusmrea']:
            part_results[key] = []
    for key in ['Emrea', 'Tmrea', 'Pmrea', 'Rmrea']:
        part_results[key].append(part.fluxesREA[key].magnitude)
    part_results["statusmrea"].append(part.fluxesREA["statusmrea"])

    # Water-use efficiency ---------------
    # All models are implemented: 'const_ppm', "const_ratio", "linear", "sqrt", "opt"
    part.WaterUseEfficiency(ppath="C3")
    #print(part.wue)
    
    # Save the results to a dictionary
    if "W_const_ppm" not in part_results.keys():
        for key in part.wue.keys():
            part_results[f"W_{key}"] = []
    for key in part.wue.keys():
        part_results[f"W_{key}"].append(part.wue[key])

    # FVS method --------------------------
    # - loop over all water-use efficiencies
    for _w in part.wue.keys():
        part.partFVS( W = part.wue[_w])
        
        # Save the results in a dictionary (use .magnitude to get the value and .units to get the units)
        if f"Efvs{_w}" not in part_results.keys():
             for key in ['Efvs', 'Tfvs', 'Pfvs', 'Rfvs', 'statusfvs']:
                 part_results[f"{key}{_w}"] = []
        for key in ['Efvs', 'Tfvs', 'Pfvs', 'Rfvs']:
            part_results[f"{key}{_w}"].append(part.fluxesFVS[key].magnitude)
        part_results[f"statusfvs{_w}"].append(part.fluxesFVS["statusfvs"])

    # FVS method - select one water-use efficiency model or use an external value kg_co2/kg_h2o
    #part.partFVS(W=part.wue["const_ppm"])
    #print(part.fluxesFVS)

    # CECw method
    for _w in part.wue.keys():
        part.partCECw(W=part.wue[_w])
        
        # Save the results in a dictionary (use .magnitude to get the value and .units to get the units)
        if f"Ececw{_w}" not in part_results.keys():
            for key in ['Ececw', 'Tcecw', 'Pcecw', 'Rcecw']:
                part_results[f"{key}{_w}"] = []
        for key in ['Ececw', 'Tcecw', 'Pcecw', 'Rcecw']:
            part_results[f"{key}{_w}"].append(part.fluxesCECw[key].magnitude)
            
    # CEA method
    part.partCEA()
    
    # Save the results in a dictionary (use .magnitude to get the value and .units to get the units)
    if "Ecea" not in part_results.keys():
        for key in ['Ecea', 'Tcea', 'Pcea', 'Rcea', 'statuscea']:
            part_results[key] = []
    for key in ['Ecea', 'Tcea', 'Pcea', 'Rcea']:
        part_results[key].append(part.fluxesCEA[key].magnitude)
    part_results["statuscea"].append(part.fluxesCEA["statuscea"])
    
    del df, part

part_results = pd.DataFrame(part_results, index=dates)
print(part_results)
part_results.to_csv("PartitioningResults.csv")

