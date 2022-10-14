
[![DOI](https://zenodo.org/badge/441544177.svg)](https://zenodo.org/badge/latestdoi/441544177)

---
# Contact information 

Author: Einara Zahn\
email: einaraz@princeton.edu, einara.zahn@gmail.com

A description of the methods included in this script can be found in [Zahn et al., 2021](https://www.sciencedirect.com/science/article/pii/S0168192321004767?via%3Dihub) "Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components"



---
## Description

1. Performs quality control and pre-processing of high frequency eddy-covariance data
   - Check for physically realistic values
   - Detection of outliers
   - Rotation of coordinates
   - Extraction of fluctuations
   - Density corrections for CO<sub>2</sub> and H<sub>2</sub>O measured by open-path gas analyzers ("instantaneous" WPL correction, based on the paper 
                <a href="https://link.springer.com/article/10.1007%2Fs10546-006-9105-1">Detto and Katul, 2007</a> )
   - Stationarity test
   - Gap filling
2. Implements three partitioning methods:
    1. Conditional Eddy Covariance (CEC)
         [Zahn et al., 2021](https://www.sciencedirect.com/science/article/pii/S0168192321004767?via%3Dihub) "Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components"            
    2.  Modified Relaxed Eddy Accumulation (MREA)
          [Thomas et al., 2008](https://www.sciencedirect.com/science/article/pii/S0168192308000737)
            "Estimating daytime subcanopy respiration from conditional sampling methods 
            applied to multi-scalar high frequency turbulence time series"
    3. Flux-Variance Similarity (FVS)
          [Scanlon and Sahu, 2008](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008WR006932)
            "Correlation-based flux partitioning of water vapor and carbon dioxide fluxes: 
            Method simplification and estimation of canopy water use efficiency" and 
          [Scanlon et al., 2019](https://www.sciencedirect.com/science/article/pii/S016819231930348X?via%3Dihub)
            "Correlation-based flux partitioning of water vapor and carbon dioxide fluxes: 
            Method simplification and estimation of canopy water use efficiency"
        
---
## Available files 

The following files are available:

  - **Partitioning.py** contains the class Partitioning. It has all necessary methods to first implement quality control and data pre-processing. It contains the three partitionong methods to partition ET and CO<sub>2</sub> fluxes into stomatal and non-stomatal components
  - **auxfunctions.py** contains auxiliary functions for pre-processing and to compute the water-use efficiency. Adapted from <a href="https://github.com/usda-ars-ussl/fluxpart">FluxPart</a>
  - **main.py** contains an example of how to use the script. Examples of files containing raw high-frequency data are included in the folder RawData30min.

---
## Format of input text files 

This script works with text files separated by commas.

The following variables are required by the code if raw high-frequency data is used as input (see the code for requirements to use pre-processed data as input):
  - index: date and time of acquisition.
  - w:  velocity in the z direction (m/s).
  - u: fluctuations of velocity in the x direction (m/s).
  - v: fluctuations of velocity in the y direction (m/s).
  - Ts:  sonic temperature (Celsius).
  - co2: carbon dioxide density (mg/m3).
  - h2o: water vapor density (g/m3).
  - P: pressure (kPa).

