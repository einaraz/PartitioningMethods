---
# Contact information 

Author: Einara Zahn
email: einaraz@princeton.edu, einara.zahn@gmail.com

To cite this code

[![DOI](https://zenodo.org/badge/441544177.svg)](https://zenodo.org/badge/latestdoi/441544177)



---
## Description

The class Partitioning implements three partitioning methods:

1. Conditional Eddy Covariance (CEC):
     Zahn et al., 2021, Agr. For. Met., "Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide
        Fluxes into Ground and Plant Components"
                        
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

<ul>
  <li> Partitioning.py contains the class Partitioning. It implements three partitionong methods to
       partition ET and CO<sub>2</sub> fluxes into stomatal and non-stomatal components
  <li> auxfunctions.py - contains auxiliary functions to compute the water-use efficiency. Adapted from <a href="https://github.com/usda-ars-ussl/fluxpart">FluxPart</a>
  <li> main.py - contains an example of how to implement the partitioning methods. Three text files 
          with half-houly time series are available: '2018-04-09-1700.csv', '2018-04-09-1730.csv', '2018-04-09-1800.csv'
</ul>

---
## Format of input data 

**This code assumes that quality control (e.g., removal of outliers) and post-processing (coordinate
rotation, density correction, extraction of fluctuations, etc) were previously implemented.**

To implement CEC and MREA, only the time series of fluctuations of carbon dioxide density (c_p),
water vapor density (q_p) and vertical velocity (w_p) are necessary. FVS requires the water-use efficiency,
which can be computed given that additional variables are given to the code (list below).

Three times series in the files '2018-04-09-1700.csv', '2018-04-09-1730.csv', '2018-04-09-1800.csv'
are included in this folder as examples. The following pre processing was done:
<ul>
    <li> Quality control (removing outliers, despiking, flags of instruments, etc)
    <li> Rotation of coordinates (double rotation) for velocity components u, v, w measured by sonic anemometer
    <li> Density corrections for instantaneous fluctuations of CO<sub>2</sub> (c_p) and H<sub>2</sub>O (q_p) ("instantaneous" WPL correction) based on the paper 
                <a href="https://link.springer.com/article/10.1007%2Fs10546-006-9105-1">Detto and Katul, 2007</a> </li>
    <li> Primed quantities ("_p") are fluctuations computed by extracting the linear trend
    <li> Air temperature (T) and virtual temperature (Tv) computed from the sonic temperature (Ts)
</ul>

Their columns contain the following variables that are necessary to implement the partitioning methods and to compute
water-use efficiency (W). Alternativaly, W can be given to the code.
   <ul>
     <li> index - date and time of acquisition
     <li> w_p  - fluctuations of velocity in the z direction (m/s)
     <li> u_p  - fluctuations of velocity in the x direction (m/s)
     <li> v_p  - fluctuations of velocity in the y direction (m/s)
     <li> P    - pressure (kPa)
     <li> co2  - carbon dioxide density (mg/m3)
     <li> h2o  - water vapor density (g/m3)
     <li> T    - air temperature (Celsius)
     <li> Tv   - virtual temperature (Celsius)
     <li> c_p  - fluctuations of carbon dioxide density (mg/m3) - corrected for external density fluctuations and linear detrended
     <li>  q_p  - fluctuations of water vapor density (g/m3) - corrected for external density fluctuations and linear detrended
    <li>  T_p  - fluctuations of air temperature (Celsius)
    <li> Tv_p - fluctuations of virtual temperature (Celsius)
   </ul>
       
