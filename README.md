# ----------------------------------------------------------------------
# General information --------------------------------------------------

Author:  Einara Zahn
contact: einaraz@princeton.edu, einara.zahn@gmail.com

# ----------------------------------------------------------------------
# Description of files -------------------------------------------------

Partitioning.py - contains the class Partitioning - implements functions to
                  partition ET and CO2 fluxes into stomatal and non-stomatal components
auxfunctions.py - contains auxiliary functions to compute the water-use efficiency
main.py - shows an example of how to implement the partitioning methods. Three text files 
          with half-houly time series are available: '2018-04-09-1700.csv', '2018-04-09-1730.csv', '2018-04-09-1800.csv'

# ----------------------------------------------------------------------
# Format of input data -------------------------------------------------

This code assumes that quality control (e.g., removal of outliers) and post-processing (coordinate
rotation, density correction, extraction of fluctuations, etc) were previously implemented.

To implement CEC and MREA, only the time series of fluctuations of carbon dioxide density (c_p),
water vapor density (q_p) and vertical velocity (w_p) are necessary. FVS requires the water-use efficiency,
which can be computed given that additional variables are given to the code (list below)

Three times series in the files '2018-04-09-1700.csv', '2018-04-09-1730.csv', '2018-04-09-1800.csv'
are included in this folder as examples. The following pre-processing was done:
    --> Quality control (removing outliers, despiking, flags of instruments, etc)
    --> Rotation of coordinates (double rotation) for velocity components u, v, w measured by csat
    --> Density corrections for instantaneous fluctuations of co2 (c_p) and h2o (q_p) ("instantaneous" WPL correction) based on the paper 
                Detto, M. and Katul, G. G., 2007. "Simplified expressions for adjusting higher-order turbulent statistics obtained from open path gas analyzers"
                Boundary-Layer Meteorology, 10.1007/s10546-006-9105-1
    --> Primed quantities ("_p") are fluctuations computed by extracting the linear trend
    --> Air temperature (T) and virtual temperature (Tv) computed from the sonic temperature (Ts)


Their columns contain the following data that are necessary to implement the partitioning methods and to compute
water-use efficiency (W). Alternativaly, W can be given to the code.
   index - datetime
   w_p  - fluctuations of velocity in the z direction (m/s)
   u_p  - fluctuations of velocity in the x direction (m/s)
   v_p  - fluctuations of velocity in the y direction (m/s)
   P    - pressure (kPa)
   co2  - carbon dioxide density (mg/m3)
   h2o  - water vapor density (g/m3)
   T    - air temperature (Celsius)
   Tv   - virtual temperature (Celsius)
   c_p  - fluctuations of carbon dioxide density (mg/m3) - WPL corrected and linear detrended
   q_p  - fluctuations of water vapor density (g/m3) - WPL corrected and linear detrended
   T_p  - fluctuations of air temperature (Celsius)
   Tv_p - fluctuations of virtual temperature (Celsius)


# Available Partitioning methods ---------------------------------------

--> Conditional Eddy Covariance (CEC):
     Zahn et al., 2021, Agr. For. Met., "Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide
        Fluxes into Ground and Plant Components"
                        
--> Modified Relaxed Eddy Accumulation (MREA)
     Thomas et al., 2008 (Agr For Met)
        "Estimating daytime subcanopy respiration from conditional sampling methods 
        applied to multi-scalar high frequency turbulence time series"
        https://www.sciencedirect.com/science/article/pii/S0168192308000737
        
--> Flux-Variance Similarity (FVS)
      Scanlon et al., 2019, Agr. For. Met.
        "Correlation-basedflux partitioning of water vapor and carbon dioxidefluxes: 
        Method simplification and estimation of canopy water use efficiency"
# PartitioningMethods