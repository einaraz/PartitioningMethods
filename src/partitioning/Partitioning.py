# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from .auxfunctions import ci_const_ppm, cica_const_ratio, cica_linear, cica_sqrt, \
                         sat_vapor_press, vapor_press_deficit, vapor_press_deficit_mass, fes, LinearDetrend, \
                         Stats5min, find_spikes

"""
Author: Einara Zahn <einaraz@princeton.edu>, <einara.zahn@gmail.com>
Last update: Dec 24, 2021; Oct 12 2022             
"""

class Constants:
    """
    Define constants
    """
    Rd         = 287                   # gas constant of dry air - J/K/kg
    Lv         = 2.453*10**6           # latent heat - J/kg
    rho_w      = 1000                  # density of water - kg/m3
    VON_KARMAN = 0.4                   # von Karman constant
    MWdryair   = 0.0289645             # Molecular weight of dry air
    MWvapor    = 0.018016              # Molecular weight of vapor
    MWco2      = 0.044010              # Molecular weight of carbon dioxide
    Rco2       = 8.3144598/MWco2       # Gas constant for CO2
    Rvapor     = 8.3144598/MWvapor     # Gas constant for water vapor
    diff_ratio = 1.6                   # Ratio of diffusivities water/co2
    g          = 9.8160                # gravity m/s2
    # Constants used for water use efficiency (see Partitioning.WaterUseEfficiency for definitions)
    wue_constants = { "C3": {"const_ppm": 280, "const_ratio": 0.70,        "linear": (1.0, 1.6e-4) , "sqrt": (22e-9) },
                      "C4": {"const_ppm": 130, "const_ratio": 0.44,        "linear": (1.0, 2.7e-4) , "sqrt":  np.nan } }

class Partitioning(object):
    """
    Inputs:
        hi - canopy height (m)
        zi - eddy covariance measurement height (m)
        df - dataframe with data (e.g., 30min, but any length works), each variable in a column
             If raw data is used, pre-processing is first implemented, following the steps:
                --> Quality control (removing outliers, despiking, flags of instruments, etc)
                --> Rotation of coordinates (double rotation) for velocity components u, v, w measured by csat
                --> Density corrections for instantaneous fluctuations of co2 (c_p) and h2o (q_p) measured by open-gas analyser ("instantaneous" WPL correction) based on the paper 
                            Detto, M. and Katul, G. G., 2007. "Simplified expressions for adjusting higher-order turbulent statistics obtained from open path gas analyzers"
                            Boundary-Layer Meteorology, 10.1007/s10546-006-9105-1
                --> Turbulent fluctuations, here denoted as primed quantities ("_p"), are computed 
                --> Air temperature (T) and virtual temperature (Tv) computed from the sonic temperature (Ts)
             
            Raw data requires the following variables and units:
                index - datetime
                w     - velocity in the z direction (m/s)
                u     - velocity in the x direction (m/s)
                v     - velocity in the y direction (m/s)
                Ts    - sonic temperature (Celsius)
                P     - pressure (kPa)
                co2   - carbon dioxide density (mg/m3)
                h2o   - water vapor density (g/m3)
            After pre-processing, the following additional variables are created (***)
                w_p  - fluctuations of velocity in the z direction (m/s)
                u_p  - fluctuations of velocity in the x direction (m/s)
                v_p  - fluctuations of velocity in the y direction (m/s)
                T    - thermodynamic air temperature (Celsius)
                Tv   - virtual temperature (Celsius)
                c_p  - fluctuations of carbon dioxide density (mg/m3) - (corrected for external densities (WPL) if needed)
                q_p  - fluctuations of water vapor density (g/m3)     - (corrected for external densities (WPL) if needed)
                T_p  - fluctuations of air temperature (Celsius)
                Tv_p - fluctuations of virtual temperature (Celsius)

        PreProcessing - boolean indicating if pre-processing is necessary. If True, all pre-processing steps are implemented to
                        raw data; if False, pre-processing is ignored and partitioning is imediatelly applied. In this case, 
                        the input files must contain all variables listed above (***)
        argsQC - dictionary. Contains options to be used during pre-processing regarding fluctuation extraction and if 
                        density corrections are necessary.
                 Keys:
                 density_correction - boolean. True if density corrections are necessary (open gas analyzer); 
                                               False (closed or enclosed gas analyzer)
                 fluctuations - string. Describes the type of operation used to extract fluctuations
                                  'BA': block average
                                  'LD': Linear detrending
                 maxGapsInterpolate - integer. Number of consecutive gaps that will be interpolated 
                 RemainingData      - integer (0 - 100). Percentage of the time series that should have remained after pre-processing
                                       if less than this quantity, partitioning is not implemented
    The following partinioning methods are available (references for each method below)
        - Conditional Eddy Covariance (CEC)
        - Modified Relaxed Eddy Accumulation (MREA)
        - Flux Variance Similarity (FVS)

    * Note that CEC and MREA only need time series of w_p, c_p, q_p
      The remaining quantities (e.g., P, T, Tv, etc) are only needed if the
      water use efficiency is computed for the FVS method. Alternatively, an external WUE can be
      used; in this case, FVS will only need time series of w_p, c_p, q_p
    
    """
    def __init__(self, hi, zi, freq, length, df, PreProcessing, argsQC):
        self.hi     = hi                # Canopy height (m)
        self.zi     = zi                # Measurement height (m)
        self.data   = df                # pandas dataframe containing data
        self.freq   = freq
        self.length = length
        
        if PreProcessing:
            self._checkPhysicalBounds()
            self._despike()        
            self._rotation()
            self._fluctuations(method=argsQC['fluctuations'])
            if argsQC['density_correction']: 
                self._densityCorrections(method=argsQC['fluctuations'])
            self._fillGaps(argsQC['maxGapsInterpolate'])
            if argsQC['steadyness']: 
                self._steadynessTest()
        self._checkMissingdata(argsQC['RemainingData'])

    def _checkMissingdata(self, percData):
        """
        Checks how many missing points are present
        Only accepts periods when valid data points >= percData

        Input
           percData: integer (0, 100). Percentage of the data that needs to be valid (i.e., excluding gaps) 
                                       in order to implement partitioning. If less than percData is available,
                                       the entire half-hour period is discarded
        Computes the percentage of valid data and stores in self.valid_data
        """
        maxNAN, indMAX     = self.data.isnull().sum().max(), self.data.isnull().sum().idxmax()
        total_size         = self.freq * self.length * 60  # total number of points in period
        self.valid_data    = ( (total_size - maxNAN) / total_size ) * 100
        self.data.dropna( inplace=True)

    def _checkPhysicalBounds(self):
        """
        Set to NaN those values outside a physical realistic range
        If additional variables other than the required are passed to the code,
           their physical bounds need to be added to the dictionary _bounds
           Units must match those of the input data
        """
        _bounds = {  'u': (-20, 20), 'v': (-20, 20), 'w': (-20, 20),  # m/s
                    'Ts': (-10, 50),  # Celsius
                   'co2': (0, 1500),  # mg/m3
                   'h2o': (0,  40),   #  g/m3
                     'P': (60, 150),  # kPa
                  }

        for _var in self.data.columns:
            if _var in _bounds.keys():
                self.data.loc[ (self.data[_var] < _bounds[_var][0] ) | (self.data[_var] > _bounds[_var][1]), _var ] = np.nan

    def _despike(self):
        """
        Replace outliers with NaN values
        Points are only considered outliers if no more than 8 points in sequence are above a threshold (see find_spikes in auxfunctions.py)
        Implements the test described in section 3.4 of 
            E. Zahn, T. L. Chor, N. L. Dias, A Simple Methodology for Quality Control of Micrometeorological Datasets, 
            American Journal of Environmental Engineering, Vol. 6 No. 4A, 2016, pp. 135-142. doi: 10.5923/s.ajee.201601.20
        """

        aux      = self.data[["co2", "h2o", "Ts", "w", "u", "v"]].copy()
        tt       = np.arange(aux["co2"].index.size)

        # 1st: linear detrend time series ------------------
        for _var in ["co2", "h2o", "Ts", "w", "u", "v"]:
            aux[_var] = LinearDetrend( tt, aux[_var].values)
        del tt

        # 2nd: Separate into 2-min windows -----------------
        TwoMinGroups = aux.groupby(pd.Grouper(freq='5Min'))
        TwoMinGroups = [ TwoMinGroups.get_group(x) for x in TwoMinGroups.groups if TwoMinGroups.get_group(x).index.size > 10]
        del aux

        for i in range(len(TwoMinGroups)):
            aux_group = TwoMinGroups[i].copy()
            getSpikes = aux_group.apply(find_spikes)
            
            for _var in ["co2", "h2o", "Ts", "w", "u", "v"]:
                for vdate in getSpikes[_var]:
                    self.data[_var].loc[vdate] = np.nan
        del TwoMinGroups

    def _rotation(self):
        """
        Performs rotation of coordinates using the double rotation method
        Overwrites the velocity field (u,v,w) with the rotated coordinates
        References: 
        """
        aux   = self.data[["u", "v", "w"]].copy()
        Umean = aux.mean()        
        # Calculating the angles between mean velocities
        hspeed = np.sqrt( Umean["u"]**2. + Umean["v"]**2.)
        alfax  = math.atan2(Umean["v"], Umean["u"])
        alfaz  = math.atan2(Umean["w"], hspeed)  
        # Rotating coordinates 
        aux["u_new"] =  math.cos(alfax)*math.cos(alfaz)*aux["u"]+math.sin(alfax)*math.cos(alfaz)*aux["v"]+math.sin(alfaz)*aux["w"]
        aux["v_new"] = -math.sin(alfax)*aux["u"]+math.cos(alfax)*aux["v"]
        aux["w_new"] = -math.cos(alfax)*math.sin(alfaz)*aux["u"]-math.sin(alfax)*math.sin(alfaz)*aux["v"] +math.cos(alfaz)*aux["w"]
        # Update rotated velocities in dataframe
        self.data["w"] = aux["w_new"].copy()
        self.data["u"] = aux["u_new"].copy()
        self.data["v"] = aux["v_new"].copy()
        del aux, hspeed, alfax, alfaz

    def _fluctuations(self, method):
        """
        Computes turbulent fluctuations, x' = x - X, where X is the average
        Only variables required by the partitioning algorithms are included 
        method to compute X: 
            BA: Block average
            LD: linear detrending
        Add the time series of fluctuations (variable_name + _p ) to the dataframe  
        """
        Lvars = [ 'u', 'v', 'w', 'co2', 'h2o', 'Ts']
        
        if method == "LD":
            tt = np.arange(self.data.index.size)
            for ii,_var in enumerate(Lvars):
                self.data[_var+"_p"] = LinearDetrend( tt, self.data[_var].values)
            del tt
        elif method == "BA":
            for ii,_var in enumerate(Lvars):
                self.data[_var+"_p"] = self.data[_var] - self.data[_var].mean()
        else:
            raise TypeError("Method to extract fluctuations must be 'LD' or 'BA'. ")

    def _densityCorrections(self, method):
        """
        Apply density correction to the fluctuations of co2 (co2_p) and h2o (h2o_p)
        following Detto, M. and Katul, G. G., 2007. "Simplified expressions for adjusting higher-order turbulent statistics obtained from open path gas analyzers"
                            Boundary-Layer Meteorology, 10.1007/s10546-006-9105-1
        Note that it is only necessary when co2 and h2o were measured by an open gas analyzer and their output are mass/molar densities (ex, mg/m3)
        """
        # Calculate air density------------------------------------------------
        Rd = Constants.Rd                                                                      # J/kg.K                                              
        self.data["rho_moist_air"] = 1000*self.data["P"]/(Rd*(273.15 + self.data["Ts"] ))      # mean density of moist air [kg/m3] *** assume Ts is the same as Tv ***
        self.data["rho_dry_air"]   = self.data["rho_moist_air"] - self.data["h2o"]*10**-3      # density of dry air [kg/m3] 
        # Obtain termodynamic and virtual temperatures ------------------------
        q               =  self.data["h2o"]*10**-3/self.data["rho_dry_air"]      # Instantaneous mixing ratio kg/kg
        self.data["T"]  =  (self.data["Ts"] + 273.15 )/(1.0 + 0.51*q) - 273.15   # termodynamic temperature from sonic temperature [C]
        self.data["Tv"] =  (self.data["T"]   + 273.15 )*(1.0 + 0.61*q) - 273.15  # virtual temperature from termo temperature [C]

        # We also need the fluctuations of temperature ------------------------
        if method == "LD": 
            self.data["T_p"]   = LinearDetrend( np.arange(self.data.index.size), self.data["T"].values)
            self.data["Tv_p"]  = LinearDetrend( np.arange(self.data.index.size), self.data["Tv"].values)
        else: 
            self.data["T_p"]   =  self.data["T"] - self.data["T"].mean()
            self.data["Tv_p"]  = self.data["Tv"] - self.data["Tv"].mean()
        meanT = self.data["T"].mean()                                            # mean real temperature [C]

        # Aditional variables --------------------------------------------------
        mu       = Constants.MWdryair/Constants.MWvapor            # ratio of dry air mass to water vapor mass
        mean_co2 = self.data["co2"].mean()*10**-6                  # mean co2 [kg/m3]
        mean_h2o = self.data["h2o"].mean()*10**-3                  # mean h2o [kg/m3]
        sigmaq   = (mean_h2o/self.data["rho_dry_air"].mean())      # mixing ratio [kg_wv/kg_air]
        sigmac   = (mean_co2/self.data["rho_dry_air"].mean())      # mixing ratio [kg_co2/kg_air]
        # Finally, apply the corrections ---------------------------------------
        self.data["co2_p"] = self.data["co2_p"] + ( mu*sigmac*self.data["h2o_p"]*10**-3 + mean_co2*(1.0+mu*sigmaq)*self.data["T_p"]/(meanT+273.15) )*10**6 # [mg/m3]
        self.data["h2o_p"] = self.data["h2o_p"] + ( mu*sigmaq*self.data["h2o_p"]*10**-3 + mean_h2o*(1.0+mu*sigmaq)*self.data["T_p"]/(meanT+273.15) )*1000  # [ g/m3]
        del mu, mean_co2, mean_h2o, sigmaq, sigmac, q

    def _fillGaps(self, maxGaps):
        """
        Fill gaps (nan) values in time series using linear interpolation
        It's recommended that only small gaps be interpolated.

        Input:
           maxGaps: integer > 0. Number of consecutive missing gaps that can be interpolated.
        """
        if maxGaps > 20:
            raise TypeError("Too many consecutive points to be interpolated. Consider a smaller gaps.")

        self.data.interpolate(method='linear', limit= maxGaps, limit_direction='both', inplace=True)

    def _steadynessTest(self):
        """
        Implements a stationarity test described in section 5 of
            Thomas Foken and B. Wichura, "Tools for quality assessment of surface-based flux measurements"
                Agricultural and Forest Meteorology, Volume 78, Issues 1–2, 1996, Pages 83-105.

            In computes
                 stat = | (average_cov_5min - cov_30min) / cov_30min | * 100 %, where cov is the covariance 
                           between any two variables

        Foken argues ** that steady state conditions can be assume if stat < 30 %;
        This variable can be used as a criterion for data quality and its compliance with EC requirements (steadyness)
        ** Micrometeorology (https://doi.org/10.1007/978-3-540-74666-9), p. 175

        Creates a dictionary with the steadyness statistics (in %) for variances and covariances
        self.FokenStatTest = { 
                              'wc': statistic for w'c' (total CO2 flux) 
                              'wq': statistic for w'q' (total H2O flux)
                              'wT': statistic for w'T' (sonic temperature flux) 
                              'ww': statistic for w'w' (variance of w)
                              'cc': statistic for c'c' (variance of co2)
                              'qq': statistic for q'q' (variance of h2o)
                              'tt': statistic for t't' (variance of sonic temperature)
                               }
        """
        # Five minute window statistics -------------------------
        stats5min = self.data.groupby(pd.Grouper(freq='5Min')).apply(Stats5min).dropna()
        aver_5min = stats5min.mean()
        # Statistic for entire window (i.e., 30 min) ------------
        cov_all   = self.data.cov()
        var_all   = self.data.var()
        stats_all = pd.Series([ cov_all['w_p']['co2_p'], cov_all['w_p']['h2o_p'], cov_all['w_p']['Ts_p'], 
                                var_all['w_p'],   var_all['co2_p'],   var_all['h2o_p'],   var_all['Ts_p'] ], index=[ 'wc', 'wq', 'wt', 'ww', 'cc', 'qq', 'tt'])
        # Compare 30-min to 5-min windows -----------------------
        stat_fk            = dict(abs( (stats_all - aver_5min)/stats_all )*100)
        self.FokenStatTest = { 'fkstat_%s'%svar: stat_fk[svar] for svar in ['wc', 'wq', 'wt', 'ww', 'cc', 'qq', 'tt']  }

    def WaterUseEfficiency(self, ppath='C3'):
        """
        Calculates water use efficiency in kg_co2/kg_h2o
        
        Main references:
        
        Scanlon and Sahu 2008, Water Resources Research
                       "On the correlation structure of water vapor and carbon dioxide in 
                        the atmospheric surface layer: A basis for flux partitioning"
        
        Parts of the code were adapted from Skaggs et al. 2018, Agr For Met
                       "Fluxpart: Open source software for partitioning carbon dioxide and water vapor fluxes" 
                        https://github.com/usda-ars-ussl/fluxpart
                        
        Optimization model for W from Scanlon et al., 2019, Agr. For. Met.
                       "Correlation-basedflux partitioning of water vapor and carbon dioxidefluxes: 
                       Method simplification and estimation of canopy water use efficiency"
        
        Input:
            ppath - C3 or C4 - type of photosynthesis
        

        ----------------------------------------------------------------
        Computes the water use efficiency (eq A1 in Scanlon and Sahu, 2008)
            wue = 0.65 * (c_c - c_s) / (q_c - q_s)
            
            c_c (kg/m3) and q_c (kg/m3) are near canopy concentrations of co2 and h2o:
                --> estimated from log profiles (eq A2a in Scanlon and Sahu, 2008)
            c_s (kg/m3) and q_s (kg/m3) are stomata concentrations of co2 and h2o
                --> q_s is assumed to be at saturation
                --> c_s is parameterized from different models (Skaggs et al, 2018; Scanlon et al, 2019)
            
        The following models for c_s are implemented:
        
        const_ppm:
            --> Concentrations in kg/m3 are computed from a constant value in ppm
                Values from Campbell and Norman, 1998, p. 150
                Campbell, G.  S. and Norman, J. M. (1998). 
                An Introduction to Environmental Biophysics. Springer, New York, NY.
            c_s = 280 ppm (C3 plants)
            c_s = 130 ppm (C4 plants)
            
    
        const_ratio
            --> The ratio of near canopy and stomata co2 concentrations is assumed constant (c_s/c_c = constant)
                Constants from Sinclair, T. R., Tanner, C. B., and Bennett, J. M. (1984). 
                Water-use efficiency in crop production. BioScience, 34(1):36–40
                
            c_s/c_c = 0.70 for C3 plants
            c_s/c_c = 0.44 for C4 plants
    
    
        linear
            ---> The ratio of near canopy and stomata co2 concentrations is a linear function of vpd
                 Based on the results of Morison, J. I. L. and Gifford, R. M. (1983).  
                 Stomatal sensitivity to carbon dioxide and humidity. Plant Physiology, 71(4):789–796.
                 Estimated constants from Skaggs et al (2018)
                 
            c_s/c_c = a - b * D
                a, b = 1, 1.6*10-4 Pa-1  for C3 plants
                a, b = 1, 2.7*10-4 Pa-1  for C4 plants
                D (Pa) is vapor pressure deficit based on leaf-temperature
        'sqrt'
            ---> The ratio of near canopy and stomata co2 concentrations is proportional
                 to the 1/2 power of vpd
                 Model by Katul, G. G., Palmroth, S., and Oren, R. (2009). 
                 Leaf stomatal responses to vapour pressure deficit under current and co2-enriched atmosphere 
                 explained by the economics of gas exchange. Plant, Cell & Environment, 32(8):968–979.
  
                c_s/c_c = 1 - sqrt(1.6 * lambda *  D / c_c)
            
            lambda = 22e-9 kg-CO2 / m^3 / Pa for C3 plants  (from Skaggs et al, 2018)
            Not available for C4 plants
    
        'opt'
            Optimization model proposed by Scanlon et al (2019)
            Does not need extra parameters        
            Only available for C3 plants
        """
        
        # Create a copy of the dataframe with variables that will be needed
        aux = self.data.copy()
        
        # Create dictionary that will store water use efficiency from different methods
        self.wue = {"const_ppm": np.nan , "const_ratio": np.nan, "linear": np.nan, "sqrt": np.nan , "opt": np.nan } 
        
        # Statistics  -------------------- 
        matrixCov = aux.cov()                                                  # Covariance matrix
        varq = matrixCov["h2o_p"]["h2o_p"]*10**-6                              # variance of h2o fluctuations (kg/m3)^2
        varc = matrixCov["co2_p"]["co2_p"]*10**-12                             # variance of co2 fluctuations (kg/m3)^2
        sigmac = varc**0.5                                                     # Standard deviation of co2 [kg/m3]
        sigmaq = varq**0.5                                                     # Standard deviation of h2o [kg/m3]
        corr_qc = np.corrcoef(aux["h2o_p"].values, aux["co2_p"].values)[1,0]   # correlation coefficient between q and c
        cov_wq = matrixCov["w_p"]["h2o_p"]*10**-3                              # covariance of w and q (kg/m^2/s);
        cov_wc = matrixCov["w_p"]["co2_p"]*10**-6                              # covariance of w and c (kg/m^2/s);

        # Mean variables and parameterizations ------------
        leaf_T = aux["T"].mean() + 273.15                                                            # set leaf temperature == air temperature (Kelvin)
        P = aux["P"].mean()*10**3                                                                    # mean atmospheric pressure (Pa)
        mean_rho_vapor = aux["h2o"].mean()*10**-3                                                    # mean vapor density (kg/m^3) -- NOT THE FLUCTUATIONS!
        mean_rho_co2 =  aux["co2"].mean()*10**-6                                                     # mean carbon dioxide concentration (kg/m^3) -- NOT THE FLUCTUATIONS!
        mean_Tv = aux["Tv"].mean()                                                                   # mean virtual temperature in Celsius
        rho_totair = P/(Constants.Rd*(mean_Tv+273.15))                                               # moist air density (kg/m^3)
        ustar = ( matrixCov["u_p"]["w_p"]**2.0 + matrixCov["w_p"]["v_p"]**2.0 )**(1.0/4.0)           # friction velocity [m/s]
        Qv = matrixCov["Tv_p"]["w_p"]                                                                # kinematic virtual temperature flux [K m/s]
        dd = self.hi*(2/3)                                                                           # displacement height [m]
        zeta = -Constants.VON_KARMAN*Constants.g*( self.zi - dd )*Qv/((mean_Tv + 273.15)*(ustar**3)) # Monin-Obukhov stability parameter
        zv = 0.2 * (0.1 * self.hi)                                                                   # Roughness parameter
        
        # 1 - Finding near canopy concentrations --------------------------------------------------
        # Calculating Monin Obukhov nondimensional function 
        # Following Fluxpart - Skaggs et al., 2018
        if zeta < -0.04: psi_v = 2.0 * np.log((1 + (1 - 16.0 * zeta) ** 0.5) / 2)
        elif zeta <= 0.04: psi_v = 0.0
        else: psi_v = -5.0 * zeta
        
        arg = (np.log((self.zi - dd) / zv) - psi_v) / Constants.VON_KARMAN / ustar
        ambient_h2o = mean_rho_vapor + cov_wq*arg                     # mean h2o concentration near canopy [kg/m3] - using log-law profiles
        ambient_co2 =   mean_rho_co2 + cov_wc*arg                     # mean co2 concentration near canopy [kg/m3] - using log-law profiles
        
        # 2 - Finding intercelular concentrations -------------------------------------------------
        esat = sat_vapor_press(leaf_T)                                # Intercellular saturation vapor pressure 'esat' [Pa]

        # Intercellular vapor density 
        eps = Constants.MWvapor / Constants.MWdryair
        inter_h2o = rho_totair * eps * esat / (P - (1 - eps) * esat)  # mean h2o concentration inside stomata [kg/m3]

        # vapor pressure deficit
        vpd = vapor_press_deficit(ambient_h2o, leaf_T, Constants.Rvapor)
        if vpd < 0:
            # Negative vapor pressure deficit
            return None
            
        # Calculating inside stomata co2 concentration  
        
        ci_mod_const_ppm = ci_const_ppm(P, leaf_T, Constants.Rco2, Constants.wue_constants[ppath]["const_ppm"])
        ci_mod_const_ratio = cica_const_ratio(ambient_co2, Constants.wue_constants[ppath]["const_ratio"])
        ci_mod_linear = cica_linear(ambient_co2, vpd, Constants.wue_constants[ppath]["linear"][0], Constants.wue_constants[ppath]["linear"][1])
        ci_mod_sqrt = cica_sqrt(ambient_co2, vpd, Constants.wue_constants[ppath]["sqrt"])
        
        # 3 - Compute water use efficiency ----------------------------------------------------------------------------
        
        for ci,mod in [[ci_mod_const_ppm,"const_ppm"], [ci_mod_const_ratio, "const_ratio"], [ci_mod_linear, "linear"], [ci_mod_sqrt, "sqrt"] ]:
            coef = 1.0 / Constants.diff_ratio
            wuei = coef * (ambient_co2 - ci) / (ambient_h2o - inter_h2o)
            self.wue[mod] = wuei

        # Optimization model from Scanlon et al., 2019 - only applicable to C3 plants
        if ppath == "C3":
            m = - ( varc*cov_wq - corr_qc*sigmaq*sigmac*cov_wc )/( varq*cov_wc - corr_qc*sigmaq*sigmac*cov_wq )
            vpdm = vapor_press_deficit_mass(ambient_h2o, leaf_T, Constants.Rvapor)
            if vpdm < 0 or m < 0:
                self.wue['opt'] = np.nan  # In case of negative vapor pressure deficit or m
            else: 
                self.wue['opt'] = ( Constants.diff_ratio*vpdm*m - np.sqrt( Constants.diff_ratio*vpdm*m*( ambient_co2 + Constants.diff_ratio*vpdm*m )) )/( Constants.diff_ratio*vpdm)
        else: 
            self.wue['opt'] = np.nan  # Model is not suitable for C4 and CAM plants
        del aux

    def partCEC(self, H=0.0):
        """
        Implements the Conditional Eddy Covariance method proposed by Zahn et al. 2021
                Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components
        Input:
            H - hyperbolic threshold

        Used variables: 
             - w_p - fluctuation of vertical velocity (m/s)
             - c_p - fluctuations of co2 density (mg/m3)
             - q_p - fluctuations of h2o density (g/m3)

        Create a dictionaty 'fluxesCEC' with all flux components 
             - ET - total evapotranspiration (W/m2)
             - T - plant transpiration (W/m2)
             - E - soil/surface evaporation (W/m2)
             - Fc - carbon dioxide flux (mg/m2/s)
             - R - soil/surface respiration (mg/m2/s)
             - P - plant net photosynthesis* (mg/m2/s)
        * this component represents carboxylation minus photorespiration and leaf respiration; therefore,
          it is different from gross primary productivity
        """

        per_points_Q1Q2 = 15  # smallest percentage of points that must be available in the first two quadrants
        per_poits_each  = 3   # smallest percentage of points in each quadrant

        # Creates a dataframe with variables of interest and no constraints
        auxET    = self.data[["co2_p", "h2o_p", "w_p"]]
        N        = auxET.index.size                                                               # Total number of points
        total_Fc = np.mean(auxET["co2_p"].values * auxET["w_p"].values)                           # flux [all quadrants] given in mg/(s m2)
        total_ET = (10**-3)*Constants.Lv*np.mean(auxET["h2o_p"].values * auxET["w_p"].values )    # flux [all quadrants] given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE           = self.data[ (self.data["w_p"] > 0)&(self.data["co2_p"] > 0)&(self.data["h2o_p"] > 0)&(abs(self.data["co2_p"]/self.data["co2_p"].std())>abs(H*self.data["h2o_p"].std()/self.data["h2o_p"])) ][["co2_p", "h2o_p", "w_p"]]
        R_condition_Fc = np.sum(auxE["co2_p"].values * auxE["w_p"].values )/N                           # conditional flux [1st quadrant and w'>0] given in mg/(s m2)
        E_condition_ET = (10**-3)*Constants.Lv * np.sum(auxE["h2o_p"].values * auxE["w_p"].values)/N    # conditional flux [1st quadrant and w'>0] flux given in  W/m2
        sumQ1          = (auxE['w_p'].index.size/N)*100                                                # Percentage of points in the first quadrant

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT           = self.data[ (self.data["w_p"] > 0)&(self.data["co2_p"] < 0)&(self.data["h2o_p"] > 0)&(abs(self.data["co2_p"]/self.data["co2_p"].std())>abs(H*self.data["h2o_p"].std()/self.data["h2o_p"])) ][["co2_p", "h2o_p", "w_p"]]
        P_condition_Fc = np.sum(auxT["co2_p"].values * auxT["w_p"].values)/N         # conditional flux [2nd quadrant and w'>0] given in mg/(s m2)
        T_condition_ET = (10**-3)*Constants.Lv*np.sum(auxT["h2o_p"]*auxT["w_p"])/N   # conditional flux [2nd quadrant and w'>0] flux given in  W/m2
        sumQ2 = (auxT['w_p'].index.size/N)*100                                       # Percentage of points in the second quadrant

        # Computing flux ratios and flux components of ET and Fc
        E, T, P, R, ratioET, ratioRP = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        # First condition: do we have enough points in Q1 and Q2?
        if (sumQ1 + sumQ2) < per_points_Q1Q2:
            self.fluxesCEC = { 'ET': total_ET,  'E': E,        'T': T,     'Fc': total_Fc,    'P': P,     'R': R, 'rRP': ratioRP, 'rET': ratioET, 'status': 'Q1+Q2<20'  }
        elif (sumQ1>=per_poits_each) and (sumQ2>=per_poits_each):
            # Compute all flux components

            # ET components
            ratioET = E_condition_ET/T_condition_ET
            T       =  total_ET/(1.0 + ratioET)
            E       =  total_ET/(1.0 + 1.0/ratioET)

            # Fc components
            ratioRP = R_condition_Fc/P_condition_Fc
            P       = total_Fc/(1.0 + ratioRP)  
            R       = total_Fc/(1.0 + 1.0/ratioRP)
        elif (sumQ1<per_poits_each) and (sumQ2>per_poits_each):
            # In this case, all water vapor flux is assumed to be transpiration
            ratioET = 0.0
            T       = total_ET
            E       = 0.0
            # In this case, all co2 flux is assumed to be photosynthesis
            ratioRP = 0.0
            P       = total_Fc
            R       = 0.0
        elif (sumQ1>per_poits_each) and (sumQ2<per_poits_each):
            # All fluxes are assumed to be from the ground
            ratioET = np.inf
            T       = 0.0
            E       = total_ET
            # All Fc flux is considered to be respiration
            ratioRP = np.inf
            P       = 0.0  
            R       = total_Fc
        else:
            pass

        # Check CO2 flux components ratio
        #   CO2 fluxes might be noisy in this range
        if -1.2 < ratioRP < -0.8: 
            finalstat = 'Small ratioRP'
        else: finalstat = 'OK'

        # Additional constraints may be added based on the strength of the fluxes and other combinations
        self.fluxesCEC = { 'ET': total_ET,  'E': E,        'T': T,     'Fc': total_Fc,    'P': P,     'R': R, 'rRP': ratioRP, 'rET': ratioET,  'status': finalstat  }

    def partREA(self, H=0):
        """
        Implements the Modified Relaxed Eddy Accumulation proposed by Thomas et al., 2008 (Agr For Met)
                Estimating daytime subcanopy respiration from conditional sampling methods applied to multi-scalar high frequency turbulence time series
                https://www.sciencedirect.com/science/article/pii/S0168192308000737
                New contraints defined in Zahn et al (2021)   
        Input:
            H - hyperbolic threshold
        
        Used variables: 
             - w_p - fluctuation of vertical velocity (m/s)
             - c_p - fluctuations of co2 density (mg/m3)
             - q_p - fluctuations of h2o density (g/m3)
        
        Create a dictionaty 'fluxesREA' with all flux components 
             - ET - total evapotranspiration (W/m2)
             - T - plant transpiration (W/m2)
             - E - soil/surface evaporation (W/m2)
             - Fc - net carbon dioxide flux (mg/m2/s)
             - R - soil/surface respiration (mg/m2/s)
             - P - plant net photosynthesis* (mg/m2/s)
        * this component represents carboxylation minus photorespiration and leaf respiration; therefore,
          it is different from gross primary productivity
        """

        per_points_Q1Q2 = 15  # smallest percentage of points that must be available in the first two quadrants
        per_poits_each  = 3   # smallest percentage of points in each quadrant

        # REA parameters ---------------------------------------------------
        wseries = np.array(self.data["w_p"].values)                                             #  m/s
        cseries = np.array(self.data['co2_p'].values)                                           # mg/m3
        qseries = np.array(self.data['h2o_p'].values)                                           #  g/m3
        sigmaw  = np.std(wseries)                                                               # standard deviation of vertical velocity (m/s)
        sigmac  = np.std(cseries)                                                               # standard deviation of co2 density 
        sigmaq  = np.std(qseries)                                                               # standard deviation of water vapor density        
        beta    = sigmaw/( np.mean(wseries[wseries>0]) - np.mean(wseries[wseries<0]) )          # similarity constant
        Fc      = np.cov(wseries, cseries)[0][1]                                                # CO2 flux [mg/m2/s]
        ET      = np.cov(wseries, qseries)[0][1]*(10**-3)*Constants.Lv                          # latent heat flux [W/m2]
        NN      = len(wseries)                                                                  # total number of points

        # For carbon fluxes: numerator and denominator of equation 11 in Thomas et al., 2008
        cnum = cseries[(qseries>0)&(cseries>0)&(wseries>0)&((qseries/sigmaq)>(H*sigmac/cseries))&((cseries/sigmac)>(H*sigmaq/qseries))]
        cdiv = cseries[(abs(qseries/sigmaq)>abs(H*sigmac/cseries))&(abs(cseries/sigmac)>abs(H*sigmaq/qseries))&(wseries>0)]

        # For water vapor fluxes: numerator and denominator of equation 11 in Thomas et al., 2008
        qnum = qseries[(qseries>0)&(cseries>0)&(wseries>0)&((qseries/sigmaq)>(H*sigmac/cseries))&((cseries/sigmac)>(H*sigmaq/qseries))]
        qdiv = qseries[(abs(qseries/sigmaq)>abs(H*sigmac/cseries))&(abs(cseries/sigmac)>abs(H*sigmaq/qseries))&(wseries>0)]

        # Count number of points in the first and second quadrant (no H is used here)
        Q1sum = ( len( cseries[(qseries>0)&(cseries>0)&(wseries>0)] )/NN )*100
        Q2sum = ( len( cseries[(qseries>0)&(cseries<0)&(wseries>0)] )/NN )*100

        # Check availability of points in each quadrant
        if (Q1sum + Q2sum) < per_points_Q1Q2:
            self.fluxesREA = { 'ET': ET,  'Fc': Fc, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan, 'status': 'Q1+Q2<20'  }
        elif (Q1sum>=per_poits_each) and (Q2sum>=per_poits_each):
            # Compute fluxes
            R = beta*sigmaw*( sum(cnum)/len(cdiv) )    # Respiration [mg / (s m2)]
            E = beta*sigmaw*( sum(qnum)/len(qdiv) )    # Evaporation [g / (s m2)]
            E = E*(10**-3)*Constants.Lv                # Latent heat flux [W/m2]
            P = Fc - R                                 # Photosynthesis  [mg/(s m2)]
            T = ET - E                                 # Transpiration   [W/m2]
            finalstatus = 'OK'

            # To test realistic fluxes
            if E > 1.01*ET: 
                finalstatus = 'E>ET'
                E = np.nan; T = np.nan    
                R = np.nan; P = np.nan    
            self.fluxesREA = { 'ET': ET,  'Fc': Fc, 'E': E, 'T': T, 'P': P, 'R': R, 'status': finalstatus  }
        elif (Q1sum<per_poits_each)  and (Q2sum>per_poits_each):
            # Assuming that all fluxes are from the canopy
            self.fluxesREA = {   'ET': ET,   'Fc': Fc, 'E': 0, 'T': ET, 'P': Fc, 'R': 0,  'status': 'OK' }
        elif (Q1sum>per_poits_each) and (Q2sum<per_poits_each):
            # Assuming that all fluxes are from the ground
            self.fluxesREA = {   'ET': ET,   'Fc': Fc, 'E': ET, 'T': 0, 'P': 0, 'R': Fc,  'status': 'OK' }
        else: 
            pass
 
    def partFVS(self, W):
        """
        Partitioning based on Flux Variance Similarity Theory (FVS)
        This implementation directly follows the paper by Scanlon et al., 2019, Agr. For. Met.
                       "Correlation-based flux partitioning of water vapor and carbon dioxidefluxes: 
                       Method simplification and estimation of canopy water use efficiency"
        Parts of the implementation are adapted from Skaggs et al. 2018, Agr For Met
                       "Fluxpart: Open source software for partitioning carbon dioxide and watervaporfluxes" 

        Input:
              W - water use efficiency [kg_co2/kg_h2o]
                  If not available, W can be computed from any of the models
                  in the function WaterUseEfficiency

        Used variables: 
             - w_p - fluctuation of vertical velocity (m/s)
             - c_p - fluctuations of co2 density (mg/m3)
             - q_p - fluctuations of h2o density (g/m3)

        Creates a dictionary 'fluxesFVS' containing all flux components
             - ET - total evapotranspiration (W/m2)
             - T - plant transpiration (W/m2)
             - E - soil/surface evaporation (W/m2)
             - Fc - carbon dioxide flux (mg/m2/s)
             - R - soil/surface respiration (mg/m2/s)
             - P - plant net photosynthesis* (mg/m2/s)
        * this component represents carboxylation minus photorespiration and leaf respiration; therefore,
          it is different from gross primary productivity
        """
        
        aux = self.data[["co2_p", "h2o_p", "w_p"]].copy()     # Create dataframe with q, c, and w only
        aux["co2_p"] = aux["co2_p"]*10**-3                    # convert c from mg/m3 to g/m3
        
        # Needed statistics ------------------------------------------------
        var_all = aux.var()                   # Variance matrix
        cov_all = aux.cov()                   # Covariance matrix
        rho = aux.corr()["co2_p"]["h2o_p"]    # Correlation coefficient between c and q
        varq = var_all['h2o_p']               # Variance of q [g/m3]^2
        varc = var_all['co2_p']               # Variance of c [g/m3]^2
        sigmaq = varq**0.5                    # Standard deviation of q [g/m3]
        sigmac = varc**0.5                    # Standard deviation of c [g/m3]
        Fq = cov_all["h2o_p"]["w_p"]          # water vapor flux    [g/m2/s]
        Fc = cov_all["co2_p"]["w_p"]          # Carbon dioxide flux [g/m2/s]

        # Check if conditions are satisfied (equations 13a-b from Scanlon 2019)
        A = (sigmac/sigmaq)/rho
        B = Fc/Fq
        C = rho*(sigmac/sigmaq)

        # Check mathematical constraints Eq (13) in Scanlon et al., 2019
        if rho < 0:
            if A <= B < C: pass  # constraints 13a
            else: 
                self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan, 'status': "13a not satisfied"  }
                return None       # if it does not obey, stop here
        else:
            if B < C: pass       # constraints 13b
            else: 
                self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan, 'status': "13b not satisfied"  }
                return None       # if it does not obey, stop here
        
        # 1 - Calcula var_cp and rho_cpcr (eq. 7 in Scanlon et al., 2019) 
        
        # Variance cp (7a)
        num = (1.0 - rho*rho)*(varq*varc*W**2.0)*(varq*Fc*Fc - 2.0*rho*sigmaq*sigmac*Fc*Fq + varc*Fq*Fq)
        den = ( varc*Fq + varq*Fc*W - rho*sigmaq*sigmac*(Fc + Fq*W) )**2.0
        var_cp = num/den                    # Variance of P
        # Correlation cp, cr (7b)
        num = (1.0 - rho*rho)*varq*varc*(Fc - Fq*W)**2.0
        den = (varq*Fc*Fc - 2.0*rho*sigmaq*sigmac*Fq*Fc + varc*Fq*Fq)*(varc - 2.0*rho*sigmaq*sigmac*W + varq*W*W)
        rho_cpcr2 = num/den
        
        # 2 - Obtain flux components (eq. 6 in Scanlon et al., 2019) 
    
        # Compute roots and test if they are real 
        
        arg1 = 1.0 - (1.0 - W*W*varq/var_cp)/rho_cpcr2
        if arg1 < 0: 
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan, 'status': 'arg1 < 0'  }
            return None      # Root is not real; stop partitioning
            
        arg2 = 1.0 - (1.0 - varc/var_cp)/rho_cpcr2
        if arg2 < 0: 
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan, 'status': 'arg2 < 0'  }
            return None      # Root is not real; stop partitioning
            
        # Roots are real. Proceed to check sign of fluxes 
        ratio_ET = - rho_cpcr2 + rho_cpcr2*np.sqrt(arg1)
        if ratio_ET < 0.0:
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan , 'status': 'rET < 0' }
            return None    # Following imposed constraint that T, E > 0

        # Test ratio of carbon fluxes: from Fluxpart - Skaggs et al, 2018
        if rho < 0 and sigmac/sigmaq < rho*W:
            ratio_RP = - rho_cpcr2 + rho_cpcr2*np.sqrt( arg2 )
        else:
            ratio_RP = - rho_cpcr2 - rho_cpcr2*np.sqrt( arg2 )
        
        if ratio_RP > 0.0:
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan, 'status': 'rRP > 0'  }
            return None    # Following imposed constraint that P<0 and R>0
        
        # Obtaining flux components -------------------------------------------------
        T = Fq/(1.0 + ratio_ET)
        E = Fq - T
        
        T = T*(10**-3)*Constants.Lv  # in W/m2
        E = E*(10**-3)*Constants.Lv  # in W/m2
        
        P = Fc/(1.0 + ratio_RP)  #  g/m2/s
        R = Fc - P               #  g/m2/s
        P = P*10**3              # mg/m2/s
        R = R*10**3              # mg/m2/s
        
        # Convert total fluxes
        Fq = Fq*(10**-3)*Constants.Lv     # in W/m2
        Fc = Fc*10**3                     # mg/m2/s

        # Check CO2 flux components ratio
        #   CO2 fluxes might be noisy in this range
        ratioRP = R/P
        
        if -1.2 < ratioRP < -0.8: 
            finalstat = 'Small ratioRP'
        else: finalstat = 'OK'

        finalstat
        # Add final values to dictionary
        self.fluxesFVS = { 'ET': Fq,  'Fc': Fc, 'E': E, 'T': T, 'P': P, 'R': R, 'status': finalstat  }

    def partCEA(self, H=0.00):
        """
        Implements the Conditional Eddy Accumulation
        """
        # Creates a dataframe with variables of interest and no constraints
        unitLE   = Constants.Lv*10**-3
        df       = self.data.copy()
        auxET    = df[["c_p", "q_p", 'w_p' ]].copy()
        auxETcov = auxET.cov()                            # covariance
        N        = auxET['w_p'].index.size                # Total number of points
        total_Fc = auxETcov['c_p']['w_p']                 # flux [all quadrants] given in mg/(s m2)
        total_ET = auxETcov["q_p"]['w_p'] * unitLE        # flux [all quadrants] given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE    = df[ (df['w_p'] > 0)&(df["c_p"] > 0)&(df["q_p"] > 0)&(abs(df["c_p"]/df["c_p"].std())>abs(H*df["q_p"].std()/df["q_p"])) ][["c_p", "q_p", 'w_p']]
        auxE_n  = df[ (df['w_p'] < 0)&(df["c_p"] < 0)&(df["q_p"] < 0)&(abs(df["c_p"]/df["c_p"].std())>abs(H*df["q_p"].std()/df["q_p"])) ][["c_p", "q_p", 'w_p']]
        C1      = auxE['c_p'].mean()
        C2      = auxE_n['c_p'].mean()
        Q1      = auxE['q_p'].mean()
        Q2      = auxE_n['q_p'].mean()
        sum1    = (  auxE['c_p'].index.size / N) * 100    # Number of points on the first quadrant
        sum2    = (auxE_n['c_p'].index.size / N) * 100    # Number of points on the first quadrant

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT    = df[ (df['w_p'] > 0)&(df["c_p"] < 0)&(df["q_p"] > 0)&(abs(df["c_p"]/df["c_p"].std())>abs(H*df["q_p"].std()/df["q_p"])) ][["c_p", "q_p", 'w_p']]
        auxT_n  = df[ (df['w_p'] < 0)&(df["c_p"] > 0)&(df["q_p"] < 0)&(abs(df["c_p"]/df["c_p"].std())>abs(H*df["q_p"].std()/df["q_p"])) ][["c_p", "q_p", 'w_p']]
        C3      = auxT['c_p'].mean()
        C4      = auxT_n['c_p'].mean()
        Q3      = auxT['q_p'].mean()
        Q4      = auxT_n['q_p'].mean()
        sum3    = (  auxT['c_p'].index.size / N) * 100    # Number of points on the first quadrant
        sum4    = (auxT_n['c_p'].index.size / N) * 100    # Number of points on the first quadrant

        # Computing flux ratios and flux components of ET and Fc
        if ( sum1 > 5 ) and ( sum2 > 5 ) and ( sum3 > 5 )and ( sum4 > 5 ):

            ratioET = (Q1 - Q2)/(Q3 - Q4)
            T       = total_ET/(1.0 + ratioET)
            E       = total_ET/(1.0 + 1.0/ratioET)

            ratioRP = (C1 - C2)/(C3 - C4)
            P       = total_Fc/(1.0 + ratioRP)
            R       = total_Fc/(1.0 + 1.0/ratioRP)

            self.fluxesCEA = { 'ET': total_ET,
                                'E': E,
                                'T': T,
                               'Fc': total_Fc,
                                'P': P,
                                'R': R,
                          'ratioET': ratioET,
                          'ratioRP': ratioRP,
                            'sumQ1': sum1 + sum2,
                            'sumQ2': sum3 + sum4,
                              'wue': P/T * 100  }
        else:
            self.fluxesCEA = {  'ET': total_ET,
                                 'E': np.nan,
                                 'T': np.nan,
                                'Fc': total_Fc,
                                 'P': np.nan,
                                 'R': np.nan,
                           'ratioET': np.nan,
                           'ratioRP': np.nan,
                             'sumQ1': sum1 + sum2,
                             'sumQ2': sum3 + sum4,
                               'wue': np.nan  }

    def partCECw(self, WUE, H=0.00):
        """
        Implements a combination of the Conditional Eddy Covariance method proposed by Zahn et al. 2021
               and the water use efficiency
        """
        # Creates a dataframe with variables of interest and no constraints
        df       = self.data.copy()
        auxET    = df[["c_p", "q_p", 'w_p']].copy()
        auxETcov = auxET.cov()                          # covariance
        N        = auxET['w_p'].index.size              # Total number of points
        total_Fc = auxETcov['c_p']['w_p']               # flux [all quadrants] given in mg/(s m2)
        total_ET = auxETcov["q_p"]['w_p']*10**3         # flux [all quadrants] given in mg/(s m2)

        if WUE > (total_Fc/total_ET):
            self.fluxesCEC_WUE = { 'ET': np.nan,
                                    'E': np.nan,
                                    'T': np.nan,
                                   'Fc': np.nan,
                                    'P': np.nan,
                                    'R': np.nan }
            return 0

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE           = df[ (df['w_p'] > 0)&(df["c_p"] > 0)&(df["q_p"] > 0)&(abs(df["c_p"]/df["c_p"].std())>abs(H*df["q_p"].std()/df["q_p"])) ][["c_p", "q_p", 'w_p']]
        R_condition_Fc =  sum(auxE["c_p"]*auxE['w_p'])/N                      # conditional flux [1st quadrant and w'>0] given in mg/(s m2)
        E_condition_ET = (sum(auxE["q_p"]*auxE['w_p'])/N)                     # conditional flux [1st quadrant and w'>0] flux given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT           = df[ (df['w_p'] > 0)&(df["c_p"] < 0)&(df["q_p"] > 0)&(abs(df["c_p"]/df["c_p"].std())>abs(H*df["q_p"].std()/df["q_p"])) ][["c_p", "q_p", 'w_p']]
        P_condition_Fc =  sum(auxT["c_p"]*auxT['w_p'])/N                       # conditional flux [2nd quadrant and w'>0] given in mg/(s m2)
        T_condition_ET = (sum(auxT["q_p"]*auxT['w_p'])/N)                     # conditional flux [2nd quadrant and w'>0] flux given in  W/m2

        # Compute needed parameters (ratios first) # *unitLE 
        if T_condition_ET > 0:
            ratioET = E_condition_ET/T_condition_ET
            ratioRP = R_condition_Fc/P_condition_Fc
        else:
            self.fluxesCEC_WUE = { 'ET': total_ET*(10**-3)*(10**-3)*Constants.Lv,
                                    'E': total_ET*(10**-3)*(10**-3)*Constants.Lv,
                                    'T': 0,
                                   'Fc': total_Fc,
                                    'P': 0,
                                    'R': total_Fc }
            return 0

        # Compute Z -------------------------------
        if ratioET > 0:
            Z = WUE * ( ratioRP / ratioET )
        else:
            self.fluxesCEC_WUE = { 'ET': total_ET*(10**-3)*(10**-3)*Constants.Lv,
                                    'E': 0,
                                    'T': total_ET*(10**-3)*(10**-3)*Constants.Lv,
                                   'Fc': total_Fc,
                                    'P': total_Fc,
                                    'R': 0 }
            return 0

        # Compute flux components -----------------
        R = ( total_Fc - WUE*total_ET )/(1 - WUE/Z)       # in (mg/kg)/m2/s
        P = total_Fc - R                                  # in (mg/kg)/m2/s
        E = (R/Z)*(10**-3)*(10**-3)*Constants.Lv          # in W/m2
        T = total_ET*(10**-3)*(10**-3)*Constants.Lv - E   # in W/m2
        del ratioET, ratioRP

        # ratios for spatial fluxes

        self.fluxesCEC_WUE = { 'ET': total_ET*(10**-3)*(10**-3)*Constants.Lv,
                                'E': E,
                                'T': T,
                               'Fc': total_Fc,
                                'P': P,
                                'R': R }
