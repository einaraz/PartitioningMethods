# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from auxfunctions import ci_const_ppm, cica_const_ratio, cica_linear, cica_sqrt, sat_vapor_press, vapor_press_deficit, vapor_press_deficit_mass, fes


"""
Author: Einara Zahn <einaraz@princeton.edu>, <einara.zahn@gmail.com>
Last update: Dec 24, 2021
"""

class Constants:
    """
    Define constants
    """
    Rd = 287                      # gas constant of dry air - J/K/kg
    Lv = 2.453*10**6              # latent heat - J/kg
    rho_w = 1000                  # density of water - kg/m3
    VON_KARMAN=0.4                # von Karman constant
    MWdryair=0.0289645            # Molecular weight of dry air
    MWvapor=0.018016              # Molecular weight of vapor
    MWco2=0.044010                # Molecular weight of carbon dioxide
    Rco2=8.3144598/MWco2          # Gas constant for CO2
    Rvapor=8.3144598/MWvapor      # Gas constant for water vapor
    diff_ratio=1.6                # Ratio of diffusivities water/co2
    g=9.8160                      # gravity m/s2
    # Constants used for water use efficiency (see Partitioning.WaterUseEfficiency for definitions)
    wue_constants = { "C3": {"const_ppm": 280, "const_ratio": 0.70,        "linear": (1.0, 1.6e-4) , "sqrt": (22e-9) },
                      "C4": {"const_ppm": 130, "const_ratio": 0.44,        "linear": (1.0, 2.7e-4) , "sqrt":  np.nan } }

class Partitioning(object):
    """
    Inputs:
        hi - canopy height (m)
        zi - eddy covariance measurement height (m)
        df - dataframe with data (e.g., 30min, but any length works), each variable in a column
             All variables should be previously processed (despiking, rotation of coordinates, detrending, etc)
             
             In the accompanying files, the following data processing was previously applied:
             --> Quality control (removing outliers, despiking, flags of instruments, etc)
             --> Rotation of coordinates (double rotation) for velocity components u, v, w measured by csat
             --> Density corrections for instantaneous fluctuations of co2 (c_p) and h2o (q_p) ("instantaneous" WPL correction) based on the paper 
                         Detto, M. and Katul, G. G., 2007. "Simplified expressions for adjusting higher-order turbulent statistics obtained from open path gas analyzers"
                         Boundary-Layer Meteorology, 10.1007/s10546-006-9105-1
             --> Primed quantities ("_p") were computed from extracting the linear trend of the time series
             --> Air temperature (T) and virtual temperature (Tv) computed from the sonic temperature (Ts)
             
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
             
    The following partinioning methods are available (references for each method below)
        - Conditional Eddy Covariance (CEC)
        - Modified Relaxed Eddy Accumulation (MREA)
        - Flux Variance Similarity (FVS)
    * Note that CEC and MREA only need time series of w_p, c_p, q_p
      The remaining quantities (e.g., P, T, Tv, etc) are only needed if the
      water use efficiency is computed. Alternatively, an external WUE can be
      used; in this case, FVS will only need time series of w_p, c_p, q_p
    
    """
    def __init__(self, hi, zi, df):
        self.hi = hi                  # Canopy height (m)
        self.zi = zi                  # Measurement height (m)
        self.data = df                # pandas dataframe containing data

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
        matrixCov = aux.cov()                                              # Covariance matrix
        varq = matrixCov["q_p"]["q_p"]*10**-6                              # variance of h2o fluctuations (kg/m3)^2
        varc = matrixCov["c_p"]["c_p"]*10**-12                             # variance of co2 fluctuations (kg/m3)^2
        sigmac = varc**0.5                                                 # Standard deviation of co2 [kg/m3]
        sigmaq = varq**0.5                                                 # Standard deviation of h2o [kg/m3]
        corr_qc = np.corrcoef(aux["q_p"].values, aux["c_p"].values)[1,0]   # correlation coefficient between q and c
        cov_wq = matrixCov["w_p"]["q_p"]*10**-3                            # covariance of w and q (kg/m^2/s);
        cov_wc = matrixCov["w_p"]["c_p"]*10**-6                            # covariance of w and c (kg/m^2/s);

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
        # Following fluxpart - Skaggs et al., 2018
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

    def partCEC(self, H=0.25):
        """
        Implements the Conditional Eddy Covariance method proposed by Zahn et al. 2021
                Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components
        
        Needed variables: 
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
        
        # Creates a dataframe with variables of interest and no constraints
        auxET = self.data[["c_p", "q_p", "w_p"]]
        N=auxET["w_p"].size                                                # Total number of points
        total_Fc = sum(auxET["c_p"]*auxET["w_p"])/N                        # flux [all quadrants] given in mg/(s m2)
        total_ET = (10**-3)*Constants.Lv*sum(auxET["q_p"]*auxET["w_p"])/N  # flux [all quadrants] given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE  = self.data[ (self.data["w_p"] > 0)&(self.data["c_p"] > 0)&(self.data["q_p"] > 0)&(abs(self.data["c_p"]/self.data["c_p"].std())>abs(H*self.data["q_p"].std()/self.data["q_p"])) ][["c_p", "q_p", "w_p"]]
        R_condition_Fc = sum(auxE["c_p"]*auxE["w_p"])/N                   # conditional flux [1st quadrant and w'>0] given in mg/(s m2)
        E_condition_ET = (10**-3)*Constants.Lv*sum(auxE["q_p"]*auxE["w_p"])/N  # conditional flux [1st quadrant and w'>0] flux given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT  = self.data[ (self.data["w_p"] > 0)&(self.data["c_p"] < 0)&(self.data["q_p"] > 0)&(abs(self.data["c_p"]/self.data["c_p"].std())>abs(H*self.data["q_p"].std()/self.data["q_p"])) ][["c_p", "q_p", "w_p"]]
        P_condition_Fc = sum(auxT["c_p"]*auxT["w_p"])/N                   # conditional flux [2nd quadrant and w'>0] given in mg/(s m2)
        T_condition_ET = (10**-3)*Constants.Lv*sum(auxT["q_p"]*auxT["w_p"])/N  # conditional flux [2nd quadrant and w'>0] flux given in  W/m2
        
        
        # Computing flux ratios and flux components of ET and Fc
        
        # 1 - when all components are non-zero 
        if abs(T_condition_ET) > 0.0 and abs(E_condition_ET) > 0.0:
            ratioET=E_condition_ET/T_condition_ET
            T = total_ET/(1.0 + ratioET);  
            E = total_ET/(1.0 + 1.0/ratioET);
        
        if abs(R_condition_Fc) > 0.0 and abs(P_condition_Fc) > 0.0:
            ratioRP = R_condition_Fc/P_condition_Fc
            P = total_Fc/(1.0 + ratioRP);  
            R = total_Fc/(1.0 + 1.0/ratioRP)
        
        # 2 - if conditional flux of E or R happens to be zero (no points)
        if abs(T_condition_ET) > 0.0 and E_condition_ET == 0.0:
            # In this case, all water fluxes are transpiration
            ratioET=0.0
            T=total_ET
            E=0.0
        
        if abs(P_condition_Fc) > 0.0 and R_condition_Fc == 0.0:
            # In this case, all co2 flux is photosynthesis
            ratioRP=0.0
            P=total_Fc
            R=0.0
            
        # 3 - if conditional flux of T or P happens to be zero (no points)
        if T_condition_ET == 0.0 and abs(E_condition_ET) > 0.0:
            ratioET=np.inf
            T=0.0
            E=total_ET
        
        if P_condition_Fc == 0.0 and abs(R_condition_Fc) > 0.0:
            ratioRP=np.inf
            P=0.0  
            R=total_Fc
        
        # Additional constraints may be added based on the strength of the fluxes and other combinations
        
        self.fluxesCEC = { 'ET': total_ET,  'E': E,        'T': T,     'Fc': total_Fc,    'P': P,     'R': R     }

    def partREA(self, H=0.25):
        """
        Implements the Modified Relaxed Eddy Accumulation proposed by Thomas et al., 2008 (Agr For Met)
                Estimating daytime subcanopy respiration from conditional sampling methods applied to multi-scalar high frequency turbulence time series
                https://www.sciencedirect.com/science/article/pii/S0168192308000737
        
        Needed variables: 
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
        
        # REA parameters ---------------------------------------------------
        wseries = np.array(self.data["w_p"].values)                                             #  m/s
        cseries = np.array(self.data['c_p'].values)                                             # mg/m3
        qseries = np.array(self.data['q_p'].values)                                             #  g/m3
        sigmaw = np.std(wseries)                                                                # standard deviation of vertical velocity
        sigmac = np.std(cseries)                                                                # standard deviation of co2 density
        sigmaq = np.std(qseries)                                                                # standard deviation of water vapor density        
        betaH = sigmaw/( np.mean(wseries[wseries>0]) - np.mean(wseries[wseries<0]) )            # similarity constant
        Fc = np.cov(wseries, cseries)[0][1]                                                     # CO2 flux [mg/m2/s]
        ET = np.cov(wseries, qseries)[0][1]*(10**-3)*Constants.Lv                               # latent heat flux [W/m2]

        # For carbon fluxes: numerator and denominator of equation 11 in Thomas et al., 2008
        cnum = cseries[(qseries>0)&(cseries>0)&(wseries>0)&((qseries/sigmaq)>(H*sigmac/cseries))&((cseries/sigmac)>(H*sigmaq/qseries))]
        cdiv = cseries[(abs(qseries/sigmaq)>abs(H*sigmac/cseries))&(abs(cseries/sigmac)>abs(H*sigmaq/qseries))&(wseries>0)]

        # For water vapor fluxes: numerator and denominator of equation 11 in Thomas et al., 2008
        qnum = qseries[(qseries>0)&(cseries>0)&(wseries>0)&((qseries/sigmaq)>(H*sigmac/cseries))&((cseries/sigmac)>(H*sigmaq/qseries))]
        qdiv = qseries[(abs(qseries/sigmaq)>abs(H*sigmac/cseries))&(abs(cseries/sigmac)>abs(H*sigmaq/qseries))&(wseries>0)]
       
        # If no points were in the first quadrant following the conditions of the method, attribute all fluxes
        #             to stomatal components
        if len(cdiv) == 0 or len(qdiv) == 0:
            self.fluxesREA = {   'ET': ET,   'Fc': Fc, 'E': 0, 'T': ET, 'P': Fc, 'R': 0 }
        else:
            R = betaH*sigmaw*( sum(cnum)/len(cdiv) )   # Respiration [mg / (s m2)]
            E = betaH*sigmaw*( sum(qnum)/len(qdiv) )   # Evaporation [g / (s m2)]
            E = E*(10**-3)*Constants.Lv                # Latent heat flux [W/m2]
            P = Fc - R                                 # Photosynthesis  [mg/(s m2)]
            T = ET - E                                 # Transpiration   [W/m2]
            
            # To test realistic fluxes
            if E > 1.01*ET: 
                E = np.nan; T = np.nan    
                R = np.nan; P = np.nan    
                
            # ------------------------------------------------------------------
            self.fluxesREA = { 'ET': ET,  'Fc': Fc, 'E': E, 'T': T, 'P': P, 'R': R  }

    def partFVS(self, W):
        """
        Partitioning based on Flux Variance Similarity Theory (FVS)
        This implementation directly follows the paper by Scanlon et al., 2019, Agr. For. Met.
                       "Correlation-basedflux partitioning of water vapor and carbon dioxidefluxes: 
                       Method simplification and estimation of canopy water use efficiency"
        Parts of the implementation are adapted from Skaggs et al. 2018, Agr For Met
                       "Fluxpart: Open source software for partitioning carbon dioxide and watervaporfluxes" 
        
        Needed variables: 
             - w_p - fluctuation of vertical velocity (m/s)
             - c_p - fluctuations of co2 density (mg/m3)
             - q_p - fluctuations of h2o density (g/m3)
        Input:
              W - water use efficiency [kg_co2/kg_h2o]
                  If not available, W can be computed from any of the models
                  in the function WaterUseEfficiency

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
        
        aux = self.data[["c_p", "q_p", "w_p"]].copy()     # Create dataframe with q, c, and w only
        aux["c_p"] = aux["c_p"]*10**-3                    # convert c from mg/m3 to g/m3
        
        # Needed statistics ------------------------------------------------
        var_all = aux.var()                 # Variance matrix
        cov_all = aux.cov()                 # Covariance matrix
        rho = aux.corr()["c_p"]["q_p"]      # Correlation coefficient between c and q
        varq = var_all['q_p']               # Variance of q [g/m3]^2
        varc = var_all['c_p']               # Variance of c [g/m3]^2
        sigmaq = varq**0.5                  # Standard deviation of q [g/m3]
        sigmac = varc**0.5                  # Standard deviation of c [g/m3]
        Fq = cov_all["q_p"]["w_p"]            # water vapor flux    [g/m2/s]
        Fc = cov_all["c_p"]["w_p"]            # Carbon dioxide flux [g/m2/s]

        # Check if conditions are satisfied (equations 13a-b from Scanlon 2019)
        A = (sigmac/sigmaq)/rho
        B = Fc/Fq
        C = rho*(sigmac/sigmaq)

        # Check mathematical constraints Eq (13) in Scanlon et al., 2019
        if rho < 0:
            if A <= B < C: pass  # constraints 13a
            else: 
                self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan  }
                return None       # if it does not obey, stop here
        else:
            if B < C: pass       # constraints 13b
            else: 
                self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan  }
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
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan  }
            return None      # Root is not real; stop partitioning
            
        arg2 = 1.0 - (1.0 - varc/var_cp)/rho_cpcr2
        if arg2 < 0: 
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan  }
            return None      # Root is not real; stop partitioning
            
        # Roots are real. Proceed to check sign of fluxes 
        ratio_ET = - rho_cpcr2 + rho_cpcr2*np.sqrt(arg1)
        if ratio_ET < 0.0:
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan  }
            return None    # Following imposed constraint that T, E > 0

        # Test ratio of carbon fluxes: from Fluxpart - Skaggs et al, 2018
        if rho < 0 and sigmac/sigmaq < rho*W:
            ratio_RP = - rho_cpcr2 + rho_cpcr2*np.sqrt( arg2 )
        else:
            ratio_RP = - rho_cpcr2 - rho_cpcr2*np.sqrt( arg2 )
        
        if ratio_RP > 0.0:
            self.fluxesFVS = { 'ET': Fq*(10**-3)*Constants.Lv,  'Fc': Fc*10**3, 'E': np.nan, 'T': np.nan, 'P': np.nan, 'R': np.nan  }
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
        
        # Add final values to dictionary
        self.fluxesFVS = { 'ET': Fq,  'Fc': Fc, 'E': E, 'T': T, 'P': P, 'R': R  }

