import numpy as np
from scipy import stats
import pandas as pd

# Auxiliary functions to compute the water-use efficiency
# These functions are from the algorithm fluxpart: 
#                 Skaggs et al. 2018, Agr For Met
#                "Fluxpart: Open source software for partitioning carbon dioxide and watervaporfluxes" 
# https://github.com/usda-ars-ussl/fluxpart

# Water-use efficiency computation -------------------
def ci_const_ppm(pressure, temperature, Rco2, ci_ppm):
    """Return ci = intercellular CO2 concentration, kg/m^3."""
    return ci_ppm * 1e-6 * pressure / Rco2 / temperature

def cica_const_ratio(ambient_co2, const):
    """ci/ca is constant."""
    """Return ci = intercellular CO2 concentration, kg/m^3."""
    return const * ambient_co2

def cica_linear(ambient_co2, vpd, b, m):
    """ci/ca is a decreasing linear function of vapor pressure deficit."""
    """Return ci = intercellular CO2 concentration, kg/m^3."""
    # b is unitless with a value of ~1, and m (> 0) has units of Pa^-1
    return ambient_co2 * (b - m * vpd)

def cica_sqrt(ambient_co2, vpd, lambd):
    """ci/ca is a function of sqrt(`vpd`/ca)."""
    """Return ci = intercellular CO2 concentration, kg/m^3."""
    # lambd has units of kg-co2 / m^3 / Pa
    return ambient_co2 * (1 - np.sqrt(1.6 * lambd * vpd / ambient_co2))

def sat_vapor_press(t_kelvin):
    tr = 1 - 373.15 / t_kelvin
    arg = 13.3185 * tr - 1.9760 * tr ** 2 - 0.6445 * tr ** 3 - 0.1299 * tr ** 4
    return 101325.0 * np.exp(arg)

def vapor_press_deficit(rho_vapor, t_kelvin, Rwv):
    return sat_vapor_press(t_kelvin) - rho_vapor * Rwv * t_kelvin

def vapor_press_deficit_mass(rho_vapor, t_kelvin, Rwv):
    return vapor_press_deficit(rho_vapor, t_kelvin, Rwv) / Rwv / t_kelvin

def fes(T):
    # return in Pa
    return 611.71*np.exp(2.501/0.461*10**3*(1/273.15-1/T) );

# processing -------------------------------------------------------

def LinearDetrend( tt, yy):
    # Linear regression ignoring nan values
    # Return detrended time series
    mask = ~np.isnan(tt) & ~np.isnan(yy)
    slope, intercept, _, _, _ = stats.linregress(tt[mask], yy[mask])
    yy_new = yy - (slope*tt + intercept)
    return yy_new

def Stats5min(x):
    """
    Returns covariances and variances for 5-min windows
    """
    if x.index.size > 100:
        c = x.cov()
        v = x.var()
        return pd.Series([ c['w_p']['co2_p'], c['w_p']['h2o_p'],  c['w_p']['Ts_p'], 
                           v['w_p'],   v['co2_p'],   v['h2o_p'],  v['Ts_p'] ],      index=[ 'wc', 'wq', 'wt', 'ww', 'cc', 'qq', 'tt'])
    else:
        return pd.Series([ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ], index=[ 'wc', 'wq', 'wt', 'ww', 'cc', 'qq', 'tt'])

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """

    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.nanmedian(arr)
    return np.nanmedian(abs(arr - med))

def find_spikes(data):
    """
    Receives a 2-min dataframe and search for outliers

    Input:
        data: pandas Series for each varible 
    Output:
        spikes_dates_list: list containing the dates (indexes) when spikes where detected
    """
    # Make copy of data
    aux_df   = pd.DataFrame({'data': data.values}, index=data.index)
    # compute statistics
    _mad     = mad(data)
    score_up = data.median() + 7 * _mad / 0.6745
    score_dn = data.median() - 7 * _mad / 0.6745
    aux_df['binary'] = 0
    aux_df['binary'][ (aux_df['data'].values>score_up) | (aux_df['data'].values<score_dn) ] = 1
    aux_df['consecutive'] = aux_df['binary'].groupby((aux_df['binary'] != aux_df['binary'].shift()).cumsum()).transform('size') * aux_df['binary']
    spikes_dates_list = aux_df[ (aux_df['consecutive'].values <= 8) & (aux_df['consecutive'].values >0) ].index
    spikes_dates_list = list(spikes_dates_list)
    del aux_df, score_up, score_dn
    return spikes_dates_list
