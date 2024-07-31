from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    """ci/ca is constant.
    Return ci = intercellular CO2 concentration, kg/m^3."""
    return const * ambient_co2


def cica_linear(ambient_co2, vpd, b, m):
    """ci/ca is a decreasing linear function of vapor pressure deficit.
    Return ci = intercellular CO2 concentration, kg/m^3."""
    # b is unitless with a value of ~1, and m (> 0) has units of Pa^-1
    return ambient_co2 * (b - m * vpd)


def cica_sqrt(ambient_co2, vpd, lambd):
    """ci/ca is a function of sqrt(`vpd`/ca).
    Return ci = intercellular CO2 concentration, kg/m^3."""
    # lambd has units of kg-co2 / m^3 / Pa
    return ambient_co2 * (1 - np.sqrt(1.6 * lambd * vpd / ambient_co2))


def sat_vapor_press(t_kelvin):
    """Computes saturation vapor pressure in Pa.

    Args:
        t_kelvin (float): _description_

    Returns:
        float: saturation vapor pressure in Pa
    """
    tr = 1 - 373.15 / t_kelvin
    arg = 13.3185 * tr - 1.9760 * tr**2 - 0.6445 * tr**3 - 0.1299 * tr**4
    return 101325.0 * np.exp(arg)


def vapor_press_deficit(rho_vapor, t_kelvin, Rwv):
    """Computes the vapor pressure deficit in Pa.

    Args:
        rho_vapor (float): water vapor density in kg/m^3
        t_kelvin (float): temperature in Kelvin
        Rwv (float): gas constant for water vapor in J/kg/K

    Returns:
        float: vapor pressure deficit in Pa
    """
    return sat_vapor_press(t_kelvin) - rho_vapor * Rwv * t_kelvin


def vapor_press_deficit_mass(rho_vapor, t_kelvin, Rwv):
    """Computes the vapor pressure deficit in kg/m^3.

    Args:
        rho_vapor (float): water vapor density in kg/m^3
        t_kelvin (float): temperature in Kelvin
        Rwv (float): gas constant for water vapor in J/kg/K

    Returns:
        float: vapor pressure deficit in kg/m^3
    """
    return vapor_press_deficit(rho_vapor, t_kelvin, Rwv) / Rwv / t_kelvin


def fes(T):
    """Return the saturated vapor pressure in Pa at temperature T in K."""
    return 611.71 * np.exp(2.501 / 0.461 * 10**3 * (1 / 273.15 - 1 / T))


def LinearDetrend(tt, yy):
    """Apply a linear detrend to the data.

    Args:
        tt (array): time array
        yy (array): data array

    Returns:
        yy_new: detrended data
    """
    mask = ~np.isnan(tt) & ~np.isnan(yy)
    slope, intercept, _, _, _ = stats.linregress(tt[mask], yy[mask])
    yy_new = yy - (slope * tt + intercept)
    return yy_new


def Stats5min(x):
    """
    Compute the statistics of the 5-min data

    Input:
        x: pandas DataFrame with the 5-min data

    Returns:
        pandas Series with the statistics of the data
    """
    if x.index.size > 100:
        c = x.cov()
        v = x.var()
        return pd.Series(
            [
                c["w_p"]["co2_p"],
                c["w_p"]["h2o_p"],
                c["w_p"]["Ts_p"],
                v["w_p"],
                v["co2_p"],
                v["h2o_p"],
                v["Ts_p"],
            ],
            index=["wc", "wq", "wt", "ww", "cc", "qq", "tt"],
        )
    else:
        return pd.Series(
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            index=["wc", "wq", "wt", "ww", "cc", "qq", "tt"],
        )


def mad(arr):
    """
    Compute the Median Absolute Deviation (MAD).

    The MAD is a robust measure of the variability of a univariate sample of quantitative data.

    Parameters
    ----------
    arr : array-like
        Input array or object that can be converted to an array. This array is flattened if it is not already 1-D.

    Returns
    -------
    float
        The median absolute deviation of the input array.

    Notes
    -----
    The Median Absolute Deviation (MAD) is defined as the median of the absolute deviations from the median of the data:
        MAD = median(|X - median(X)|)
    It is a robust measure of the variability of a dataset.

    References
    ----------
    .. [1] "Median absolute deviation." Wikipedia, The Free Encyclopedia.
           https://en.wikipedia.org/wiki/Median_absolute_deviation

    """

    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.nanmedian(arr)
    return np.nanmedian(abs(arr - med))


def find_spikes(data):
    """
    Detect spikes (outliers) in a pandas Series.

    Parameters
    ----------
    data : pandas Series
        Input time series data to search for spikes (outliers).

    Returns
    -------
    spikes_dates_list : list
        List containing the dates (indexes) where spikes were detected.

    Notes
    -----
    A spike (outlier) is identified based on the median absolute deviation (MAD) method,
    where values outside a threshold are considered spikes.

    """
    # Make copy of data
    aux_df = pd.DataFrame({"data": data.values}, index=data.index)
    # compute statistics
    _mad = mad(data)
    score_up = data.median() + 7 * _mad / 0.6745
    score_dn = data.median() - 7 * _mad / 0.6745
    aux_df["binary"] = 0
    aux_df.loc[(aux_df["data"] > score_up) | (aux_df["data"] < score_dn), "binary"] = 1
    aux_df["consecutive"] = (
        aux_df["binary"]
        .groupby((aux_df["binary"] != aux_df["binary"].shift()).cumsum())
        .transform("size")
        * aux_df["binary"]
    )
    spikes_dates_list = aux_df[
        (aux_df["consecutive"].values <= 8) & (aux_df["consecutive"].values > 0)
    ].index
    spikes_dates_list = list(spikes_dates_list)
    del aux_df, score_up, score_dn
    return spikes_dates_list


def FilterLowFrequencies(x, fs, t_cut):
    """
    Computes fft and kills frequencies corresponding
    to time scales equal to or larger than t_cut
    Adapted from Gaby Katul

    Filters a time series using Fourier transform methods

    Parameters:
    ----------
    x : array-like
        Input time series data.
    fs : float
        Sampling frequency of the time series.
    t_cut : float
        Cutoff time in minutes for the low-pass filter.

    Returns:
    -------
    x_new : array-like
        Filtered time series data.

    Raises:
    ------
    ValueError:
        If an invalid filter_type is provided.

    """

    NN = len(x)
    # Generating a time vector ---------------------------------
    t = np.arange(0, NN) * (1 / fs)

    # Cutoff in minutes converted to cycles per period----------
    n_cut = round(max(t) / 60 / t_cut) - 1
    n_cut = max(n_cut, 0)

    # Extract the mean -----------------------------------------
    x = x - np.mean(x)

    # Compute fourier coefficients -----------------------------
    xn_FFT = np.fft.fft(x)

    # Set all Fourier Coefficients to zero for freq < cutoff ---
    xn_FFTf = xn_FFT[0:NN]
    xn_FFTf = np.append(xn_FFTf, [xn_FFT[0]])
    xn_FFTf[0 : (n_cut + 1)] = 0
    xn_FFTf[-(n_cut + 1) :] = 0
    x_new = np.real(np.fft.ifft(xn_FFTf[0:NN]))

    return x_new


def max_time_lag_crosscorrel(
    df, sampling_freq, max_lag_seconds, type_lag, saveplotlag=False
):
    """
    Compute the lag with the highest correlation between CO2 or H2O concentration and vertical velocity (w).
    Lag is applied to the CO2 or H2O concentration time series only, while w is kept fixed.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series with columns 'co2', 'h2o', and 'w'.
    - sampling_freq (float): Sampling frequency of the data in Hz.
    - max_lag_seconds (float): Maximum lag to consider in seconds.
    - type_lag (str): Type of lag to consider ('negative', 'positive', or 'both').

    Returns:
    - best_lag_co2 (int): Lag (in number of samples) with the highest correlation for CO2.
    - best_lag_h2o (int): Lag (in number of samples) with the highest correlation for H2O.
    """
    tseries = df.copy()
    # Subtract the mean to compute the fluctuations
    tseries["co2"] = tseries["co2"] - np.mean(tseries["co2"])
    tseries["h2o"] = tseries["h2o"] - np.mean(tseries["h2o"])
    tseries["w"] = tseries["w"] - np.mean(tseries["w"])

    # Maximum lag in points
    max_lag_points = int(max_lag_seconds * sampling_freq)

    correlations_co2 = []
    correlations_h2o = []
    if type_lag == "negative":
        lags = range(-max_lag_points, 0)
    elif type_lag == "positive":
        lags = range(0, max_lag_points)
    else:
        lags = range(-max_lag_points, max_lag_points + 1)

    # Loop to determine the correlation for each lag
    for lag in lags:
        tseries_lag = tseries.copy()
        tseries_lag["co2"] = tseries_lag["co2"].shift(-lag)
        tseries_lag["h2o"] = tseries_lag["h2o"].shift(-lag)
        tseries_lag.dropna(inplace=True)
        correlations_co2.append(tseries_lag.corr()["co2"]["w"])
        correlations_h2o.append(tseries_lag.corr()["h2o"]["w"])

    # Find the lag with the maximum absolute correlation
    max_corr_index_co2 = np.argmax(np.abs(correlations_co2))
    max_corr_index_h2o = np.argmax(np.abs(correlations_h2o))
    best_lag_co2 = lags[max_corr_index_co2]
    best_lag_h2o = lags[max_corr_index_h2o]

    if saveplotlag:
        plt.plot(np.array(lags) / sampling_freq, correlations_co2, label="CO2")
        plt.plot(np.array(lags) / sampling_freq, correlations_h2o, label="H2O")
        plt.title(
            "Best lag for CO2: %.2f s , Best lag for H2O: %.2f s"
            % (best_lag_co2 / sampling_freq, best_lag_h2o / sampling_freq)
        )
        plt.legend()
        plt.ylabel("Correlation")
        plt.xlabel("Lag (sec)")
        # check if directiory exists
        if not os.path.exists("TimeLagCorrelationFigures"):
            os.makedirs("TimeLagCorrelationFigures")
        plt.savefig(
            "TimeLagCorrelationFigures/CrossCorrelation%s.png"
            % df.index[0].strftime("%Y%m%d%H%M")
        )
        plt.close()

    return best_lag_co2, best_lag_h2o
