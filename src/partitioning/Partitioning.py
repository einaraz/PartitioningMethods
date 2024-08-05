# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import pint
from .auxfunctions import (
    ci_const_ppm,
    cica_const_ratio,
    cica_linear,
    cica_sqrt,
    sat_vapor_press,
    vapor_press_deficit,
    vapor_press_deficit_mass,
    LinearDetrend,
    Stats5min,
    find_spikes,
    FilterLowFrequencies,
    max_time_lag_crosscorrel,
    Constants,
)

# Create a unit registry
ureg = pint.UnitRegistry()


class Partitioning(object):
    """
    Initializes the Partitioning class.

    Parameters
    ----------
    hi : float
        Canopy height (m).
    zi : float
        Eddy covariance measurement height (m).
    freq : int
        Sampling frequency (Hz).
    length : int
        Length of the time series (in minutes).
    df : pandas.DataFrame
        DataFrame with data (e.g., 30min intervals, but any length works), each variable in a column.
        If raw data is used, pre-processing is first implemented following these steps:
           - Quality control (removing outliers, despiking, flags of instruments, etc)
           - Rotation of coordinates (double rotation) for velocity components u, v, w measured by CSAT
           - Density corrections for instantaneous fluctuations of CO2 (c_p) and H2O (q_p) measured by open-gas analyser
             ("instantaneous" WPL correction) based on the paper: Detto, M. and Katul, G. G., 2007. "Simplified expressions for adjusting higher-order 
              turbulent statistics obtained from open path gas analyzers". Boundary-Layer Meteorology, 10.1007/s10546-006-9105-1
           - Turbulent fluctuations, here denoted as primed quantities ("_p"), are computed
           - Air temperature (T) and virtual temperature (Tv) computed from the sonic temperature (Ts)

        Raw data requires the following variables and units:
            - index : datetime
            - w : velocity in the z direction (m/s)
            - u : velocity in the x direction (m/s)
            - v : velocity in the y direction (m/s)
            - Ts : sonic temperature (Celsius)
            - P : pressure (kPa)
            - CO2 : carbon dioxide density (mg/m3)
            - H2O : water vapor density (g/m3)

        After pre-processing, the following additional variables are created:
            - w_p : fluctuations of velocity in the z direction (m/s)
            - u_p : fluctuations of velocity in the x direction (m/s)
            - v_p : fluctuations of velocity in the y direction (m/s)
            - T : thermodynamic air temperature (Celsius)
            - Tv : virtual temperature (Celsius)
            - co2_p : fluctuations of carbon dioxide density (mg/m3) - (corrected for external densities (WPL) if needed)
            - h2o_p : fluctuations of water vapor density (g/m3) - (corrected for external densities (WPL) if needed)
            - Ts_p : fluctuations of sonic air temperature (Celsius)
            - Tv_p : fluctuations of virtual temperature (Celsius)

    PreProcessing : bool, optional
        Indicates if pre-processing is necessary. If True, all pre-processing steps are implemented to raw data. If False,
        pre-processing is ignored and partitioning is immediately applied. In this case, the input files must contain all
        pre-processed variables listed above.

    argsQC : dict, optional
        Contains options to be used during pre-processing regarding fluctuation extraction and if density corrections are
        necessary. All options have default values, but can be modified if needed.

        Keys
        -----
            density_correction - bool
                True if density corrections are necessary (open gas analyzer); False (closed or enclosed gas analyzer).
            fluctuations - str
                Describes the type of operation used to extract fluctuations:
                'BA': block average
                'LD': Linear detrending
                'FL': Filter low frequencies. Requires filtercut to indicate the cutoff time in minutes.
            filtercut - int
                Cutoff time in minutes for the low-pass filter. Only used if method is 'FL'.
            maxGapsInterpolate - int
                Number of consecutive gaps that will be interpolated.
            RemainingData - int
                Percentage (0-100) of the time series that should remain after pre-processing. If less than this quantity, partitioning is not implemented.
            steadyness - bool
                If True, Foken's stationarity test is implemented to check if the data is stationary. If False, the test is not
                implemented. The test is only informative and does not remove data, which is left to the user's discretion.
            saveprocessed - bool
                If True, the processed data is saved to a CSV file.
            time_lag_correction - bool
                If True, a time lag correction is applied to the CO2 and H2O time series relative to the W time series.
            max_lag_seconds - int
                Maximum time lag in seconds to consider for correlation. Defaults to 5 seconds.
            type_lag - str
                Specifies the type of lag to consider. Options are 'positive', 'negative', or 'both'. Defaults to 'positive'.
                'Positive' means that CO2 and H2O lag behind W as expected in closed-path systems when the tube delays the signal.


    Notes: Available Partitioning Methods
        - Conditional Eddy Covariance (CEC)
        - Modified Relaxed Eddy Accumulation (MREA)
        - Flux Variance Similarity (FVS)
        - Conditional Eddy Accumulation (CEA)
        - Conditional Eddy Covariance + WUE (CECw)

    CEC, CEA, and MREA only need time series of w_p, co2_p, h2o_p. The remaining quantities (e.g., P, T, Tv, etc.) are only needed if the
    water use efficiency (WUE) is computed for the FVS and CECw method. Alternatively, an external WUE can be used; in this case, FVS and CECw
    will only need time series of w_p, h2o_p, co2_p.
    """

    def __init__(self, hi, zi, freq, length, df, PreProcessing, argsQC):
        self.hi = hi * ureg.meter
        self.zi = zi * ureg.meter
        self.data = df
        self.freq = freq / ureg.second
        self.length = length * ureg.minute
        self.valid = True
        self.default_argsQC = {
            "time_lag_correction": False,  # If True, a time lag correction is applied to the CO2 and H2O time series relative to the W time series
            "max_lag_seconds": 5,  # Maximum time lag in seconds to consider for correlation
            "saveplotlag": False,  # If True, saves a plot of the cross-correlation function between the CO2 and H2O time series with respect to the W time series
            "type_lag": "positive",  # Specifies the type of lag to consider ('negative', 'positive', or 'both')
            "fluctuations": "LD",  # Method to compute fluctuations: 'LD' (linear detrending), 'BA' (block average), 'FL' (filter low frequencies)
            "filtercut": 5,  # Cutoff timescale to filter low frequencies (in minutes). Needed when FL is selected as fluctuation method
            "density_correction": True,  # If True, density corrections are implemented during pre-processing (depends on type of gas analyzer used)
            "maxGapsInterpolate": 5,  # Intervals of up to 5 missing values are filled by linear interpolation
            "RemainingData": 95,  # Only proceed with partioning if 95% of initial data is available after pre-processing
            "steadyness": False,  # Compute statistic to check stationarity (will not delete data based on this test)
            "saveprocessed": False,  # If True, save the intermediate processed data including all corrections and fluctuations
        }

        self.argsQC = {**self.default_argsQC, **argsQC}

        self.units = {
            # Raw data
            "u": ureg.meter / ureg.second,
            "v": ureg.meter / ureg.second,
            "w": ureg.meter / ureg.second,
            "Ts": ureg.degC,
            "co2": ureg.milligram / ureg.meter**3,
            "h2o": ureg.gram / ureg.meter**3,
            "Tair": ureg.degC,
            "P": ureg.kilopascal,
            # Variables computed during pre-processing
            "w_p": ureg.meter / ureg.second,
            "u_p": ureg.meter / ureg.second,
            "v_p": ureg.meter / ureg.second,
            "T": ureg.degC,
            "Tv": ureg.degC,
            "co2_p": ureg.milligram / ureg.meter**3,
            "h2o_p": ureg.gram / ureg.meter**3,
            "Ts_p": ureg.kelvin,
            "Tv_p": ureg.kelvin,
            "rho_moist_air": ureg.kilogram / ureg.meter**3,
        }

        self._checkMissingdata(percData=self.argsQC.get("RemainingData"))
        self._checkUnits()
        if PreProcessing:
            self._checkPhysicalBounds()
            self._despike()
            self._rotation()
            if self.argsQC.get("time_lag_correction"):
                self._time_lag_correction(
                    max_lag_seconds=self.argsQC.get("max_lag_seconds"),
                    type_lag=self.argsQC.get("type_lag"),
                    saveplotlag=self.argsQC.get("saveplotlag"),
                )
            self._fluctuations(
                method=self.argsQC.get("fluctuations"),
                filter_cut=self.argsQC.get("filtercut"),
            )
            if self.argsQC.get("density_correction"):
                self._densityCorrections(method=self.argsQC.get("fluctuations"))
            self._fillGaps(self.argsQC.get("maxGapsInterpolate"))
            if self.argsQC.get("steadyness"):
                self._steadynessTest()
        self._checkMissingdata(percData=self.argsQC.get("RemainingData"), dropna_=True)

    def _checkUnits(self):
        """Check if units of temperature, CO2, H2O and pressure are correct."""
        temp_range = [0, 70]  # C
        co2_range = [200, 1200]  # mg/m3
        h2o_range = [0, 50]  # g/m3
        press_range = [60, 150]  # kPa

        # Check that not all values are NaN
        if self.data.isnull().all().all():
            return None

        mean_Ts = self.data[self.data["Ts"] > 0]["Ts"].median()
        mean_co2 = self.data[self.data["co2"] > 0]["co2"].median()
        mean_h2o = self.data[self.data["h2o"] > 0]["h2o"].median()
        mean_P = self.data[self.data["P"] > 0]["P"].median()

        if not temp_range[0] < mean_Ts < temp_range[1]:
            raise ValueError(
                f"Mean sonic temperature {mean_Ts} not in Celsius or data quality is poor\n"
            )
        if not co2_range[0] < mean_co2 < co2_range[1]:
            raise ValueError(
                f"Mean CO2 {mean_co2} not in mg/m3 or data quality is poor\n"
            )
        if not h2o_range[0] < mean_h2o < h2o_range[1]:
            raise ValueError(
                f"Mean H2O {mean_h2o} not in g/m3 or data quality is poor\n"
            )
        if not press_range[0] < mean_P < press_range[1]:
            raise ValueError(
                f"Mean atm pressure {mean_P} not in kPa or data quality is poor\n"
            )

    def _checkMissingdata(self, percData, dropna_=False):
        """
        Checks how many missing points are present and only accepts periods when valid data points >= percData.

        Parameters
        ----------
        percData : int
            Percentage of the data that needs to be valid (i.e., excluding gaps) in order to implement partitioning.
            If less than percData is available, the entire half-hour period is discarded. Must be between 0 and 100.

        Stores
        ------
        self.valid_data : float
            The percentage of valid data points.
        """
        maxNAN, indMAX = (  # noqa: F841
            self.data.isnull().sum().max(),
            self.data.isnull().sum().idxmax(),
        )
        total_size = (
            self.freq.magnitude * self.length.magnitude * 60
        )  # total number of points in period
        self.valid_data = ((total_size - maxNAN) / total_size) * 100
        if (self.valid_data < percData) and self.valid:
            self.valid = False
            raise ValueError(
                f"*** Too many missing points {maxNAN}. Less than {percData}\% is available for partitioning. Delete period and try again.\n"
            )
        if dropna_:
            self.data.dropna(inplace=True)

    def _checkPhysicalBounds(self):
        """
        Sets values outside a physically realistic range to NaN.

        If additional variables other than the required ones are passed to the code, their physical bounds need to be
        added to the dictionary `_bounds`. Units must match those of the input data.

        Attributes
        ----------
        self.data : pandas.DataFrame
            DataFrame containing the input data with each variable in a column.

        Notes
        -----
        The `_bounds` dictionary contains the physical bounds for the required variables:
            - "u" : (-20, 20) m/s
            - "v" : (-20, 20) m/s
            - "w" : (-20, 20) m/s
            - "Ts" : (-10, 50) Celsius
            - "co2" : (0, 1500) mg/m3
            - "h2o" : (0, 40) g/m3
            - "P" : (60, 150) kPa

        For each variable in `self.data`, if the variable is in `_bounds`, values outside the specified bounds are set to NaN.
        """
        _bounds = {
            "u": (-20, 20),
            "v": (-20, 20),
            "w": (-20, 20),  # m/s
            "Ts": (-10, 50),  # Celsius
            "co2": (0, 1500),  # mg/m3
            "h2o": (0, 40),  #  g/m3
            "P": (60, 150),  # kPa
        }

        for _var in self.data.columns:
            if _var in _bounds.keys():
                self.data.loc[
                    (self.data[_var] < _bounds[_var][0])
                    | (self.data[_var] > _bounds[_var][1]),
                    _var,
                ] = np.nan

    def _despike(self):
        """
        Replaces outliers with NaN values.

        Points are only considered outliers if no more than 8 points in sequence are above a threshold (see `find_spikes`
        in `auxfunctions.py`). Implements the test described in section 3.4 of:

            E. Zahn, T. L. Chor, N. L. Dias, A Simple Methodology for Quality Control of Micrometeorological Datasets,
            American Journal of Environmental Engineering, Vol. 6 No. 4A, 2016, pp. 135-142. doi: 10.5923/s.ajee.201601.20

        Attributes
        ----------
        self.data : pandas.DataFrame
            DataFrame containing the input data with each variable in a column.

        Notes
        -----
        The following steps are performed:

        1. Linear detrend of time series for variables: "co2", "h2o", "Ts", "w", "u", "v".
        2. Separation into 2-minute windows.
        3. Identification and replacement of spikes with NaN values for the above variables.
        """

        aux = self.data[["co2", "h2o", "Ts", "w", "u", "v"]].copy()
        tt = np.arange(aux["co2"].index.size)

        # 1st: linear detrend time series ------------------
        for _var in ["co2", "h2o", "Ts", "w", "u", "v"]:
            aux[_var] = LinearDetrend(tt, aux[_var].values)
        del tt

        # 2nd: Separate into 2-min windows -----------------
        TwoMinGroups = aux.groupby(pd.Grouper(freq="5Min"))
        TwoMinGroups = [
            TwoMinGroups.get_group(x)
            for x in TwoMinGroups.groups
            if TwoMinGroups.get_group(x).index.size > 10
        ]
        del aux

        for i in range(len(TwoMinGroups)):
            aux_group = TwoMinGroups[i].copy()
            getSpikes = aux_group.apply(find_spikes)

            for _var in ["co2", "h2o", "Ts", "w", "u", "v"]:
                for vdate in getSpikes[_var]:
                    self.data.loc[vdate, _var] = np.nan
        del TwoMinGroups

    def _rotation(self):
        """
        Performs rotation of coordinates using the double rotation method.

        Overwrites the velocity field (u, v, w) with the rotated coordinates.

        References
        ----------
        [Include relevant references here]

        Attributes
        ----------
        self.data : pandas.DataFrame
            DataFrame containing the input data with velocity components u, v, and w.

        Notes
        -----
        The double rotation method aligns the coordinate system with the mean flow direction. The steps involved are:

        1. Calculation of mean velocities.
        2. Calculation of the angles between mean velocities.
        3. Rotation of coordinates using these angles.
        4. Updating the DataFrame with the rotated velocities.
        """
        aux = self.data[["u", "v", "w"]].copy()
        Umean = aux.mean()
        # Calculating the angles between mean velocities
        hspeed = np.sqrt(Umean["u"] ** 2.0 + Umean["v"] ** 2.0)
        alfax = math.atan2(Umean["v"], Umean["u"])
        alfaz = math.atan2(Umean["w"], hspeed)
        # Rotating coordinates
        aux["u_new"] = (
            math.cos(alfax) * math.cos(alfaz) * aux["u"]
            + math.sin(alfax) * math.cos(alfaz) * aux["v"]
            + math.sin(alfaz) * aux["w"]
        )
        aux["v_new"] = -math.sin(alfax) * aux["u"] + math.cos(alfax) * aux["v"]
        aux["w_new"] = (
            -math.cos(alfax) * math.sin(alfaz) * aux["u"]
            - math.sin(alfax) * math.sin(alfaz) * aux["v"]
            + math.cos(alfaz) * aux["w"]
        )
        # Update rotated velocities in dataframe
        self.data["w"] = aux["w_new"].copy()
        self.data["u"] = aux["u_new"].copy()
        self.data["v"] = aux["v_new"].copy()
        del aux, hspeed, alfax, alfaz

    def _time_lag_correction(self, max_lag_seconds, type_lag, saveplotlag):
        """
        Corrects the time lag between the 'co2' and 'h2o' time series relative to the 'w' time series
           by shifting the 'co2' and 'h2o' time series accordingly while keeping the 'w' time series fixed.

        Parameters:
        - max_lag_seconds (int): The maximum time lag in seconds to consider for correlation. Defaults to 5 seconds.
        - type_lag (str): Specifies the type of lag to consider. Options are 'positive', 'negative', or 'both'.
                          Defaults to 'positive', meaning only positive lags will be considered, implying that co2 and h2o lag behind w
        - saveplotlag (bool): If True, saves a plot of the cross-correlation function between the 'co2' and 'h2o' time series with respect to the 'w' time series.

        This method calculates the optimal lag for the 'co2' and 'h2o' time series with respect to the 'w' time series
        and shifts them accordingly to align with the 'w' time series. The lags are determined by finding the maximum
        absolute correlation.

        Returns:
        - None: The method updates the 'co2' and 'h2o' columns in the instance's data attribute in place.
        """
        lag_co2, lag_h2o = max_time_lag_crosscorrel(
            df=self.data[["co2", "h2o", "w"]],
            sampling_freq=self.freq.magnitude,
            max_lag_seconds=max_lag_seconds,
            type_lag=type_lag,
            saveplotlag=saveplotlag,
        )
        self.data["co2"] = self.data["co2"].shift(-lag_co2)
        self.data["h2o"] = self.data["h2o"].shift(-lag_h2o)

    def _fluctuations(self, method, filter_cut):
        """
        Computes turbulent fluctuations, x' = x - X, where X is the average.

        Only variables required by the partitioning algorithms are included.

        Parameters
        ----------
        method : str
            Method to compute X:
            'BA' : Block average
            'FL' : Filter low frequencies above filter_cut (in min)
            'LD' : Linear detrending
        filter_cut : int
            Cutoff time in minutes for the low-pass filter. Only used if method is 'FL'.

        Raises
        ------
        TypeError
            If the method to extract fluctuations is not 'LD' or 'BA'.

        Attributes
        ----------
        self.data : pandas.DataFrame
            DataFrame containing the input data with variables "u", "v", "w", "co2", "h2o", "Ts".

        Notes
        -----
        Adds the time series of fluctuations (variable_name + '_p') to the DataFrame.
        """
        Lvars = ["u", "v", "w", "co2", "h2o", "Ts"]

        if method == "LD":
            tt = np.arange(self.data.index.size)
            for ii, _var in enumerate(Lvars):
                self.data[_var + "_p"] = LinearDetrend(tt, self.data[_var].values)
            del tt
        elif method == "FL":
            for ii, _var in enumerate(Lvars):
                self.data[_var + "_p"] = FilterLowFrequencies(
                    self.data[_var].values, self.freq.magnitude, filter_cut
                )
        elif method == "BA":
            for ii, _var in enumerate(Lvars):
                self.data[_var + "_p"] = self.data[_var] - self.data[_var].mean()
        else:
            raise TypeError(
                "Method to extract fluctuations must be 'LD', 'BA' or 'FL \n"
            )

    def _densityCorrections(self, method):
        """
        Applies density correction to the fluctuations of CO2 (co2_p) and H2O (h2o_p).

        Follows the method described in:
            Detto, M. and Katul, G. G., 2007. "Simplified expressions for adjusting higher-order turbulent statistics obtained from open path gas analyzers",
            Boundary-Layer Meteorology, 10.1007/s10546-006-9105-1.

        Note that this correction is necessary only when CO2 and H2O were measured by an open gas analyzer and their outputs are mass/molar densities (e.g., mg/m3).

        Parameters
        ----------
        method : str
            Method to compute temperature fluctuations:
            'LD' : Linear detrending
            'BA' : Block average

        Attributes
        ----------
        self.data : pandas.DataFrame
            DataFrame containing the input data with variables "P", "Ts", "co2", "h2o", "u", "v", "w".

        Notes
        -----
        - Calculates air density, thermodynamic temperature, and virtual temperature.
        - Computes temperature fluctuations based on the specified method.
        - Applies corrections to CO2 and H2O fluctuations using the specified method.

        Raises
        ------
        TypeError
            If the method to compute temperature fluctuations is not 'LD' or 'BA'.
        """
        # Calculate air density------------------------------------------------
        Rd = Constants.Rd.magnitude  # J/kg.K
        self.data["rho_moist_air"] = (
            1000 * self.data["P"] / (Rd * (273.15 + self.data["Ts"]))
        )  # mean density of moist air [kg/m3] *** assume Ts is the same as Tv ***
        self.data["rho_dry_air"] = (
            self.data["rho_moist_air"] - self.data["h2o"] * 10**-3
        )  # density of dry air [kg/m3]
        # Obtain termodynamic and virtual temperatures ------------------------
        q = (
            self.data["h2o"] * 10**-3 / self.data["rho_dry_air"]
        )  # Instantaneous mixing ratio kg/kg
        self.data["T"] = (self.data["Ts"] + 273.15) / (
            1.0 + 0.51 * q
        ) - 273.15  # termodynamic temperature from sonic temperature [C]
        self.data["Tv"] = (self.data["T"] + 273.15) * (
            1.0 + 0.61 * q
        ) - 273.15  # virtual temperature from termo temperature [C]

        # We also need the fluctuations of temperature ------------------------
        if method == "LD":
            self.data["T_p"] = LinearDetrend(
                np.arange(self.data.index.size), self.data["T"].values
            )
            self.data["Tv_p"] = LinearDetrend(
                np.arange(self.data.index.size), self.data["Tv"].values
            )
        else:
            self.data["T_p"] = self.data["T"] - self.data["T"].mean()
            self.data["Tv_p"] = self.data["Tv"] - self.data["Tv"].mean()
        meanT = self.data["T"].mean()  # mean real temperature [C]

        # Aditional variables --------------------------------------------------
        mu = (
            Constants.MWdryair.magnitude / Constants.MWvapor.magnitude
        )  # ratio of dry air mass to water vapor mass
        mean_co2 = self.data["co2"].mean() * 10**-6  # mean co2 [kg/m3]
        mean_h2o = self.data["h2o"].mean() * 10**-3  # mean h2o [kg/m3]
        sigmaq = (
            mean_h2o / self.data["rho_dry_air"].mean()
        )  # mixing ratio [kg_wv/kg_air]
        sigmac = (
            mean_co2 / self.data["rho_dry_air"].mean()
        )  # mixing ratio [kg_co2/kg_air]

        # Finally, apply the corrections ---------------------------------------
        self.data["co2_p"] = (
            self.data["co2_p"]
            + (
                mu * sigmac * self.data["h2o_p"] * 10**-3
                + mean_co2 * (1.0 + mu * sigmaq) * self.data["T_p"] / (meanT + 273.15)
            )
            * 10**6
        )  # [mg/m3]
        self.data["h2o_p"] = (
            self.data["h2o_p"]
            + (
                mu * sigmaq * self.data["h2o_p"] * 10**-3
                + mean_h2o * (1.0 + mu * sigmaq) * self.data["T_p"] / (meanT + 273.15)
            )
            * 1000
        )  # [ g/m3]
        del mu, mean_co2, mean_h2o, sigmaq, sigmac, q

    def _fillGaps(self, maxGaps):
        """
        Fills gaps (NaN values) in time series using linear interpolation.

        It is recommended that only small gaps be interpolated.

        Parameters
        ----------
        maxGaps : int
            Number of consecutive missing gaps that can be interpolated. Should be 0 < maxGaps < 20.

        Raises
        ------
        TypeError
            If maxGaps is greater than 20, indicating that too many consecutive points are being interpolated.

        Attributes
        ----------
        self.data : pandas.DataFrame
            DataFrame containing the input data with potential gaps.
        """
        if maxGaps > 20:
            raise TypeError(
                "Too many consecutive points to be interpolated. Consider a smaller gap (up to 20 points). \n"
            )

        self.data.interpolate(
            method="linear", limit=maxGaps, limit_direction="both", inplace=True
        )

    def _steadynessTest(self):
        """
        Implements a stationarity test described in section 5 of:
            Thomas Foken and B. Wichura, "Tools for quality assessment of surface-based flux measurements",
            Agricultural and Forest Meteorology, Volume 78, Issues 1–2, 1996, Pages 83-105.

        Computes the stationarity statistic:
            stat = | (average_cov_5min - cov_30min) / cov_30min | * 100 %
        where cov is the covariance between any two variables.

        Foken argues that steady state conditions can be assumed if stat < 30 %.
        This variable can be used as a criterion for data quality and its compliance with EC requirements (steadiness).

        Reference:
        Foken, T., Micrometeorology, https://doi.org/10.1007/978-3-540-74666-9, p. 175.

        Creates a dictionary with the steadiness statistics (in %) for variances and covariances.

        Attributes
        ----------
        self.FokenStatTest : dict
            Dictionary containing the steadiness statistics for various fluxes and variances:
            - 'wc': statistic for w'c' (total CO2 flux)
            - 'wq': statistic for w'q' (total H2O flux)
            - 'wT': statistic for w'T' (sonic temperature flux)
            - 'ww': statistic for w'w' (variance of w)
            - 'cc': statistic for c'c' (variance of CO2)
            - 'qq': statistic for q'q' (variance of H2O)
            - 'tt': statistic for t't' (variance of sonic temperature)

        Notes
        -----
        - Computes 5-minute window statistics.
        - Compares 30-minute statistics to the average of 5-minute windows.
        """
        # Five minute window statistics -------------------------
        stats5min = self.data.groupby(pd.Grouper(freq="5Min")).apply(Stats5min).dropna()
        aver_5min = stats5min.mean()
        # Statistic for entire window (i.e., 30 min) ------------
        cov_all = self.data.cov()
        var_all = self.data.var()
        stats_all = pd.Series(
            [
                cov_all["w_p"]["co2_p"],
                cov_all["w_p"]["h2o_p"],
                cov_all["w_p"]["Ts_p"],
                var_all["w_p"],
                var_all["co2_p"],
                var_all["h2o_p"],
                var_all["Ts_p"],
            ],
            index=["wc", "wq", "wt", "ww", "cc", "qq", "tt"],
        )
        # Compare 30-min to 5-min windows -----------------------
        stat_fk = dict(abs((stats_all - aver_5min) / stats_all) * 100)
        self.FokenStatTest = {
            "fkstat_%s" % svar: stat_fk[svar]
            for svar in ["wc", "wq", "wt", "ww", "cc", "qq", "tt"]
        }
        # print results as a table with wc wq wt ww cc qq tt on top and their respective values below
        print("------------------------")
        print(" Foken's Stationarity Test (%)")
        print(pd.DataFrame(self.FokenStatTest, index=[0]))
        print("\n")

    def TurbulentStats(self):
        """
        # Calculate turbulent statistics (scales, standard deviations, and correlations).

        Returns
        ----------
        self.turbstats : dict

            Keys:
            -----
                'ustar': float
                    friction velocity [m/s]
                'cstar': float
                    scale for CO2 [mg/m3]
                'qstar': float
                    scale for H2O [g/m3]
                'tstar': float
                    scale for temperature [K]
                'zeta': float
                    Monin-Obukhov stability parameter
                'std_t': float
                    standard deviation of temperature [K]
                'std_q': float
                    standard deviation of H2O [g/m3]
                'std_c': float
                    standard deviation of CO2 [mg/m3]
                'std_w': float
                    standard deviation of w [m/s]
                'std_u': float
                    standard deviation of u [m/s]
                'std_v': float
                    floatstandard deviation of v [m/s]
                'rqc': float
                    correlation between H2O and CO2
                'rqt': float
                    correlation between H2O and temperature
                'rct': float
                    correlation between CO2 and temperature
                'Fc': float
                    covariance between w and CO2 [mg/m2/s]
                'LE': float
                    latent heat flux [W/m2]
                'H': float
                    sensible heat flux [W/m2]
        """
        aux = self.data[["u_p", "v_p", "w_p", "co2_p", "h2o_p", "Ts_p"]].copy()
        matrixCov = aux.cov()  # Covariance matrix
        matrixSTD = aux.std()  # Covariance matrix
        matrixCorr = aux.corr(method="pearson")  # Correlation matrix

        # Calculate scales -------------------------------------------------------
        ustar = (matrixCov["u_p"]["w_p"] ** 2.0 + matrixCov["w_p"]["v_p"] ** 2.0) ** (
            1.0 / 4.0
        )  # Friction velocity [m/s]
        cstar = matrixCov["co2_p"]["w_p"] / ustar * self.units["co2"]  # mg/m3
        qstar = matrixCov["h2o_p"]["w_p"] / ustar * self.units["h2o"]  #  g/m3
        tstar = matrixCov["Ts_p"]["w_p"] / ustar * self.units["Ts_p"]  # K
        d = (2.0 / 3.0) * self.hi  # Displacement height [m]
        Qv = (
            matrixCov["Ts_p"]["w_p"] * self.units["Ts_p"] * self.units["w"]
        )  # kinematic heat flux [K m/s]
        meanTkelvin = (self.data["Tv"].mean() + 273.15) * self.units["Ts_p"]
        rho_moist_air = (
            self.data["rho_moist_air"].mean() * self.units["rho_moist_air"]
        )  # kg/m3
        LE_wm2 = matrixCov["w_p"]["h2o_p"] * (10**-3) * Constants.Lv.magnitude
        LE_wm2 = LE_wm2 * ureg.watt / ureg.meter**2
        H_wm2 = (
            matrixCov["w_p"]["Ts_p"]
            * (10**3)
            * Constants.cp.magnitude
            * rho_moist_air.magnitude
        )
        H_wm2 = H_wm2 * ureg.watt / ureg.meter**2

        if self.hi.magnitude < self.zi.magnitude:
            zeta = (
                -0.4
                * Constants.g
                * (self.zi - d)
                * Qv
                / ((meanTkelvin) * ((ustar * self.units["u"]) ** 3))
            )
        else:
            zeta = np.nan

        self.turbstats = {
            "ustar": ustar * self.units["u"],
            "cstar": cstar,
            "qstar": qstar,
            "tstar": tstar,
            "zeta": zeta,
            "std_t": matrixSTD["Ts_p"] * self.units["Ts_p"],
            "std_q": matrixSTD["h2o_p"] * self.units["h2o"],
            "std_c": matrixSTD["co2_p"] * self.units["co2"],
            "std_w": matrixSTD["w_p"] * self.units["w"],
            "std_u": matrixSTD["u_p"] * self.units["u"],
            "std_v": matrixSTD["v_p"] * self.units["v"],
            "rqc": matrixCorr["h2o_p"]["co2_p"] * ureg.dimensionless,
            "rqt": matrixCorr["h2o_p"]["Ts_p"] * ureg.dimensionless,
            "rct": matrixCorr["co2_p"]["Ts_p"] * ureg.dimensionless,
            "Fc": matrixCov["w_p"]["co2_p"] * self.units["co2"] * self.units["w"],
            "LE": LE_wm2,
            "H": H_wm2,
        }

    def WaterUseEfficiency(self, ppath="C3"):
        """
        Calculates water use efficiency in kg_co2/kg_h2o.

        Main references:
        - Scanlon and Sahu 2008, Water Resources Research
          "On the correlation structure of water vapor and carbon dioxide in
          the atmospheric surface layer: A basis for flux partitioning"
        - Parts of the code were adapted from Skaggs et al. 2018, Agr For Met
          "Fluxpart: Open source software for partitioning carbon dioxide and water vapor fluxes"
          https://github.com/usda-ars-ussl/fluxpart
        - Optimization model for W from Scanlon et al., 2019, Agr. For. Met.
          "Correlation-based flux partitioning of water vapor and carbon dioxide fluxes:
          Method simplification and estimation of canopy water use efficiency"

        Parameters
        ----------
        ppath : str
            Type of photosynthesis ('C3' or 'C4').

        Models
        
            Computes the water use efficiency (eq A1 in Scanlon and Sahu, 2008):
                - wue = 0.65 * (c_c - c_s) / (q_c - q_s)
                - c_c (kg/m3) and q_c (kg/m3) are near canopy concentrations of CO2 and H2O
                    - Estimated from log profiles (eq A2a in Scanlon and Sahu, 2008).
                - c_s (kg/m3) and q_s (kg/m3) are stomata concentrations of CO2 and H2O
                    - q_s is assumed to be at saturation.
                    - c_s is parameterized from different models (Skaggs et al., 2018; Scanlon et al., 2019).
        
            The following models for c_s are implemented
    
            const_ppm:
                - Concentrations in kg/m3 are computed from a constant value in ppm.
                - Values from Campbell and Norman, 1998, p. 150.
                  Campbell, G. S. and Norman, J. M. (1998). An Introduction to Environmental Biophysics. Springer, New York, NY.
                - c_s = 280 ppm (C3 plants).
                - c_s = 130 ppm (C4 plants).
        
            const_ratio:
                - The ratio of near canopy and stomata CO2 concentrations is assumed constant (c_s/c_c = constant).
                - Constants from Sinclair, T. R., Tanner, C. B., and Bennett, J. M. (1984).
                  Water-use efficiency in crop production. BioScience, 34(1):36–40.
                - c_s/c_c = 0.70 for C3 plants.
                - c_s/c_c = 0.44 for C4 plants.
        
            linear:
                - The ratio of near canopy and stomata CO2 concentrations is a linear function of VPD.
                - Based on the results of Morison, J. I. L. and Gifford, R. M. (1983).
                  Stomatal sensitivity to carbon dioxide and humidity. Plant Physiology, 71(4):789–796.
                  Estimated constants from Skaggs et al (2018).
                - c_s/c_c = a - b * D
                - a, b = 1, 1.6*10-4 Pa-1 for C3 plants.
                - a, b = 1, 2.7*10-4 Pa-1 for C4 plants.
                - D (Pa) is vapor pressure deficit based on leaf-temperature.
    
            sqrt:
                - The ratio of near canopy and stomata CO2 concentrations is proportional
                  to the 1/2 power of VPD.
                - Model by Katul, G. G., Palmroth, S., and Oren, R. (2009).
                  Leaf stomatal responses to vapour pressure deficit under current and CO2-enriched atmosphere
                  explained by the economics of gas exchange. Plant, Cell & Environment, 32(8):968–979.
                - c_s/c_c = 1 - sqrt(1.6 * lambda * D / c_c)
                - lambda = 22e-9 kg-CO2 / m^3 / Pa for C3 plants (from Skaggs et al., 2018).
                - Not available for C4 plants.
        
            opt:
                - Optimization model proposed by Scanlon et al (2019).
                - Does not need extra parameters.
                - Only available for C3 plants.
        
        Returns
        ----------
        self.wue : dict
            Dictionary containing the water use efficiency from different methods:
            - 'const_ppm': float
                WUE from constant ppm [kg_co2/kg_h2o].
            - 'const_ratio': float
                WUE from constant ratio [kg_co2/kg_h2o].
            - 'linear': float
                WUE from linear model [kg_co2/kg_h2o].
            - 'sqrt': float
                WUE from sqrt model [kg_co2/kg_h2o].
            - 'opt': float
                WUE from optimization model [kg_co2/kg_h2o].
        """

        # Create a copy of the dataframe with variables that will be needed
        aux = self.data.copy()

        # Create dictionary that will store water use efficiency from different methods
        self.wue = {
            "const_ppm": np.nan,
            "const_ratio": np.nan,
            "linear": np.nan,
            "sqrt": np.nan,
            "opt": np.nan,
        }

        # Statistics  --------------------
        matrixCov = aux.cov()  # Covariance matrix
        varq = (
            matrixCov["h2o_p"]["h2o_p"] * 10**-6
        )  # variance of h2o fluctuations (kg/m3)^2
        varc = (
            matrixCov["co2_p"]["co2_p"] * 10**-12
        )  # variance of co2 fluctuations (kg/m3)^2
        sigmac = varc**0.5  # Standard deviation of co2 [kg/m3]
        sigmaq = varq**0.5  # Standard deviation of h2o [kg/m3]
        corr_qc = np.corrcoef(aux["h2o_p"].values, aux["co2_p"].values)[
            1, 0
        ]  # correlation coefficient between q and c
        cov_wq = matrixCov["w_p"]["h2o_p"] * 10**-3  # covariance of w and q (kg/m^2/s);
        cov_wc = matrixCov["w_p"]["co2_p"] * 10**-6  # covariance of w and c (kg/m^2/s);

        # Mean variables and parameterizations ------------
        leaf_T = (
            aux["T"].mean() + 273.15
        )  # set leaf temperature == air temperature (Kelvin)
        P = aux["P"].mean() * 10**3  # mean atmospheric pressure (Pa)
        mean_rho_vapor = (
            aux["h2o"].mean() * 10**-3
        )  # mean vapor density (kg/m^3) -- NOT THE FLUCTUATIONS!
        mean_rho_co2 = (
            aux["co2"].mean() * 10**-6
        )  # mean carbon dioxide concentration (kg/m^3) -- NOT THE FLUCTUATIONS!
        mean_Tv = aux["Tv"].mean()  # mean virtual temperature in Celsius
        rho_totair = P / (
            Constants.Rd.magnitude * (mean_Tv + 273.15)
        )  # moist air density (kg/m^3)
        ustar = (matrixCov["u_p"]["w_p"] ** 2.0 + matrixCov["w_p"]["v_p"] ** 2.0) ** (
            1.0 / 4.0
        )  # friction velocity [m/s]
        Qv = matrixCov["Tv_p"]["w_p"]  # kinematic virtual temperature flux [K m/s]
        dd = self.hi.magnitude * (2 / 3)  # displacement height [m]
        zeta = (
            -Constants.VON_KARMAN
            * Constants.g.magnitude
            * (self.zi.magnitude - dd)
            * Qv
            / ((mean_Tv + 273.15) * (ustar**3))
        )  # Monin-Obukhov stability parameter
        zv = 0.2 * (0.1 * self.hi.magnitude)  # Roughness parameter

        # 1 - Finding near canopy concentrations --------------------------------------------------
        # Calculating Monin Obukhov nondimensional function
        # Following Fluxpart - Skaggs et al., 2018

        # Limiting zeta to avoid numerical errors
        zeta_min = -5.0
        zeta_max = 5.0
        if zeta < zeta_min:
            zeta = zeta_min
        elif zeta > zeta_max:
            zeta = zeta_max

        if zeta < -0.04:
            psi_v = 2.0 * np.log((1 + (1 - 16.0 * zeta) ** 0.5) / 2)
        elif zeta <= 0.04:
            psi_v = 0.0
        else:
            psi_v = -5.0 * zeta

        arg = (
            (np.log((self.zi.magnitude - dd) / zv) - psi_v)
            / Constants.VON_KARMAN
            / ustar
        )
        ambient_h2o = (
            mean_rho_vapor + cov_wq * arg
        )  # mean h2o concentration near canopy [kg/m3] - using log-law profiles
        ambient_co2 = (
            mean_rho_co2 + cov_wc * arg
        )  # mean co2 concentration near canopy [kg/m3] - using log-law profiles

        # 2 - Finding intercelular concentrations -------------------------------------------------
        esat = sat_vapor_press(
            leaf_T
        )  # Intercellular saturation vapor pressure 'esat' [Pa]

        # Intercellular vapor density
        eps = Constants.MWvapor.magnitude / Constants.MWdryair.magnitude
        inter_h2o = (
            rho_totair * eps * esat / (P - (1 - eps) * esat)
        )  # mean h2o concentration inside stomata [kg/m3]

        # vapor pressure deficit
        vpd = vapor_press_deficit(ambient_h2o, leaf_T, Constants.Rvapor.magnitude)

        if vpd < 0:
            raise ValueError(
                "Negative vapor pressure deficit. Check the input data and try again or remove period.\n"
            )

        # Calculating inside stomata co2 concentration

        ci_mod_const_ppm = ci_const_ppm(
            P,
            leaf_T,
            Constants.Rco2.magnitude,
            Constants.wue_constants[ppath]["const_ppm"].magnitude,
        )
        ci_mod_const_ratio = cica_const_ratio(
            ambient_co2, Constants.wue_constants[ppath]["const_ratio"]
        )
        ci_mod_linear = cica_linear(
            ambient_co2,
            vpd,
            Constants.wue_constants[ppath]["linear"][0],
            Constants.wue_constants[ppath]["linear"][1].magnitude,
        )
        ci_mod_sqrt = cica_sqrt(
            ambient_co2, vpd, Constants.wue_constants[ppath]["sqrt"].magnitude
        )

        # 3 - Compute water use efficiency ----------------------------------------------------------------------------
        for ci, mod in [
            [ci_mod_const_ppm, "const_ppm"],
            [ci_mod_const_ratio, "const_ratio"],
            [ci_mod_linear, "linear"],
            [ci_mod_sqrt, "sqrt"],
        ]:
            coef = 1.0 / Constants.diff_ratio
            wuei = coef * (ambient_co2 - ci) / (ambient_h2o - inter_h2o)
            self.wue[mod] = wuei

        # Optimization model from Scanlon et al., 2019 - only applicable to C3 plants
        if ppath == "C3":
            m = -(varc * cov_wq - corr_qc * sigmaq * sigmac * cov_wc) / (
                varq * cov_wc - corr_qc * sigmaq * sigmac * cov_wq
            )
            vpdm = vapor_press_deficit_mass(
                ambient_h2o, leaf_T, Constants.Rvapor.magnitude
            )
            if vpdm < 0 or m < 0:
                self.wue["opt"] = (
                    np.nan
                )  # In case of negative vapor pressure deficit or m
            else:
                self.wue["opt"] = (
                    Constants.diff_ratio * vpdm * m
                    - np.sqrt(
                        Constants.diff_ratio
                        * vpdm
                        * m
                        * (ambient_co2 + Constants.diff_ratio * vpdm * m)
                    )
                ) / (Constants.diff_ratio * vpdm)
        else:
            self.wue["opt"] = np.nan  # Model is not suitable for C4 and CAM plants
        del aux

    def partCEC(self, H=0.0):
        """
        Implements the Conditional Eddy Covariance method proposed by Zahn et al. 2021.

        Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components.

        Parameters
        ----------
        H : float, optional
            Hyperbolic threshold, by default 0.0.

        Attributes
        ----------       
        Attributes: self.fluxesCEC
            Contains all the flux components and status of the calculation.
        
            - ET - float
                Total evapotranspiration (W/m2).
            - T - float
                Plant transpiration (W/m2).
            - Ecec - float
                Soil/surface evaporation (W/m2).
            - Fc - float
                Carbon dioxide flux (mg/m2/s).
            - Rcec - float
                Soil/surface respiration (mg/m2/s).
            - Pcec - float
                Plant net photosynthesis* (mg/m2/s).
            - statuscec - str
                Status of the calculation.

        Notes
        -----
        This component represents carboxylation minus photorespiration and leaf respiration; therefore,
        it is different from gross primary productivity.
        """

        per_points_Q1Q2 = 15  # smallest percentage of points that must be available in the first two quadrants
        per_poits_each = 3  # smallest percentage of points in each quadrant

        # Creates a dataframe with variables of interest and no constraints
        auxET = self.data[["co2_p", "h2o_p", "w_p"]].copy()
        N = auxET.index.size
        total_Fc = np.mean(
            auxET["co2_p"].values * auxET["w_p"].values
        )  # flux [all quadrants] given in mg/(s m2)
        total_ET = (
            (10**-3)
            * Constants.Lv.magnitude
            * np.mean(auxET["h2o_p"].values * auxET["w_p"].values)
        )  # flux [all quadrants] given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE = self.data[
            (self.data["w_p"] > 0)
            & (self.data["co2_p"] > 0)
            & (self.data["h2o_p"] > 0)
            & (
                abs(self.data["co2_p"] / self.data["co2_p"].std())
                > abs(H * self.data["h2o_p"].std() / self.data["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        R_condition_Fc = (
            np.sum(auxE["co2_p"].values * auxE["w_p"].values) / N
        )  # conditional flux [1st quadrant and w'>0] given in mg/(s m2)
        E_condition_ET = (
            (10**-3)
            * Constants.Lv.magnitude
            * np.sum(auxE["h2o_p"].values * auxE["w_p"].values)
            / N
        )  # conditional flux [1st quadrant and w'>0] flux given in  W/m2
        sumQ1 = (
            auxE["w_p"].index.size / N
        ) * 100  # Percentage of points in the first quadrant

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT = self.data[
            (self.data["w_p"] > 0)
            & (self.data["co2_p"] < 0)
            & (self.data["h2o_p"] > 0)
            & (
                abs(self.data["co2_p"] / self.data["co2_p"].std())
                > abs(H * self.data["h2o_p"].std() / self.data["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        P_condition_Fc = (
            np.sum(auxT["co2_p"].values * auxT["w_p"].values) / N
        )  # conditional flux [2nd quadrant and w'>0] given in mg/(s m2)
        T_condition_ET = (
            (10**-3) * Constants.Lv.magnitude * np.sum(auxT["h2o_p"] * auxT["w_p"]) / N
        )  # conditional flux [2nd quadrant and w'>0] flux given in  W/m2
        sumQ2 = (
            auxT["w_p"].index.size / N
        ) * 100  # Percentage of points in the second quadrant

        # Computing flux ratios and flux components of ET and Fc
        E, T, P, R, ratioET, ratioRP = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # First condition: do we have enough points in Q1 and Q2?
        if (sumQ1 + sumQ2) < per_points_Q1Q2:
            self.fluxesCEC = {
                "ETcec": total_ET * ureg.watt / ureg.meter**2,
                "Ecec": E * ureg.watt / ureg.meter**2,
                "Tcec": T * ureg.watt / ureg.meter**2,
                "Fccec": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Pcec": P * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rcec": R * ureg.milligram / ureg.meter**2 / ureg.second,
                # "rRPcec": ratioRP,
                # "rETcec": ratioET,
                "statuscec": "Q1+Q2<20",
            }
        elif (sumQ1 >= per_poits_each) and (sumQ2 >= per_poits_each):
            # ET components
            ratioET = E_condition_ET / T_condition_ET
            T = total_ET / (1.0 + ratioET)
            E = total_ET / (1.0 + 1.0 / ratioET)

            # Fc components
            ratioRP = R_condition_Fc / P_condition_Fc
            P = total_Fc / (1.0 + ratioRP)
            R = total_Fc / (1.0 + 1.0 / ratioRP)
        elif (sumQ1 < per_poits_each) and (sumQ2 > per_poits_each):
            # In this case, all water vapor flux is assumed to be transpiration
            ratioET = 0.0
            T = total_ET
            E = 0.0
            # In this case, all co2 flux is assumed to be photosynthesis
            ratioRP = 0.0
            P = total_Fc
            R = 0.0
        elif (sumQ1 > per_poits_each) and (sumQ2 < per_poits_each):
            # All fluxes are assumed to be from the ground
            ratioET = np.inf
            T = 0.0
            E = total_ET
            # All Fc flux is considered to be respiration
            ratioRP = np.inf
            P = 0.0
            R = total_Fc
        else:
            pass

        # Check CO2 flux components ratio
        #   CO2 fluxes might be noisy in this range
        if -1.2 < ratioRP < -0.8:
            finalstat = "Small ratioRP"
        else:
            finalstat = "OK"

        # Additional constraints may be added based on the strength of the fluxes and other combinations
        self.fluxesCEC = {
            "ETcec": total_ET * ureg.watt / ureg.meter**2,
            "Ecec": E * ureg.watt / ureg.meter**2,
            "Tcec": T * ureg.watt / ureg.meter**2,
            "Fccec": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
            "Pcec": P * ureg.milligram / ureg.meter**2 / ureg.second,
            "Rcec": R * ureg.milligram / ureg.meter**2 / ureg.second,
            # "rRP": ratioRP,
            # "rET": ratioET,
            "statuscec": finalstat,
        }

    def partREA(self, H=0):
        """
        Implements the Modified Relaxed Eddy Accumulation proposed by Thomas et al., 2008 (Agr For Met).

        Estimating daytime subcanopy respiration from conditional sampling methods applied to multi-scalar high frequency turbulence time series
        https://www.sciencedirect.com/science/article/pii/S0168192308000737
        New contraints defined in Zahn et al (2021).

        Parameters
        ----------
        H : float, optional
            Hyperbolic threshold, by default 0.0.

        Attributes
        ----------       
        Attributes: self.fluxesREA
            Dictionary with the following flux components:
            
            - ET - float
                Total evapotranspiration (W/m2).
            - Tmrea - float
                Plant transpiration (W/m2).
            - Emrea - float
                Soil/surface evaporation (W/m2).
            - Fc - float
                Net carbon dioxide flux (mg/m2/s).
            - Rmrea - float
                Soil/surface respiration (mg/m2/s).
            - Pmrea - float
                Plant net photosynthesis* (mg/m2/s).
            - statusmrea - str
                Status of the calculation.

        Notes
        -----
        This component represents carboxylation minus photorespiration and leaf respiration; therefore,
        it is different from gross primary productivity.
        """

        per_points_Q1Q2 = 15  # smallest percentage of points that must be available in the first two quadrants
        per_poits_each = 3  # smallest percentage of points in each quadrant

        # REA parameters ---------------------------------------------------
        wseries = np.array(self.data["w_p"].values)  #  m/s
        cseries = np.array(self.data["co2_p"].values)  # mg/m3
        qseries = np.array(self.data["h2o_p"].values)  #  g/m3
        sigmaw = np.std(wseries)  # standard deviation of vertical velocity (m/s)
        sigmac = np.std(cseries)  # standard deviation of co2 density
        sigmaq = np.std(qseries)  # standard deviation of water vapor density
        beta = sigmaw / (
            np.mean(wseries[wseries > 0]) - np.mean(wseries[wseries < 0])
        )  # similarity constant
        Fc = np.cov(wseries, cseries)[0][1]  # CO2 flux [mg/m2/s]
        ET = (
            np.cov(wseries, qseries)[0][1] * (10**-3) * Constants.Lv.magnitude
        )  # latent heat flux [W/m2]
        NN = len(wseries)  # total number of points

        # For carbon fluxes: numerator and denominator of equation 11 in Thomas et al., 2008
        cnum = cseries[
            (qseries > 0)
            & (cseries > 0)
            & (wseries > 0)
            & ((qseries / sigmaq) > (H * sigmac / cseries))
            & ((cseries / sigmac) > (H * sigmaq / qseries))
        ]
        cdiv = cseries[
            (abs(qseries / sigmaq) > abs(H * sigmac / cseries))
            & (abs(cseries / sigmac) > abs(H * sigmaq / qseries))
            & (wseries > 0)
        ]

        # For water vapor fluxes: numerator and denominator of equation 11 in Thomas et al., 2008
        qnum = qseries[
            (qseries > 0)
            & (cseries > 0)
            & (wseries > 0)
            & ((qseries / sigmaq) > (H * sigmac / cseries))
            & ((cseries / sigmac) > (H * sigmaq / qseries))
        ]
        qdiv = qseries[
            (abs(qseries / sigmaq) > abs(H * sigmac / cseries))
            & (abs(cseries / sigmac) > abs(H * sigmaq / qseries))
            & (wseries > 0)
        ]

        # Count number of points in the first and second quadrant (no H is used here)
        Q1sum = (len(cseries[(qseries > 0) & (cseries > 0) & (wseries > 0)]) / NN) * 100
        Q2sum = (len(cseries[(qseries > 0) & (cseries < 0) & (wseries > 0)]) / NN) * 100

        # Check availability of points in each quadrant
        if (Q1sum + Q2sum) < per_points_Q1Q2:
            self.fluxesREA = {
                "ETmrea": ET * ureg.watt / ureg.meter**2,
                "Fcmrea": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Emrea": np.nan * ureg.watt / ureg.meter**2,
                "Tmrea": np.nan * ureg.watt / ureg.meter**2,
                "Pmrea": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rmrea": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusmrea": "Q1+Q2<20",
            }
        elif (Q1sum >= per_poits_each) and (Q2sum >= per_poits_each):
            # Compute fluxes
            R = beta * sigmaw * (sum(cnum) / len(cdiv))  # Respiration [mg / (s m2)]
            E = beta * sigmaw * (sum(qnum) / len(qdiv))  # Evaporation [g / (s m2)]
            E = E * (10**-3) * Constants.Lv.magnitude  # Latent heat flux [W/m2]
            P = Fc - R  # Photosynthesis  [mg/(s m2)]
            T = ET - E  # Transpiration   [W/m2]
            finalstatus = "OK"

            # To test realistic fluxes
            if E > 1.01 * ET:
                finalstatus = "E>ET"
                E = np.nan
                T = np.nan
                R = np.nan
                P = np.nan
            self.fluxesREA = {
                "ETmrea": ET * ureg.watt / ureg.meter**2,
                "Fcmrea": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Emrea": E * ureg.watt / ureg.meter**2,
                "Tmrea": T * ureg.watt / ureg.meter**2,
                "Pmrea": P * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rmrea": R * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusmrea": finalstatus,
            }
        elif (Q1sum < per_poits_each) and (Q2sum > per_poits_each):
            # Assuming that all fluxes are from the canopy
            self.fluxesREA = {
                "ETmrea": ET * ureg.watt / ureg.meter**2,
                "Fcmrea": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Emrea": 0 * ureg.watt / ureg.meter**2,
                "Tmrea": ET * ureg.watt / ureg.meter**2,
                "Pmrea": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rmrea": 0 * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusmrea": "OK",
            }
        elif (Q1sum > per_poits_each) and (Q2sum < per_poits_each):
            # Assuming that all fluxes are from the ground
            self.fluxesREA = {
                "ETmrea": ET * ureg.watt / ureg.meter**2,
                "Fcmrea": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Emrea": ET * ureg.watt / ureg.meter**2,
                "Tmrea": 0 * ureg.watt / ureg.meter**2,
                "Pmrea": 0 * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rmrea": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusmrea": "OK",
            }
        else:
            pass

    def partFVS(self, W):
        """
        Implements the Flux Variance Similarity Theory proposed by Scanlon et al., 2019.

        Direct Partitioning of GPP and Re in a Subalpine Forest Ecosystem.

        Parameters
        ----------
        W : float, optional [kg_co2/kg_h2o]
            Water use efficiency, by default 0.

        Attributes
        ----------       
        Attributes: self.fluxesFVS
            Contains all the flux components and status of the calculation.
        
            - ET - float
                Total evapotranspiration (W/m2).
            - Tfvs - float
                Plant transpiration (W/m2).
            - Efvs - float
                Soil/surface evaporation (W/m2).
            - Fc - float
                Net carbon dioxide flux (mg/m2/s).
            - Rfvs - float
                Soil/surface respiration (mg/m2/s).
            - Pfvs - float
                Plant net photosynthesis* (mg/m2/s).
            - status - str
                Status of the calculation.

        Notes
        -----
        This component represents carboxylation minus photorespiration and leaf respiration; therefore,
        it is different from gross primary productivity.
        """

        aux = self.data[
            ["co2_p", "h2o_p", "w_p"]
        ].copy()  # Create dataframe with q, c, and w only
        aux["co2_p"] = aux["co2_p"] * 10**-3  # convert c from mg/m3 to g/m3

        # Needed statistics ------------------------------------------------
        var_all = aux.var()  # Variance matrix
        cov_all = aux.cov()  # Covariance matrix
        rho = aux.corr()["co2_p"]["h2o_p"]  # Correlation coefficient between c and q
        varq = var_all["h2o_p"]  # Variance of q [g/m3]^2
        varc = var_all["co2_p"]  # Variance of c [g/m3]^2
        sigmaq = varq**0.5  # Standard deviation of q [g/m3]
        sigmac = varc**0.5  # Standard deviation of c [g/m3]
        Fq = cov_all["h2o_p"]["w_p"]  # water vapor flux    [g/m2/s]
        Fc = cov_all["co2_p"]["w_p"]  # Carbon dioxide flux [g/m2/s]

        # Check if conditions are satisfied (equations 13a-b from Scanlon 2019)
        A = (sigmac / sigmaq) / rho
        B = Fc / Fq
        C = rho * (sigmac / sigmaq)

        # Check mathematical constraints Eq (13) in Scanlon et al., 2019
        if rho < 0:
            if A <= B < C:
                pass  # constraints 13a
            else:
                self.fluxesFVS = {
                    "ETfvs": Fq
                    * (10**-3)
                    * Constants.Lv.magnitude
                    * ureg.watt
                    / ureg.meter**2,
                    "Fcfvs": Fc * 10**3 * ureg.milligram / ureg.meter**2 / ureg.second,
                    "Efvs": np.nan * ureg.watt / ureg.meter**2,
                    "Tfvs": np.nan * ureg.watt / ureg.meter**2,
                    "Pfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                    "Rfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                    "statusfvs": "13a not satisfied",
                }
                return None  # if it does not obey, stop here
        else:
            if B < C:
                pass  # constraints 13b
            else:
                self.fluxesFVS = {
                    "ETfvs": Fq
                    * (10**-3)
                    * Constants.Lv.magnitude
                    * ureg.watt
                    / ureg.meter**2,
                    "Fcfvs": Fc * 10**3 * ureg.milligram / ureg.meter**2 / ureg.second,
                    "Efvs": np.nan * ureg.watt / ureg.meter**2,
                    "Tfvs": np.nan * ureg.watt / ureg.meter**2,
                    "Pfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                    "Rfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                    "statusfvs": "13b not satisfied",
                }
                return None  # if it does not obey, stop here

        # 1 - Calcula var_cp and rho_cpcr (eq. 7 in Scanlon et al., 2019)

        # Variance cp (7a)
        num = (
            (1.0 - rho * rho)
            * (varq * varc * W**2.0)
            * (varq * Fc * Fc - 2.0 * rho * sigmaq * sigmac * Fc * Fq + varc * Fq * Fq)
        )
        den = (varc * Fq + varq * Fc * W - rho * sigmaq * sigmac * (Fc + Fq * W)) ** 2.0
        var_cp = num / den  # Variance of P
        # Correlation cp, cr (7b)
        num = (1.0 - rho * rho) * varq * varc * (Fc - Fq * W) ** 2.0
        den = (
            varq * Fc * Fc - 2.0 * rho * sigmaq * sigmac * Fq * Fc + varc * Fq * Fq
        ) * (varc - 2.0 * rho * sigmaq * sigmac * W + varq * W * W)
        rho_cpcr2 = num / den

        # 2 - Obtain flux components (eq. 6 in Scanlon et al., 2019)

        # Compute roots and test if they are real

        arg1 = 1.0 - (1.0 - W * W * varq / var_cp) / rho_cpcr2
        if arg1 < 0:
            self.fluxesFVS = {
                "ETfvs": Fq
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Fcfvs": Fc * 10**3 * ureg.milligram / ureg.meter**2 / ureg.second,
                "Efvs": np.nan * ureg.watt / ureg.meter**2,
                "Tfvs": np.nan * ureg.watt / ureg.meter**2,
                "Pfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusfvs": "arg1 < 0",
            }
            return None  # Root is not real; stop partitioning

        arg2 = 1.0 - (1.0 - varc / var_cp) / rho_cpcr2
        if arg2 < 0:
            self.fluxesFVS = {
                "ETfvs": Fq
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Fcfvs": Fc * 10**3 * ureg.milligram / ureg.meter**2 / ureg.second,
                "Efvs": np.nan * ureg.watt / ureg.meter**2,
                "Tfvs": np.nan * ureg.watt / ureg.meter**2,
                "Pfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusfvs": "arg2 < 0",
            }
            return None  # Root is not real; stop partitioning

        # Roots are real. Proceed to check sign of fluxes
        ratio_ET = -rho_cpcr2 + rho_cpcr2 * np.sqrt(arg1)
        if ratio_ET < 0.0:
            self.fluxesFVS = {
                "ETfvs": Fq
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Fcfvs": Fc * 10**3 * ureg.milligram / ureg.meter**2 / ureg.second,
                "Efvs": np.nan * ureg.watt / ureg.meter**2,
                "Tfvs": np.nan * ureg.watt / ureg.meter**2,
                "Pfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusfvs": "rET < 0",
            }
            return None  # Following imposed constraint that T, E > 0

        # Test ratio of carbon fluxes: from Fluxpart - Skaggs et al, 2018
        if rho < 0 and sigmac / sigmaq < rho * W:
            ratio_RP = -rho_cpcr2 + rho_cpcr2 * np.sqrt(arg2)
        else:
            ratio_RP = -rho_cpcr2 - rho_cpcr2 * np.sqrt(arg2)

        if ratio_RP > 0.0:
            self.fluxesFVS = {
                "ETfvs": Fq
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Fcfvs": Fc * 10**3 * ureg.milligram / ureg.meter**2 / ureg.second,
                "Efvs": np.nan * ureg.watt / ureg.meter**2,
                "Tfvs": np.nan * ureg.watt / ureg.meter**2,
                "Pfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rfvs": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "statusfvs": "rRP > 0",
            }
            return None  # Following imposed constraint that P<0 and R>0

        # Obtaining flux components -------------------------------------------------
        T = Fq / (1.0 + ratio_ET)
        E = Fq - T

        T = T * (10**-3) * Constants.Lv.magnitude  # in W/m2
        E = E * (10**-3) * Constants.Lv.magnitude  # in W/m2

        P = Fc / (1.0 + ratio_RP)  #  g/m2/s
        R = Fc - P  #  g/m2/s
        P = P * 10**3  # mg/m2/s
        R = R * 10**3  # mg/m2/s

        # Convert total fluxes
        Fq = Fq * (10**-3) * Constants.Lv.magnitude  # in W/m2
        Fc = Fc * 10**3  # mg/m2/s

        # Check CO2 flux components ratio
        #   CO2 fluxes might be noisy in this range
        ratioRP = R / P

        if -1.2 < ratioRP < -0.8:
            finalstat = "Small ratioRP"
        else:
            finalstat = "OK"

        finalstat
        # Add final values to dictionary
        self.fluxesFVS = {
            "ETfvs": Fq * ureg.watt / ureg.meter**2,
            "Fcfvs": Fc * ureg.milligram / ureg.meter**2 / ureg.second,
            "Efvs": E * ureg.watt / ureg.meter**2,
            "Tfvs": T * ureg.watt / ureg.meter**2,
            "Pfvs": P * ureg.milligram / ureg.meter**2 / ureg.second,
            "Rfvs": R * ureg.milligram / ureg.meter**2 / ureg.second,
            "statusfvs": finalstat,
        }

    def partCEA(self, H=0.00):
        """
        Implements the Conditional Eddy Accumulation method proposed by Zahn et al. 2024.

        Numerical Investigation of Observational Flux Partitioning Methods for Water Vapor and Carbon Dioxide

        Parameters
        ----------
        H : float, optional
            Hyperbolic threshold, by default 0.0.

        Attributes
        ----------       
        Attributes: self.fluxesCEA
            Contains all the flux components and status of the calculation.
        
            - ET - float
                Total evapotranspiration (W/m2).
            - Tcea - float
                Plant transpiration (W/m2).
            - Ecea - float
                Soil/surface evaporation (W/m2).
            - Fc - float
                Carbon dioxide flux (mg/m2/s).
            - Rcea - float
                Soil/surface respiration (mg/m2/s).
            - Pcea - float
                Plant net photosynthesis* (mg/m2/s).
            - statuscea - str
                Status of the calculation.

        Notes
        -----
        This component represents carboxylation minus photorespiration and leaf respiration; therefore,
        it is different from gross primary productivity.
        """
        # Creates a dataframe with variables of interest and no constraints
        unitLE = Constants.Lv.magnitude * 10**-3
        df = self.data.copy()
        auxET = df[["co2_p", "h2o_p", "w_p"]].copy()
        auxETcov = auxET.cov()  # covariance
        N = auxET["w_p"].index.size  # Total number of points
        total_Fc = auxETcov["co2_p"]["w_p"]  # flux [all quadrants] given in mg/(s m2)
        total_ET = (
            auxETcov["h2o_p"]["w_p"] * unitLE
        )  # flux [all quadrants] given in  W/m2

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE = df[
            (df["w_p"] > 0)
            & (df["co2_p"] > 0)
            & (df["h2o_p"] > 0)
            & (
                abs(df["co2_p"] / df["co2_p"].std())
                > abs(H * df["h2o_p"].std() / df["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        auxE_n = df[
            (df["w_p"] < 0)
            & (df["co2_p"] < 0)
            & (df["h2o_p"] < 0)
            & (
                abs(df["co2_p"] / df["co2_p"].std())
                > abs(H * df["h2o_p"].std() / df["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        C1 = auxE["co2_p"].mean()
        C2 = auxE_n["co2_p"].mean()
        Q1 = auxE["h2o_p"].mean()
        Q2 = auxE_n["h2o_p"].mean()
        sum1 = (
            auxE["co2_p"].index.size / N
        ) * 100  # Number of points on the first quadrant
        sum2 = (
            auxE_n["co2_p"].index.size / N
        ) * 100  # Number of points on the first quadrant

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT = df[
            (df["w_p"] > 0)
            & (df["co2_p"] < 0)
            & (df["h2o_p"] > 0)
            & (
                abs(df["co2_p"] / df["co2_p"].std())
                > abs(H * df["h2o_p"].std() / df["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        auxT_n = df[
            (df["w_p"] < 0)
            & (df["co2_p"] > 0)
            & (df["h2o_p"] < 0)
            & (
                abs(df["co2_p"] / df["co2_p"].std())
                > abs(H * df["h2o_p"].std() / df["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        C3 = auxT["co2_p"].mean()
        C4 = auxT_n["co2_p"].mean()
        Q3 = auxT["h2o_p"].mean()
        Q4 = auxT_n["h2o_p"].mean()
        sum3 = (
            auxT["co2_p"].index.size / N
        ) * 100  # Number of points on the first quadrant
        sum4 = (
            auxT_n["co2_p"].index.size / N
        ) * 100  # Number of points on the first quadrant

        # Computing flux ratios and flux components of ET and Fc
        if (sum1 > 2) and (sum2 > 2) and (sum3 > 2) and (sum4 > 2):
            ratioET = (Q1 - Q2) / (Q3 - Q4)
            T = total_ET / (1.0 + ratioET)
            E = total_ET / (1.0 + 1.0 / ratioET)

            ratioRP = (C1 - C2) / (C3 - C4)
            P = total_Fc / (1.0 + ratioRP)
            R = total_Fc / (1.0 + 1.0 / ratioRP)
            status_message = "OK"

            # Check sign of fluxes
            if P > 0.0:
                P = np.nan
                R = np.nan
                T = np.nan
                E = np.nan
                status_message = "unrealistic fluxes"

            self.fluxesCEA = {
                "ETcea": total_ET * ureg.watt / ureg.meter**2,
                "Ecea": E * ureg.watt / ureg.meter**2,
                "Tcea": T * ureg.watt / ureg.meter**2,
                "Fccea": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Pcea": P * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rcea": R * ureg.milligram / ureg.meter**2 / ureg.second,
                # "ratioETcea": ratioET,
                # "ratioRPcea": ratioRP,
                #  "sumQ1cea": sum1 + sum2,
                #  "sumQ2cea": sum3 + sum4,
                "wuecea": P * 10**-3 / (T / unitLE),
                "statuscea": status_message,
            }
        else:
            self.fluxesCEA = {
                "ETcea": total_ET * ureg.watt / ureg.meter**2,
                "Ecea": np.nan * ureg.watt / ureg.meter**2,
                "Tcea": np.nan * ureg.watt / ureg.meter**2,
                "Fccea": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Pcea": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rcea": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                # "ratioETcea": np.nan,
                # "ratioRPcea": np.nan,
                # "sumQ1cea": sum1 + sum2,
                # "sumQ2cea": sum3 + sum4,
                "wuecea": np.nan,
                "statuscea": "Not enough points",
            }

    def partCECw(self, W, H=0.00):
        """
        Implements Conditional Eddy Covariance + water use efficiency proposed by Zahn et al. 2024.

        Numerical Investigation of Observational Flux Partitioning Methods for Water Vapor and Carbon Dioxide

        Parameters
        ----------
        W : float, optional
            Water use efficiency, by default 0.

        Attributes
        ----------       
        Attributes: self.fluxesCECw
            Contains all the flux components and status of the calculation.

            Dictionary with the following flux components:
            - ET - float
                Total evapotranspiration (W/m2).
            - Tcecw - float
                Plant transpiration (W/m2).
            - Ececw - float
                Soil/surface evaporation (W/m2).
            - Fc - float
                Net carbon dioxide flux (mg/m2/s).
            - Rcecw - float
                Soil/surface respiration (mg/m2/s).
            - Pcecw - float
                Plant net photosynthesis* (mg/m2/s).
            - statuscecw - str
                Status of the calculation.

        Notes
        -----
        This component represents carboxylation minus photorespiration and leaf respiration; therefore,
        it is different from gross primary productivity.
        """
        # Creates a dataframe with variables of interest and no constraints
        df = self.data.copy()
        auxET = df[["co2_p", "h2o_p", "w_p"]].copy()
        auxETcov = auxET.cov()  # covariance
        N = auxET["w_p"].index.size  # Total number of points
        total_Fc = auxETcov["co2_p"]["w_p"]  # mg/(s m2)
        total_ET = auxETcov["h2o_p"]["w_p"]  # g/(s m2)

        if W > (total_Fc / (10**3 * total_ET)):
            self.fluxesCECw = {
                "ETcecw": total_ET
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Ececw": np.nan * ureg.watt / ureg.meter**2,
                "Tcecw": np.nan * ureg.watt / ureg.meter**2,
                "Fccecw": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Pcecw": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rcecw": np.nan * ureg.milligram / ureg.meter**2 / ureg.second,
            }
            return 0

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the first quadrant
        auxE = df[
            (df["w_p"] > 0)
            & (df["co2_p"] > 0)
            & (df["h2o_p"] > 0)
            & (
                abs(df["co2_p"] / df["co2_p"].std())
                > abs(H * df["h2o_p"].std() / df["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        R_condition_Fc = sum(auxE["co2_p"] * auxE["w_p"]) / N
        E_condition_ET = sum(auxE["h2o_p"] * auxE["w_p"]) / N

        # Creates a dataframe with variables of interest and conditioned on updrafts and on the second quadrant
        auxT = df[
            (df["w_p"] > 0)
            & (df["co2_p"] < 0)
            & (df["h2o_p"] > 0)
            & (
                abs(df["co2_p"] / df["co2_p"].std())
                > abs(H * df["h2o_p"].std() / df["h2o_p"])
            )
        ][["co2_p", "h2o_p", "w_p"]]
        P_condition_Fc = (
            sum(auxT["co2_p"] * auxT["w_p"]) / N
        )  # conditional flux [2nd quadrant and w'>0] given in mg/(s m2)
        T_condition_ET = (
            sum(auxT["h2o_p"] * auxT["w_p"]) / N
        )  # conditional flux [2nd quadrant and w'>0] flux given in  W/m2

        # Compute needed parameters (ratios first)
        if T_condition_ET > 0:
            ratioET = E_condition_ET / T_condition_ET
            ratioRP = R_condition_Fc / P_condition_Fc
        else:
            self.fluxesCECw = {
                "ETcecw": total_ET
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Ececw": total_ET
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Tcecw": 0 * ureg.watt / ureg.meter**2,
                "Fccecw": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Pcecw": 0 * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rcecw": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
            }
            return 0

        # Compute Z -------------------------------
        if ratioET > 0:
            Z = W * (ratioRP / ratioET)
        else:
            self.fluxesCECw = {
                "ETcecw": total_ET
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Ececw": 0 * ureg.watt / ureg.meter**2,
                "Tcecw": total_ET
                * (10**-3)
                * Constants.Lv.magnitude
                * ureg.watt
                / ureg.meter**2,
                "Fccecw": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Pcecw": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
                "Rcecw": 0 * ureg.milligram / ureg.meter**2 / ureg.second,
            }
            return 0

        # Compute flux components -----------------
        R = (total_Fc - W * total_ET * 10**3) / (1 - W / Z)  # in (mg/kg)/m2/s
        P = total_Fc - R  # in (mg/kg)/m2/s
        E = (R / Z) * (10**-6) * Constants.Lv.magnitude  # in W/m2
        T = total_ET * (10**-3) * Constants.Lv.magnitude - E  # in W/m2
        del ratioET, ratioRP

        self.fluxesCECw = {
            "ETcecw": total_ET
            * (10**-3)
            * Constants.Lv.magnitude
            * ureg.watt
            / ureg.meter**2,
            "Ececw": E * ureg.watt / ureg.meter**2,
            "Tcecw": T * ureg.watt / ureg.meter**2,
            "Fccecw": total_Fc * ureg.milligram / ureg.meter**2 / ureg.second,
            "Pcecw": P * ureg.milligram / ureg.meter**2 / ureg.second,
            "Rcecw": R * ureg.milligram / ureg.meter**2 / ureg.second,
        }
