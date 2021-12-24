import numpy as np

# Auxiliary functions to compute the water-use efficiency
# These functions are from the algorithm fluxpart: 
#                 Skaggs et al. 2018, Agr For Met
#                "Fluxpart: Open source software for partitioning carbon dioxide and watervaporfluxes" 
# https://github.com/usda-ars-ussl/fluxpart

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
