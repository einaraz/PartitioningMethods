import numpy as np
from scipy.signal import correlate

def find_time_lag(velocity, gas_concentration):
    """
    Calculate the time lag between the velocity and gas concentration signals
    using cross-correlation.
    
    Parameters:
    velocity (array-like): Time series of vertical wind velocity.
    gas_concentration (array-like): Time series of gas concentration (e.g., CO2).
    
    Returns:
    int: Lag index, where positive values indicate gas_concentration lags behind velocity.
    """
    # Subtract the mean to remove the DC component
    velocity = velocity - np.mean(velocity)
    gas_concentration = gas_concentration - np.mean(gas_concentration)
    
    # Calculate the cross-correlation
    correlation = correlate(velocity, gas_concentration, mode='full')
    print(correlation)
    input("...")
    # The lag corresponds to the index of the maximum correlation
    lag_index = np.argmax(correlation) - (len(gas_concentration) - 1)
        
    return lag_index

def correct_time_lag(velocity, gas_concentration, lag_index):
    """
    Correct the gas concentration data by shifting it to align with the velocity data.
    
    Parameters:
    velocity (array-like): Time series of vertical wind velocity.
    gas_concentration (array-like): Time series of gas concentration (e.g., CO2).
    lag_index (int): The number of time steps by which to shift gas_concentration.
    
    Returns:
    tuple: Corrected velocity and gas concentration time series.
    """
    if lag_index > 0:
        # Shift gas concentration forward (pad with NaNs at the end)
        corrected_gas = np.roll(gas_concentration, -lag_index)
        corrected_gas[-lag_index:] = np.nan  # Fill the end with NaNs
        
    elif lag_index < 0:
        # Shift gas concentration backward (pad with NaNs at the start)
        corrected_gas = np.roll(gas_concentration, -lag_index)
        corrected_gas[:abs(lag_index)] = np.nan  # Fill the start with NaNs
    else:
        # No shift needed
        corrected_gas = gas_concentration
    
    return velocity, corrected_gas

# Example usage
velocity = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
gas_concentration = np.array([5., 6., 7., 8., 9., 10., 1., 2., 3., 4.])

# Find the time lag
lag_index = find_time_lag(velocity, gas_concentration)
print("Time Lag Index:", lag_index)

# Correct the time lag
corrected_velocity, corrected_gas_concentration = correct_time_lag(velocity, gas_concentration, lag_index)

print("Corrected Gas Concentration:", corrected_gas_concentration)
