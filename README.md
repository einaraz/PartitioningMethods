
[![DOI](https://zenodo.org/badge/441544177.svg)](https://zenodo.org/badge/latestdoi/441544177)

---
# Contact information 

Author: Einara Zahn\
email: einaraz@princeton.edu, einara.zahn@gmail.com

# Installation Guide

To install the package, follow these steps:

1. **Clone or download the repository** to your computer.

2. **Navigate to the main folder** of the repository:

    ```sh
    cd PartitioningMethods/
    ```

3. **Install the package** using pip:

    ```sh
    pip install .
    ```

    Note: To avoid conflicts with local Python libraries, it is recommended to install the package inside a virtual environment.

4. **Verify the installation** by importing the package in a Python interpreter:

    ```sh
    python -c "import partitioning"
    ```

    If no errors occur, the package is installed correctly.

## How to use it
For a complete example of how to use the partitioning module, see ```main.py``` and ```main_parellel.py```

```sh
from partitioning import Partitioning

# Crate a partitioning object - it takes raw data and applies all necessary corrections
part = Partitioning(
            hi=2.5,
            zi=4.0,
            freq=20,
            length=30,
            df=df,
            PreProcessing=True,
            argsQC=processing_args)

# To implement CEC
part.partCEC()
print(part.fluxesCEC)
```

---
## Description

The package provides tools for processing and analyzing high-frequency eddy-covariance data. Key features include:

1. **Quality Control and Pre-processing**:
   - Checks for physically realistic values
   - Detects outliers
   - Implements time lag corrections for closed-path gas analyzers
   - Rotates coordinates
   - Extracts fluctuations (block average, linear detrending, and filtering operations available)
   - Applies density corrections for CO<sub>2</sub> and H<sub>2</sub>O measured by open-path gas analyzers ("instantaneous" WPL correction, based on the paper [Detto and Katul, 2007](https://link.springer.com/article/10.1007%2Fs10546-006-9105-1))
   - Performs stationarity tests
   - Fills gaps

2. **Water-Use Efficiency Parameterizations**:
   Implements five parameterizations as described in [Zahn et al., 2021](https://www.sciencedirect.com/science/article/pii/S0168192321004767?via%3Dihub) "Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components".

3. **Partitioning Methods**:
   Implements five methods to partition eddy-covariance data and output T and E (W/m<sup>2</sup>) and P and R (mg_CO<sub>2</sub>/m<sup>2</sup>/s):
   
   1. **Conditional Eddy Covariance (CEC)**:
      [Zahn et al., 2021](https://www.sciencedirect.com/science/article/pii/S0168192321004767?via%3Dihub) "Direct Partitioning of Eddy-Covariance Water and Carbon Dioxide Fluxes into Ground and Plant Components".
   
   2. **Modified Relaxed Eddy Accumulation (MREA)**:
      [Thomas et al., 2008](https://www.sciencedirect.com/science/article/pii/S0168192308000737) "Estimating daytime subcanopy respiration from conditional sampling methods applied to multi-scalar high frequency turbulence time series".
   
   3. **Flux-Variance Similarity (FVS)**:
      [Scanlon and Sahu, 2008](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008WR006932) "Correlation-based flux partitioning of water vapor and carbon dioxide fluxes: Method simplification and estimation of canopy water use efficiency" and
      [Scanlon et al., 2019](https://www.sciencedirect.com/science/article/pii/S016819231930348X?via%3Dihub) "Correlation-based flux partitioning of water vapor and carbon dioxide fluxes: Method simplification and estimation of canopy water use efficiency".
   
   4. **Conditional Eddy Accumulation (CEA)**:
      [Zahn et al., 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JG008025) "Numerical Investigation of Observational Flux Partitioning Methods for Water Vapor and Carbon Dioxide".
   
   5. **Conditional Eddy Covariance with Water-Use Efficiency (CECw)**:
      [Zahn et al., 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JG008025) "Numerical Investigation of Observational Flux Partitioning Methods for Water Vapor and Carbon Dioxide".

## Available Files

The following files are available in the repository:

- `src/partitioning/Partitioning.py`: Contains the `Partitioning` class with methods for quality control, data pre-processing, and five partitioning methods to separate ET and CO₂ fluxes into stomatal and non-stomatal components.
- `src/partitioning/auxfunctions.py`: Includes auxiliary functions for pre-processing and computing water-use efficiency, some of them adapted from [FluxPart](https://github.com/usda-ars-ussl/fluxpart).
- `main.py`: Provides an example of how to use the script, with raw high-frequency data examples included in the `RawData30min` folder.
- `main_parallel.py`: Demonstrates how to run the script in parallel to process multiple files simultaneously.


---
## Format of Input Text Files

This script works with text files separated by commas (CSV format).

The following variables are required by the code when using raw high-frequency data as input (see the code for requirements when using pre-processed data as input):

- **index**: Date and time of acquisition in the format `[yyyy-mm-dd HH:MM]`.
- **w**: Vertical velocity component (m/s).
- **u**: Streamwise velocity component (m/s).
- **v**: Cross-stream velocity component (m/s).
- **Ts**: Sonic temperature (Celsius).
- **co2**: Carbon dioxide density (mg_CO2/m³).
- **h2o**: Water vapor density (g_H2O/m³).
- **P**: Pressure (kPa).


## References for papers and datasets

- Zahn, E., Bou-Zeid, E., Good, S. P., Katul, G. G., Thomas, C. K., Ghannam, K., Smith, J. A., Chamecki, M., Dias, N. L., Fuentes, J. D., Alfieri, J. G., Kwon, H., Caylor, K. K., Gao, Z., Soderberg, K., Bambach, N. E., Hipps, L. E., Prueger, J. H., & Kustas, W. P. (2022). Direct partitioning of eddy-covariance water and carbon dioxide fluxes into ground and plant components. *Agricultural and Forest Meteorology, 315*, 108790. [https://doi.org/10.1016/j.agrformet.2021.108790](https://doi.org/10.1016/j.agrformet.2021.108790)

- Zahn, E., Ghannam, K., Chamecki, M., Moene, A. F., Kustas, W. P., Good, S. P., & Bou-Zeid, E. (2024). Numerical investigation of observational flux partitioning methods for water vapor and carbon dioxide. *Journal of Geophysical Research: Biogeosciences, 129*, e2024JG008025. [https://doi.org/10.1029/2024JG008025](https://doi.org/10.1029/2024JG008025)

- Zahn, E., & Bou-Zeid, E. (2024). Partitioning of water and CO2 fluxes at NEON sites into soil and plant components: A five-year dataset for spatial and temporal analysis. *Earth System Science Data Discussions* [preprint]. [https://doi.org/10.5194/essd-2024-272](https://doi.org/10.5194/essd-2024-272)

- Zahn, E., & Bou-Zeid, E. (2024). Partitioning of water and CO2 fluxes at NEON sites into soil and plant components: a five-year dataset for spatial and temporal analysis (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12191876
