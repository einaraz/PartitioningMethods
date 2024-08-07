
[![DOI](https://zenodo.org/badge/441544177.svg)](https://zenodo.org/badge/latestdoi/441544177)

---
# Contact information 

Author: Einara Zahn\
email: einaraz@princeton.edu, einara.zahn@gmail.com

## Flux Partitioning Package

This Python package implements five partitioning methods to separate evapotranspiration (ET) and CO<sub>2</sub> fluxes (Fc) into ground (evaporation and respiration) and plant (transpiration and net CO<sub>2</sub> assimilation) fluxes. It processes instantaneous raw eddy covariance measurements and returns fluxes, flux components, and additional turbulent variables.

The package includes pre-processing steps such as coordinate rotation and density corrections. While it does not perform all post-processing corrections provided by other eddy covariance flux software, users are encouraged to reach out with inquiries about adding additional pre- and post-processing techniques or to contribute to the code.

See the [Documentation](https://einaraz.github.io/PartitioningMethods/) here.

# Installation Guide

To install the package, follow these steps:

1. **Clone or download the repository** to your computer.

    ```sh
    git clone git@github.com:einaraz/PartitioningMethods.git
    ```

2. **Navigate to the main folder** of the repository:

    ```sh
    cd PartitioningMethods/
    ```

3. **Create and activate a virtual environment** (not required, but recommended to avoid conflicts with local python packages):

    > **Note**: A virtual environment is an isolated environment in which you can install packages without affecting your system-wide Python installation. This helps prevent conflicts between package versions and keeps your project dependencies organized.

    - **Create the virtual environment:**

      This command creates a directory named `venv` that contains the virtual environment. You only need to do this once per project.

      ```sh
      python -m venv venv
      ```

    - **Activate the virtual environment:**

      Activation adjusts your environment to use the packages installed in the virtual environment instead of the global Python installation. You need to activate the environment every time you start a new terminal session or work on your project.

      - **On macOS/Linux:**

        ```sh
        source venv/bin/activate
        ```

      - **On Windows:**

        ```sh
        venv\Scripts\activate
        ```

      After activation, your command prompt should change to show the name of the virtual environment, typically `(venv)`.

      If you see `(venv)` at the beginning of your command prompt, the virtual environment is active. If you don't see it, try running the activation command again. 

4. **Install the package** using pip:

    With the virtual environment activated, install your package:

    ```sh
    pip install .
    ```

    This command installs the package partitioning and all other python dependencies in the virtual environment.
   
6. **Verify the installation** by importing the package in a Python interpreter:

    To check if the package is installed correctly, open a Python interpreter and try importing it:

    ```sh
    python -c "import partitioning"
    ```

    If no errors occur, the package is installed correctly.

### Additional Information

- **Deactivating the Virtual Environment:**

  When you’re done working in the virtual environment, you can deactivate it by running:

  ```sh
  deactivate
  ```
  **Important:** After installing the package `partitioning` inside the virtual environment, you must activate the `venv` environment each time you work on the project. This ensures that you are using the correct package versions and dependencies specified for your project.


## Format of Input Text Files

This script works with text files separated by commas (CSV format). It reads eddy-covariance time series of any length, although 30-minute intervals are typically used under neutral to unstable conditions.

The following variables are required by the code when using raw high-frequency data as input (see the code for requirements when using pre-processed data as input):

- **index**: Date and time of acquisition in the format `[yyyy-mm-dd HH:MM]`.
- **w**: Vertical velocity component (m/s).
- **u**: Streamwise velocity component (m/s).
- **v**: Cross-stream velocity component (m/s).
- **Ts**: Sonic temperature (Celsius).
- **co2**: Carbon dioxide density (mg_CO2/m³).
- **h2o**: Water vapor density (g_H2O/m³).
- **P**: Pressure (kPa).
  
## How to use it
For a complete example of how to use the partitioning module, see ```main.py``` and ```main_parellel.py```
See the [Documentation](https://einaraz.github.io/PartitioningMethods/) for more details.

```sh
from partitioning import Partitioning

processing_args = {
    "density_correction": True,  # If True, density corrections are implemented during pre-processing (depends on type of gas analyzer used)
    "fluctuations": "LD",        # If "LD", linear detrending is applied to the data. BA (block averaging) and FL (filter low freqencies) are also available
    "maxGapsInterpolate": 5,     # Intervals of up to 5 missing values are filled by linear interpolation
    "RemainingData": 95,         # Only proceed with partioning if 95% of initial data is available after pre-processing
    }

# Create a partitioning object - it takes raw data and applies all necessary corrections
part = Partitioning(
            hi=2.5,    # mean canopy height [m]
            zi=4.0,    # eddy covariance measurement height [m]
            freq=20,   # sampling frequency [Hz]
            length=30, # length of time series [minutes]
            df=df,     # dataframe containing all variables descrived above
            PreProcessing=True,     # if True, performs pre-processing before applying partitioning
            argsQC=processing_args) # additional arguments

# To implement CEC
part.partCEC()
print(part.fluxesCEC)
```

The class may return errors if the data is invalid, contains excessive missing periods, or is of poor quality. If you are running the code in a loop that processes multiple files, it is recommended to implement a try/except block to handle such cases effectively and prevent interruptions in the code.

```sh
try:
    part = Partitioning(
            hi=2.5,    # mean canopy height [m]
            zi=4.0,    # eddy covariance measurement height [m]
            freq=20,   # sampling frequency [Hz]
            length=30, # length of time series [minutes]
            df=df,     # dataframe containing all variables descrived above
            PreProcessing=True,     # if True, performs pre-processing before applying partitioning
            argsQC=processing_args) # additional arguments
    part.partCEC()
    # other partitioning methods
    # save data to object
except ValueError as ve:
    print(ve)
except TypeError as te:
    print(te)
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
   > **Note**: Additional data cleaning, such as screening sensor flags, is recommended before using the partitioning class. While the package includes several quality control and assurance features, users are encouraged to perform their own tests as well. Refer to [Vickers and Mahrt, 1997](https://journals.ametsoc.org/view/journals/atot/14/3/1520-0426_1997_014_0512_qcafsp_2_0_co_2.xml) and [Zahn et al., 2016](http://article.sapub.org/10.5923.s.ajee.201601.20.html) for examples of additional tests. If you have suggestions on additional pre- and/or post-processing methods, please feel free to reach out or consider contributing to the code!

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


## References for papers and datasets

- Zahn, E., Bou-Zeid, E., Good, S. P., Katul, G. G., Thomas, C. K., Ghannam, K., Smith, J. A., Chamecki, M., Dias, N. L., Fuentes, J. D., Alfieri, J. G., Kwon, H., Caylor, K. K., Gao, Z., Soderberg, K., Bambach, N. E., Hipps, L. E., Prueger, J. H., & Kustas, W. P. (2022). Direct partitioning of eddy-covariance water and carbon dioxide fluxes into ground and plant components. *Agricultural and Forest Meteorology, 315*, 108790. [https://doi.org/10.1016/j.agrformet.2021.108790](https://doi.org/10.1016/j.agrformet.2021.108790)

- Zahn, E., Ghannam, K., Chamecki, M., Moene, A. F., Kustas, W. P., Good, S. P., & Bou-Zeid, E. (2024). Numerical investigation of observational flux partitioning methods for water vapor and carbon dioxide. *Journal of Geophysical Research: Biogeosciences, 129*, e2024JG008025. [https://doi.org/10.1029/2024JG008025](https://doi.org/10.1029/2024JG008025)

- Zahn, E., & Bou-Zeid, E. (2024). Partitioning of water and CO2 fluxes at NEON sites into soil and plant components: A five-year dataset for spatial and temporal analysis. *Earth System Science Data Discussions* [preprint]. [https://doi.org/10.5194/essd-2024-272](https://doi.org/10.5194/essd-2024-272)

- Zahn, E., & Bou-Zeid, E. (2024). Partitioning of water and CO2 fluxes at NEON sites into soil and plant components: a five-year dataset for spatial and temporal analysis (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.12191876
