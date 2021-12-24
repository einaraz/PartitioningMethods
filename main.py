import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Partitioning import Partitioning
from glob import glob

"""
  Example of how to implement the class Partitioning to obtain all flux
  components from CEC, MREA, and FVS.
"""


# List of names of all files containing half-hour data (or any length of interest)
listfiles = ['2018-04-09-1700.csv', '2018-04-09-1730.csv', '2018-04-09-1800.csv' ]

for n,filei in enumerate(listfiles):
    print("%d/%d"%(n,len(listfiles)), filei)

    """
    Reading and storing data
    """
    
    # Read csv file and store data in a dataframe
    #     Each column of the dataframe has one variable
    df = pd.read_csv(filei, index_col = 0)
    df.index = pd.to_datetime(df.index)

    # Create an object called Partitioning
    part = Partitioning(hi=0.05, zi=0.40, df=df)

    """
    Applying partitioning methods 
    """
    
    # CEC method
    part.partCEC(H=0)
    print(part.fluxesCEC)
    
    # MREA method
    part.partREA(H=0.0)
    print(part.fluxesREA)
    
    # Compute water-use efficiency
    # All models are implemented: 'const_ppm', "const_ratio", "linear", "sqrt", "opt"
    part.WaterUseEfficiency(ppath='C4')
    
    # FVS method - must enter one water-use efficiency in kg_co2/kg_h2o
    part.partFVS(W=part.wue['const_ppm'])
    print(part.fluxesFVS)
    
