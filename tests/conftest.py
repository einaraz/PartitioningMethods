import pytest
import pandas as pd
import os

@pytest.fixture
def sample_data_read():
    filei = os.path.join(os.path.dirname(__file__), '../RawData30min/2018-07-05-1000.csv')
    ecdata = pd.read_csv(
        filei,
        header=None,
        index_col=0,
        usecols=[0, 1, 2, 3, 4, 6, 8, 11, 12],
        names=["date", "u", "v", "w", "Ts", "co2", "h2o", "Tair", "P"],
        na_values=["NAN", -9999, "-9999", "#NA", "NULL"],
        skiprows=[0],
    )
    print(filei)
    ecdata.index = pd.to_datetime(ecdata.index)
    return ecdata
