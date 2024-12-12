"""Module for aggregating time series from the ERA5 dataset."""

import glob
import pandas as pd
import xarray as xr

def ninstances_per_nmonths(nmonths: int) -> int:
    """Returns the number of instances per number of months in the aggregated
    time series.
    
    Params
    ------
    nmonths: int
        Number of months to aggregate the time series
    
    Returns
    -------
    int
        Number of instances per number of months in the aggregated time series
    """
    return 12*3 - (nmonths-1)

def aggregate_era5_timeseries(timeseries: xr.DataArray,
                              nmonths: int, 
                              offset: int = 0, 
                              length: int = 70,
                              agg: str = 'mean') -> xr.DataArray:
    """Aggregates the time series by a given number of months and a given
    offset.
    
    Params
    ------
    timeseries: xr.DataArray
        Time series to be aggregated
    nmonths: int
        Number of months to aggregate the time series (from 1 to 15)
    offset: int
        Offset to aggregate the time series (from 0 to 
        ninstances_per_nmonths(nmonths)-1)
    length: int
        Length of the output time series
    agg: str
        Aggregation method (mean or sum)
    
    Returns
    -------
    xr.DataArray
        Aggregated time series    
    """
    da = timeseries.rolling(time=nmonths, center=False)
    da = da.mean() if agg == 'mean' else da.sum()
    da = da.dropna('time')
    da = da.isel(time=slice(offset, None, 12))
    da = da.isel(time=slice(0, length))
    time_out = pd.date_range('1952-01-01', periods=length, freq='YS')
    da_out = xr.DataArray(da.values, coords=[time_out], dims=['time'])
    return da_out
    
def create_aggregated_era5_timeseries() -> dict:
    """Creates the aggregated ERA5 time series.
    
    Returns
    -------
    dict
        Dictionary with the aggregated ERA5 time series
    """
    fp = '/home/tcarrasco/result/data/forests/'
    fp += 'ERA5_allvars_fldmean_CL32_36S_masked_025deg_1950_2022.nc'
    
    ds = xr.open_dataset(fp)
    ds = ds.sel(time=slice('1950-01-01', '2022-12-31'))
    data_vars = ds.data_vars
    
    outdict = {}
    for data_var in data_vars:
        if str(data_var) == 'pr':
            agg = 'sum'
        else:
            agg = 'mean'
        for nmonths in range(1, 15):
            for offset in range(ninstances_per_nmonths(nmonths)):
                da = ds[data_var]
                da_out = aggregate_era5_timeseries(da, nmonths, offset, 
                                                   agg=agg)
                key = f'{data_var}_{nmonths}_{offset}'
                outdict[key] = da_out
    return outdict
        
    

        
    
    