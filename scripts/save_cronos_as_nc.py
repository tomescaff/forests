"""This script writes the data from Cronos DB into a NetCDF file."""

import glob
import pandas as pd
import xarray as xr

def cronos_csv_to_netcdf():
    """Reads data from the Cronos database and writes it to a NetCDF file."""
    
    fp = '/home/tcarrasco/result/data/forests/CRONO_MODNEGEXP'
    fps = glob.glob(fp + '/*.csv')
    
    cat_list = []
    for i, fp in enumerate(fps):
        name = fp.split('/')[-1].split('_')[0]
        df = pd.read_csv(fp, header=0, index_col=0, parse_dates=[0])
        da = df.iloc[:, 1].to_xarray()
        da = da.rename({'index': 'time'})
        cat_list += [da.expand_dims(dim={'site': [name]}, axis=0)]
        
    cronos = xr.concat(cat_list, dim='site')
    ds = cronos.to_dataset(name='db')
    ds.to_netcdf('/home/tcarrasco/result/data/forests/cronos_db.nc')

if __name__ == '__main__':

    cronos_csv_to_netcdf()