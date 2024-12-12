"""Plotting the CRONOS data as pcolor plot"""

import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# read the data
fp = '/home/tcarrasco/result/data/forests/cronos_db.nc'
db = xr.open_dataset(fp)['db']

db = db.sel(time=slice('1950-01-01', '2022-12-31'))

sites = db.site.values
time = db.time

# plot the data
fig = plt.figure(figsize=(10, 10))
pcm = plt.pcolormesh(time.dt.year, np.arange(sites.size), db, shading='auto')
plt.yticks(np.arange(sites.size), sites)
plt.ylabel('Site')
plt.xlabel('Year')
plt.colorbar(pcm, label='Crono', shrink=0.8)
print('Min:', db.min().values, 'Max:', db.max().values)
print('N <= 0:', db.where(db <= 0).count().values)

bd = '/home/tcarrasco/result/images/forests/'
fn = f'vis_cronos_as_pcolor.png'
plt.savefig(bd + fn, format='png', dpi=300)
