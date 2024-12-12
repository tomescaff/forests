import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# read the data
fp = '/home/tcarrasco/result/data/forests/cronos_db.nc'
db = xr.open_dataset(fp)['db']
db = db.sel(time=slice('1950-01-01', '2022-12-31'))
cam2 = db.sel(site='CAM2')

cam2_log = np.log(1 + cam2)

# plot the data
fig, axs = plt.subplots(1, 2, figsize=(10, 10))

plt.sca(axs[0])
plt.plot(cam2.time.dt.year, cam2, 'k-')
plt.ylabel('CRONOS')
plt.xlabel('Year')
plt.grid(ls='--', c='grey', lw=0.5)

plt.sca(axs[1])
plt.plot(cam2.time.dt.year, cam2_log, 'k-')
plt.ylabel('log(1 + CRONOS)')
plt.xlabel('Year')
plt.grid(ls='--', c='grey', lw=0.5)

plt.tight_layout()
bd = '/home/tcarrasco/result/images/forests/'
fn = f'vis_log_timeseries.png'
plt.savefig(bd + fn, format='png', dpi=300)
print('CAM2')
print('Min:', cam2.min().values, 'Max:', cam2.max().values)
print('CAM2_log')
print('Min:', cam2_log.min().values, 'Max:', cam2_log.max().values)
