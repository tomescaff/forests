import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.append('/home/tcarrasco/result/repo/forests/')

from utils import aggregation as agg
from utils import cross_validation as crossval

fp = '/home/tcarrasco/result/data/forests/cronos_db.nc'
cronos = xr.open_dataset(fp)['db']

cam2 = cronos.sel(site='CAM2').sel(time=slice('1952-01-01', '2021-12-31'))
era5_dict = agg.create_aggregated_era5_timeseries()

predictant = cam2.values
# predictant = np.log(1 + cam2.values)
predictors_dict = {k: v.values for k, v in era5_dict.items()}

ans = crossval.forward_selection(predictant, predictors_dict)
mse_crossval = crossval.cross_validation(predictant, predictors_dict)

fig, axs = plt.subplots(2, 1, figsize=(8,10), sharex=True)

r2 = ans[1]
mse = ans[2]

K = r2.size

plt.sca(axs[0])
plt.plot(np.arange(1, K+1), r2, 'ro-')
plt.xlabel('Number of predictors')
plt.ylabel('R2')
plt.grid(ls='--', c='grey', lw=0.5)

plt.sca(axs[1])
plt.plot(np.arange(1, K+1), mse, 'ro-')
plt.plot(np.arange(1, mse_crossval.size+1), mse_crossval, 'bo-')
plt.xlabel('Number of predictors')
plt.ylabel('MSE')
plt.grid(ls='--', c='grey', lw=0.5)

plt.tight_layout()
bd = '/home/tcarrasco/result/images/forests/'
fn = f'forward_selection_and_crossval_log.png'
plt.savefig(bd + fn, format='png', dpi=300)
