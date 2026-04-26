import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob

# 1. Load all monthly CLM history files
file_list = sorted(glob.glob('/scratch/ags/cesm_archive/asc0/lnd/hist/asc0.clm2.h0.*.nc'))
print(f"Loading {len(file_list)} CLM monthly files...")
ds = xr.open_mfdataset(file_list, combine='by_coords', decode_times=False)

# 2. Extract Soil Temperature (TSOI) for the deepest layer
# Using .isel() with -1 physically grabs the absolute bottom grid cell
if 'levgrnd' in ds.dims:
    tsoi_deep = ds['TSOI'].isel(levgrnd=-1)
    layer_name = 'levgrnd'
elif 'levsoi' in ds.dims:
    tsoi_deep = ds['TSOI'].isel(levsoi=-1)
    layer_name = 'levsoi'
else:
    print("Warning: Could not find standard soil depth dimension. Attempting fallback.")
    tsoi_deep = ds['TSOI'].isel(levgrnd=-1)
    layer_name = 'levgrnd'

print(f"Extracted deepest soil layer using dimension: {layer_name}")

# 3. Calculate Area-Weighted Global Mean
# Even though this is land data, the grid is still defined by lat/lon
weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"

tsoi_global_mean = tsoi_deep.weighted(weights).mean(dim=("lat", "lon"))

# 4. Create the time axis (in Years)
months_total = len(tsoi_global_mean)
years = np.arange(months_total) / 12.0

# 5. Generate the Plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Plot the raw monthly data
# Using a brown hex color to represent the soil component
ax.plot(years, tsoi_global_mean.values, color='#8c564b', linewidth=1.5, alpha=0.7, label='Monthly Mean')

# Calculate and plot the 12-month rolling average to show the true drift
rolling_mean = tsoi_global_mean.rolling(time=12, center=True).mean()
ax.plot(years, rolling_mean.values, color='black', linewidth=2.5, label='12-Month Rolling Average')

# Formatting
ax.set_title('Soil Temperature, Spin-Up', fontsize=14, pad=15)
ax.set_xlabel('Simulation Year', fontsize=12)
ax.set_ylabel('Deep Soil Temperature (K)', fontsize=12)
ax.set_xticks(np.arange(0, 13, 1))

# Add a subtle line for the final year average to visualize equilibrium
final_year_avg = tsoi_global_mean[-12:].mean().values
ax.axhline(final_year_avg, color='red', linestyle='--', alpha=0.6, 
           label=f'Final Year Avg: {final_year_avg:.2f} K')

ax.legend()
plt.savefig('spinup_deep_soil.png', bbox_inches='tight')
print("Plot saved as 'spinup_deep_soil.png'")

plt.show()
