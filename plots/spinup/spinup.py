import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob

# 1. Load all monthly history files in chronological order
# We use chunking to prevent memory overload if the files are massive
#file_list = sorted(glob.glob('/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'))
file_list = sorted(glob.glob('/scratch/ags/cesm_archive/asc0/atm/hist/asc0.cam.h0.*.nc'))
print(f"Loading {len(file_list)} monthly files...")
ds = xr.open_mfdataset(file_list, combine='by_coords', decode_times=False)

# 2. Extract Surface Temperature (TS)
# In CESM, TS is the radiative surface temperature in Kelvin
ts = ds['TS']

# 3. Calculate Area-Weighted Global Mean
# Create a weight array based on the cosine of the latitude
weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"

# Apply the weights and average over latitude and longitude
ts_global_mean = ts.weighted(weights).mean(dim=("lat", "lon"))

# 4. Create the time axis (in Years)
# Since we have monthly files, we divide the index by 12
months_total = len(ts_global_mean)
years = np.arange(months_total) / 12.0

# 5. Generate the Matplotlib Plot
plt.style.use('seaborn-v0_8-darkgrid') # Clean scientific aesthetic
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Plot the data
ax.plot(years, ts_global_mean.values, color='#1f77b4', linewidth=1.5, marker='o', markersize=3)

# Formatting
ax.set_title('Global Mean Surface Temperature, Spin-Up', fontsize=14, pad=15)
ax.set_xlabel('Simulation Year', fontsize=12)
ax.set_ylabel('Global Mean Temperature (K)', fontsize=12)

# Set x-ticks to whole years
ax.set_xticks(np.arange(0, 13, 1))

# Add a subtle line for the final year average to visualize equilibrium
final_year_avg = ts_global_mean[-12:].mean().values
ax.axhline(final_year_avg, color='red', linestyle='--', alpha=0.6, 
           label=f'Final Year Avg: {final_year_avg:.2f} K')
ax.legend()

# Save the plot as a high-res PNG before displaying
plt.savefig('spinup_sfc_temp.png', bbox_inches='tight')
print("Plot saved as 'spinup_sfc_temp.png'")
