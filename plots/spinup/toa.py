import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob

# 1. Load all monthly history files
file_list = sorted(glob.glob('/scratch/ags/cesm_archive/asc0/atm/hist/asc0.cam.h0.*.nc'))
print(f"Loading {len(file_list)} monthly files...")
ds = xr.open_mfdataset(file_list, combine='by_coords', decode_times=False)

# 2. Calculate TOA Energy Imbalance (Incoming Solar - Outgoing Thermal)
restom = ds['FSNT'] - ds['FLNT']

# 3. Calculate Area-Weighted Global Mean
weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"

# Apply weights and average over latitude and longitude
restom_global_mean = restom.weighted(weights).mean(dim=("lat", "lon"))

# 4. Create the time axis (in Years)
months_total = len(restom_global_mean)
years = np.arange(months_total) / 12.0

# 5. Generate the Plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Plot the raw monthly data (will show a zig-zag seasonal cycle)
ax.plot(years, restom_global_mean.values, color='#d62728', linewidth=1.5, alpha=0.7, label='Monthly Mean')

# Calculate and plot the 12-month rolling average to show the true drift
# We use xarray's rolling function to smooth out the seasonal noise
rolling_mean = restom_global_mean.rolling(time=12, center=True).mean()
ax.plot(years, rolling_mean.values, color='black', linewidth=2.5, label='12-Month Rolling Average')

# Formatting
ax.set_title('Global TOA Energy Imbalance, Spin-Up', fontsize=14, pad=15)
ax.set_xlabel('Simulation Year', fontsize=12)
ax.set_ylabel('Net TOA Flux (W/m²)', fontsize=12)
ax.set_xticks(np.arange(0, 13, 1))

# Add a subtle line at Y=0 for reference
ax.axhline(0, color='grey', linestyle='--', alpha=0.8)

ax.legend()
plt.savefig('spinup_toa_e.png', bbox_inches='tight')
print("Plot saved as 'spinup_toa_e.png'")

plt.show()
