import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_jjas_temp(file_pattern):
    """Loads CESM files, extracts monsoon Surface Temperature (TS), and converts to Celsius."""
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', decode_times=False)
    # select first 36 months
    ds = ds.isel(time=slice(None, 36))
    # Convert Kelvin to Celsius
    ts_celsius = ds['TS'] - 273.15
    # Average over time
    return ts_celsius.mean(dim='time')

# 1. Define file paths
control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'    
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'   

print("Processing CESM Temperature Data...")
temp_ctrl = get_jjas_temp(control_files)
temp_50   = get_jjas_temp(case50_files)
temp_100  = get_jjas_temp(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
# Note: A difference in Celsius is the same as a difference in Kelvin
anom_50  = temp_50 - temp_ctrl
anom_100 = temp_100 - temp_ctrl

# 3. Set up the Map
lon_min, lon_max = 50.0, 100.0
lat_min, lat_max = 0.0, 35.0

fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)

# 4. Plot Panel 1: Control Climatology (Absolute values in Celsius)
ax1 = axes[0]
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
ax1.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')

plot_ctrl = temp_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=20, vmax=40, cmap='inferno', # Warm colormap for absolute temps
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Mean Surface Temperature (°C)', fontsize=12)

# 5. Plot Panels 2 & 3: Warming Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction Anomaly', '100% Reduction Anomaly']

for i, ax in enumerate(axes[1:]):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
    
    plot_anom = anom_data[i].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-3, vmax=3, cmap='RdBu_r', # Reversed: Red is warming, Blue is cooling
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# Shared colorbar for anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('Temperature Anomaly (°C)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('Surface Temperature Response to Aerosol Reductions', fontsize=18, y=1.02)
plt.savefig('india_temp_3cases.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'india_temp_3cases.png'")
plt.show()
