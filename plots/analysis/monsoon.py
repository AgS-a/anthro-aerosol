import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_jjas_precip(file_pattern):
    """Loads CESM files, extracts JJAS monsoon precip for the first 3 years, and converts to mm/day."""
    # 1. Grab all files matching the pattern and sort them chronologically
    all_files = sorted(glob.glob(file_pattern))
    
    # 2. Filter strictly for files ending in June (06), July (07), August (08), or September (09)
    jjas_files = [f for f in all_files if any(f.endswith(f"-0{m}.nc") for m in [6, 7, 8, 9])]
    
    # 3. Keep only the first 12 files (4 months/year * 3 years = 12 files)
    target_files = jjas_files[:12]
    
    # 4. Load the isolated dataset (Keeping decode_times=False bypasses calendar errors)
    ds = xr.open_mfdataset(target_files, combine='by_coords', decode_times=False)
    
    # 5. Calculate Total Precipitation (PRECC + PRECL) in mm/day
    prect = (ds['PRECC'] + ds['PRECL']) * 86400 * 1000
    
    # 6. Average over the isolated JJAS time steps
    return prect.mean(dim='time')

# 1. Define your file paths 
control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'          # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'        # 100% Reduction

# 2. Process the data
print("Processing CESM JJAS Precipitation Data...")
precip_ctrl = get_jjas_precip(control_files)
precip_50   = get_jjas_precip(case50_files)
precip_100  = get_jjas_precip(case100_files)

# 3. Calculate Anomalies (Experiment - Control)
anom_50  = precip_50 - precip_ctrl
anom_100 = precip_100 - precip_ctrl

# 4. Set up the Cartopy Map Projection
lon_min, lon_max = 50.0, 100.0
lat_min, lat_max = 0.0, 35.0

fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)

# 5. Plot Panel 1: Control Climatology (Absolute values)
ax1 = axes[0]
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
ax1.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')

plot_ctrl = precip_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=0, vmax=20, cmap='YlGnBu', 
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Mean JJAS Precipitation (mm/day)', fontsize=12)

# 6. Plot Panels 2 & 3: Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction Anomaly', '100% Reduction Anomaly']

for i, ax in enumerate(axes[1:]):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')

    plot_anom = anom_data[i].plot.pcolormesh(
       ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-5, vmax=5, cmap='RdBu', 
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# Add a shared colorbar for the two anomaly plots
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('JJAS Precipitation Anomaly (mm/day)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

# 7. Final Formatting
plt.suptitle('JJAS Monsoon Precipitation Response to Aerosol Reductions (Years 1-3)', fontsize=18, y=1.02)
plt.savefig('ind_monsoon_precip.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'ind_monsoon_precip.png'")
plt.show()
