import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_36mo_global_cdnumc(file_prefix):
    """Loads CESM files, extracts the first 36 continuous months of CDNUMC."""
    # Grab all monthly files (no seasonal filter needed for global annual mean)
    pattern = file_prefix
    all_files = sorted(glob.glob(pattern))
    
    # Load dataset
    ds = xr.open_mfdataset(all_files, combine='by_coords', decode_times=False)
    
    # Isolate exactly the first 36 continuous months
    ds = ds.isel(time=slice(0, 36))
    
    # Extract Column Cloud Droplet Number Concentration (CDNUMC)
    # CESM outputs this in massive numbers. We divide by 1e10 so the colorbar 
    # labels read "5" instead of "50,000,000,000"
    cdnumc = ds['CDNUMC'] / 1e10
    
    # Average over time
    return cdnumc.mean(dim='time')

# 1. Define file prefixes
# Assuming your files are named something like: asc0.cam.h0.0001-01.nc
print("Processing 36 Months of Global Cloud Droplet Data...")

control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'    # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'   # 100% Reduction

cdnumc_ctrl = get_36mo_global_cdnumc(control_files)
cdnumc_50   = get_36mo_global_cdnumc(case50_files)
cdnumc_100  = get_36mo_global_cdnumc(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
anom_50  = cdnumc_50 - cdnumc_ctrl
anom_100 = cdnumc_100 - cdnumc_ctrl

# 3. Set up the Global Map using Robinson Projection
fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={'projection': ccrs.Robinson()})
plt.subplots_adjust(wspace=0.1)

# 4. Plot Panel 1: Control Climatology (Absolute Droplet Number)
ax1 = axes[0]
ax1.set_global()
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Note: The map projection is Robinson, but the data transform is ALWAYS PlateCarree
plot_ctrl = cdnumc_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=0, vmax=20, cmap='BuPu', # Purple/Blue sequential colormap
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Cloud Droplet Number (10$^{10}$ m$^{-2}$)', fontsize=12)

# 5. Plot Panels 2 & 3: Droplet Number Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction Anomaly', '100% Reduction Anomaly']

for i, ax in enumerate(axes[1:]):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # Diverging colormap: Brown (fewer droplets) to Green (more droplets)
    plot_anom = anom_data[i].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-15, vmax=15, cmap='BrBG', 
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# 6. Shared colorbar for anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('Droplet Number Anomaly (10$^{10}$ m$^{-2}$)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('Global Cloud Droplet Concentration Response (36-Month Mean)', fontsize=18, y=1.05)
plt.savefig('global_cdnumc_3cases_36mo.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'global_cdnumc_3cases_36mo.png'")
plt.show()
