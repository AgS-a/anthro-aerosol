import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_36mo_global_aodvis(file_prefix):
    """Loads CESM files, extracts the first 36 continuous months of AODVIS."""
    pattern = file_prefix
    all_files = sorted(glob.glob(pattern))
    
    # Load dataset
    ds = xr.open_mfdataset(all_files, combine='by_coords', decode_times=False)
    
    # Isolate exactly the first 36 continuous months
    ds = ds.isel(time=slice(0, 36))
    
    # Extract Aerosol Optical Depth at 550nm
    aodvis = ds['AODVIS']
    
    # Average over time
    return aodvis.mean(dim='time')

# 1. Process the data
print("Processing 36 Months of Global AODVIS Data...")

control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'    # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'   # 100% Reduction

aod_ctrl = get_36mo_global_aodvis(control_files)
aod_50   = get_36mo_global_aodvis(case50_files)
aod_100  = get_36mo_global_aodvis(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
anom_50  = aod_50 - aod_ctrl
anom_100 = aod_100 - aod_ctrl

# 3. Set up the Global Map using Robinson Projection
fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={'projection': ccrs.Robinson()})
plt.subplots_adjust(wspace=0.1)

# 4. Plot Panel 1: Control Climatology (Absolute AOD)
ax1 = axes[0]
ax1.set_global()
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)

# YlOrBr (Yellow-Orange-Brown) is the standard perceptual colormap for atmospheric haze
plot_ctrl = aod_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=0, vmax=0.5, cmap='YlOrBr', 
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Aerosol Optical Depth (550nm)', fontsize=12)

# 5. Plot Panels 2 & 3: AOD Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction', '100% Reduction']

for i, ax in enumerate(axes[1:]):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # RdBu_r colormap: Blue is negative (cleaner air), Red is positive (more aerosol)
    # We set limits tightly around -0.3 to 0.3 to make the regional reductions pop
    plot_anom = anom_data[i].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-0.3, vmax=0.3, cmap='RdBu_r', 
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# 6. Shared colorbar for anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('AODVIS Anomaly', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('Global Aerosol Optical Depth (36-Month Mean)', fontsize=18, y=1.05)
plt.savefig('opt_depth.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'opt_depth.png'")
plt.show()
