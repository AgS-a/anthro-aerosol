import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_36mo_toa_imbalance(file_prefix):
    """Loads CESM files and calculates the Net TOA Energy Flux (FSNT - FLNT)."""
    # Grab all monthly files
    pattern = file_prefix
    all_files = sorted(glob.glob(pattern))
    
    # Load dataset
    ds = xr.open_mfdataset(all_files, combine='by_coords', decode_times=False)
    
    # Isolate exactly the first 36 continuous months
    ds = ds.isel(time=slice(0, 36))
    
    # Calculate Net TOA Flux (Incoming Shortwave - Outgoing Longwave)
    toa_net = ds['FSNT'] - ds['FLNT']
    
    # Average over time
    return toa_net.mean(dim='time')

# 1. Process the data
print("Processing 36 Months of Global TOA Energy Data...")

control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'    # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'   # 100% Reduction

toa_ctrl = get_36mo_toa_imbalance(control_files)
toa_50   = get_36mo_toa_imbalance(case50_files)
toa_100  = get_36mo_toa_imbalance(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
anom_50  = toa_50 - toa_ctrl
anom_100 = toa_100 - toa_ctrl

# 3. Set up the Global Map using Robinson Projection
fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={'projection': ccrs.Robinson()})
plt.subplots_adjust(wspace=0.1)

# 4. Plot Panel 1: Control Climatology (Absolute TOA Flux)
ax1 = axes[0]
ax1.set_global()
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Diverging colormap centered on zero. Tropics gain heat (Red), Poles lose heat (Blue)
plot_ctrl = toa_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=-100, vmax=100, cmap='RdBu_r', 
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Net TOA Flux ($W/m^2$)', fontsize=12)

# 5. Plot Panels 2 & 3: TOA Flux Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction Anomaly', '100% Reduction Anomaly']

for i, ax in enumerate(axes[1:]):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # Red means gaining MORE energy than control (Warming signal)
    plot_anom = anom_data[i].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-5, vmax=5, cmap='RdBu_r', 
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# 6. Shared colorbar for anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('TOA Flux Anomaly ($W/m^2$)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('Global Top of Atmosphere Energy Imbalance Response', fontsize=18, y=1.05)
plt.savefig('global_toa_imbalance_3cases.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'global_toa_imbalance_3cases.png'")
plt.show()
