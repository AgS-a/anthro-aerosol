import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_36mo_global_cldtot(file_prefix):
    """Loads CESM files, extracts the first 36 months of CLDTOT, and converts to percentage."""
    pattern = file_prefix
    all_files = sorted(glob.glob(pattern))
    
    # Load dataset
    ds = xr.open_mfdataset(all_files, combine='by_coords', decode_times=False)
    
    # Isolate exactly the first 36 continuous months
    ds = ds.isel(time=slice(0, 36))
    
    # Extract Total Cloud Fraction and convert to percentage
    cldtot_pct = ds['CLDTOT'] * 100.0
    
    # Average over time
    return cldtot_pct.mean(dim='time')

# 1. Process the data
print("Processing 36 Months of Global Cloud Fraction Data...")
control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'    # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'   # 100% Reduction
cld_ctrl = get_36mo_global_cldtot(control_files)
cld_50   = get_36mo_global_cldtot(case50_files)
cld_100  = get_36mo_global_cldtot(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
anom_50  = cld_50 - cld_ctrl
anom_100 = cld_100 - cld_ctrl

# 3. Set up the Global Map using Robinson Projection
fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={'projection': ccrs.Robinson()})
plt.subplots_adjust(wspace=0.1)

# 4. Plot Panel 1: Control Climatology (Absolute Cloud Percentage)
ax1 = axes[0]
ax1.set_global()
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)

# 'Blues' is perfect for visualizing cloud cover density
plot_ctrl = cld_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=0, vmax=100, cmap='Blues', 
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Total Cloud Fraction (%)', fontsize=12)

# 5. Plot Panels 2 & 3: Cloud Fraction Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction Anomaly', '100% Reduction Anomaly']

for i, ax in enumerate(axes[1:]):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    # RdBu colormap: Red is negative (fewer clouds), Blue is positive (more clouds)
    plot_anom = anom_data[i].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-5, vmax=5, cmap='RdBu', 
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# 6. Shared colorbar for anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('Cloud Fraction Anomaly (%)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('Global Total Cloud Fraction Response (36-Month Mean)', fontsize=18, y=1.05)
plt.savefig('global_cldtot_3cases.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'global_cldtot_3cases.png'")
plt.show()
