import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_36mo_fsns(file_pattern):
    """Loads CESM files, extracts the FIRST 36 months of FSNS, and averages."""
    ds = xr.open_mfdataset(file_pattern, combine='by_coords', decode_times=False)

    # Isolate the first 36 months (Years 1 to 3: The Termination Shock)
    ds = ds.isel(time=slice(0, 36))

    # Extract Net Solar Flux at Surface (W/m^2)
    fsns = ds['FSNS']

    # Average over those 36 months
    return fsns.mean(dim='time')

# 1. Define file paths
control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'          # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'        # 100% Reduction

print("Processing 36 Months of CESM Solar Radiation Data...")
fsns_ctrl = get_36mo_fsns(control_files)
fsns_50   = get_36mo_fsns(case50_files)
fsns_100  = get_36mo_fsns(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
anom_50  = fsns_50 - fsns_ctrl
anom_100 = fsns_100 - fsns_ctrl

# 3. Set up the Map
fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.Robinson()})
plt.subplots_adjust(wspace=0.15)

# 4. Plot Panel 1: Control Climatology (Absolute values in W/m^2)
ax1 = axes[0]
ax1.set_global() # THE FIX: Use set_global() instead of set_extent()
ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
ax1.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')

# 'magma' is a great perceptual colormap for intense radiation/energy
plot_ctrl = fsns_ctrl.plot.pcolormesh(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    vmin=150, vmax=300, cmap='magma',
    add_colorbar=False
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Net Surface Solar Flux ($W/m^2$)', fontsize=12)

# 5. Plot Panels 2 & 3: Radiation Anomalies
anom_data = [anom_50, anom_100]
titles = ['50% Reduction', '100% Reduction']

for i, ax in enumerate(axes[1:]):
    ax.set_global() # THE FIX: Use set_global() instead of set_extent()
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
    
    # Red means MORE radiation hitting the ground (warming), Blue means LESS
    plot_anom = anom_data[i].plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        vmin=-15, vmax=15, cmap='RdBu_r',
        add_colorbar=False
    )
    ax.set_title(titles[i], fontsize=14, pad=10)

# 6. Shared colorbar for anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04])
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal', extend='both')
cbar_anom.set_label('Solar Flux Anomaly ($W/m^2$)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('36-Month Mean Surface Solar Radiation Response to Aerosol Reductions', fontsize=18, y=1.02)
plt.savefig('shortwave.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'shortwave.png'")
plt.show()
