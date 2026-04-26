import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

def get_jjas_circulation(file_pattern):
    """Loads CESM files, extracts JJAS 850hPa winds & PSL for the first 3 years."""
    # 1. Grab all files matching the pattern and sort them chronologically
    all_files = sorted(glob.glob(file_pattern))
    
    # 2. Filter strictly for files ending in June (06), July (07), August (08), or September (09)
    jjas_files = [f for f in all_files if any(f.endswith(f"-0{m}.nc") for m in [6, 7, 8, 9])]
    
    # 3. Keep only the first 12 files (4 months/year * 3 years = 12 files)
    target_files = jjas_files[:12]
    
    # 4. Load the isolated dataset (Keeping decode_times=False bypasses calendar errors)
    ds = xr.open_mfdataset(target_files, combine='by_coords', decode_times=False)

    # 5. Extract Sea Level Pressure and convert from Pascals to hPa (millibars)
    psl = (ds['PSL'] / 100.0).mean(dim='time')

    # 6. Extract U and V winds at the 850 hPa level
    u850 = ds['U'].sel(lev=850, method='nearest').mean(dim='time')
    v850 = ds['V'].sel(lev=850, method='nearest').mean(dim='time')

    return psl, u850, v850

# 1. Define file paths
control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h0.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h0.*.nc'    # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h0.*.nc'   # 100% Reduction

print("Processing JJAS Circulation Data for Years 1-3...")
psl_ctrl, u_ctrl, v_ctrl = get_jjas_circulation(control_files)
psl_50, u_50, v_50       = get_jjas_circulation(case50_files)
psl_100, u_100, v_100    = get_jjas_circulation(case100_files)

# 2. Calculate Anomalies (Experiment - Control)
psl_anom_50 = psl_50 - psl_ctrl
u_anom_50   = u_50 - u_ctrl
v_anom_50   = v_50 - v_ctrl

psl_anom_100 = psl_100 - psl_ctrl
u_anom_100   = u_100 - u_ctrl
v_anom_100   = v_100 - v_ctrl

# 3. Set up the Map
lon_min, lon_max = 50.0, 100.0
lat_min, lat_max = 0.0, 35.0

fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.15)

# 4. Define Quiver Thinning (Skip every N grid points so arrows are readable)
skip = 3

# 5. Plot Panel 1: Control Climatology  
ax1 = axes[0]
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=1.2)
ax1.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')

# Plot absolute pressure contours (typically 995 to 1015 hPa over India in summer)
plot_ctrl = psl_ctrl.plot.contourf(
    ax=ax1, transform=ccrs.PlateCarree(),
    x='lon', y='lat',
    levels=np.linspace(995, 1015, 21), cmap='viridis_r', 
    add_colorbar=False
)
# Overlay absolute wind vectors
ax1.quiver(
    u_ctrl.lon[::skip].values, u_ctrl.lat[::skip].values,
    u_ctrl[::skip, ::skip].values, v_ctrl[::skip, ::skip].values,
    transform=ccrs.PlateCarree(), color='black', scale=200, width=0.003
)
ax1.set_title('Control Climatology (0% Reduction)', fontsize=14, pad=10)

cbar_ctrl = fig.colorbar(plot_ctrl, ax=ax1, orientation='horizontal', pad=0.08, fraction=0.05)
cbar_ctrl.set_label('Sea Level Pressure (hPa)', fontsize=12)

# 6. Plot Panels 2 & 3: Anomalies
anom_data_psl = [psl_anom_50, psl_anom_100]
anom_data_u = [u_anom_50, u_anom_100]
anom_data_v = [v_anom_50, v_anom_100]
titles = ['50% Reduction Anomaly', '100% Reduction Anomaly']

for i, ax in enumerate(axes[1:]):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':')
    
    # Plot pressure anomalies
    plot_anom = anom_data_psl[i].plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(),
        x='lon', y='lat',
        levels=np.linspace(-3, 3, 21), cmap='RdBu_r', # Red is lower pressure (stronger vacuum)
        add_colorbar=False
    )
    
    # Overlay wind anomaly vectors
    ax.quiver(
        u_ctrl.lon[::skip].values, u_ctrl.lat[::skip].values,
        anom_data_u[i][::skip, ::skip].values, anom_data_v[i][::skip, ::skip].values,
        transform=ccrs.PlateCarree(), color='black', scale=50, width=0.003
    )
    
    ax.set_title(titles[i], fontsize=14, pad=10)

# Shared colorbar for pressure anomalies
cbar_anom_ax = fig.add_axes([0.41, 0.08, 0.49, 0.04]) 
cbar_anom = fig.colorbar(plot_anom, cax=cbar_anom_ax, orientation='horizontal')
cbar_anom.set_label('Sea Level Pressure Anomaly (hPa)', fontsize=14)
cbar_anom.ax.tick_params(labelsize=12)

plt.suptitle('JJAS Monsoon Circulation Response (Years 1-3): 850 hPa Winds and Sea Level Pressure', fontsize=18, y=1.02)
plt.savefig('circulation.png', bbox_inches='tight', dpi=300)
print("Success! Saved as 'circulation.png'")
plt.show()
