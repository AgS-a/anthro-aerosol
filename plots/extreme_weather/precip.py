import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob

def get_jjas_extremes(file_prefix):
    """Loads daily .h1. files, extracts JJAS over India, and returns 1D rain events."""
    pattern = file_prefix
    all_files = sorted(glob.glob(pattern))
    
    # --- THE FIX: Look inside the file for time steps ---
    valid_files = []
    for f in all_files:
        try:
            # Open temporarily without decoding to prevent crashes
            with xr.open_dataset(f, decode_times=False) as temp:
                if temp.dims.get('time', 0) > 0:
                    valid_files.append(f)
                else:
                    print(f"💀 Found Zombie File: {f} (0 time steps). Skipping!")
        except Exception:
            print(f"❌ Corrupted/Unreadable file: {f}. Skipping!")
            
    print(f"\nLoading {len(valid_files)} valid daily files for {file_prefix}...")
    
    if len(valid_files) == 0:
        print(f"CRITICAL ERROR: No valid files found for {file_prefix}!")
        return np.array([]) # Return empty so the plot doesn't crash

    # Now we safely load only the good files
    ds = xr.open_mfdataset(
        valid_files,
        combine='by_coords', 
        decode_times=True, 
        use_cftime=True, 
        chunks={'time': 30}
    )
    
    ds = ds.sortby('lat').sortby('lon')
    ds = ds.isel(time=slice(0, 1095))
    ds_jjas = ds.sel(time=ds['time'].dt.month.isin([6, 7, 8, 9]))
    
    # SAFETY CHECK: Did the run crash before summer?
    if ds_jjas.dims.get('time', 0) == 0:
        print(f"⚠️ WARNING: {file_prefix} has no June-Sept data! The run likely crashed in spring.")
        return np.array([])
        
    ds_india = ds_jjas.sel(lat=slice(5, 35), lon=slice(65, 95))
    
    prect = (ds_india['PRECC'] + ds_india['PRECL']) * 86400 * 1000
    prect_values = prect.values.flatten()
    rain_events = prect_values[prect_values > 1.0]
    
    print(f" -> Successfully extracted {rain_events.size} extreme events.")
    return rain_events

# Process the data
print("Extracting extreme weather events for all 3 cases...")
control_files = '/scratch/ags/as0_ec/run/as0_ec.cam.h1.*.nc'      # 0% Reduction (Control)
case50_files  = '/scratch/ags/as50/run/as50.cam.h1.*.nc'    # 50% Reduction
case100_files = '/scratch/ags/as100/run/as100.cam.h1.*.nc'   # 100% Reduction
rain_ctrl = get_jjas_extremes(control_files)
rain_50   = get_jjas_extremes(case50_files)
rain_100  = get_jjas_extremes(case100_files)

# Generate the Histogram Plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
bins = np.arange(1, 150, 2)

# Only plot if data actually exists
if rain_ctrl.size > 0:
    ax.hist(rain_ctrl, bins=bins, density=True, histtype='step', linewidth=2.5, color='#1f77b4', label='Control')
if rain_50.size > 0:
    ax.hist(rain_50, bins=bins, density=True, histtype='step', linewidth=2.5, color='#ff7f0e', label='50% Reduction')
    ax.hist(rain_50, bins=bins, density=True, histtype='stepfilled', alpha=0.05, color='#ff7f0e')
if rain_100.size > 0:
    ax.hist(rain_100, bins=bins, density=True, histtype='step', linewidth=2.5, color='#d62728', label='100% Reduction')
    ax.hist(rain_100, bins=bins, density=True, histtype='stepfilled', alpha=0.1, color='#d62728')

ax.set_title('Probability Density of Daily Monsoon Rainfall Over India (JJAS)', fontsize=16, pad=15)
ax.set_xlabel('Daily Precipitation Intensity (mm/day)', fontsize=14)
ax.set_ylabel('Probability Density (Log Scale)', fontsize=14)
ax.set_yscale('log')
ax.axvline(50, color='black', linestyle='--', alpha=0.7, label='Extreme Event Threshold (>50 mm/day)')
ax.legend(loc='upper right', fontsize=12)

plt.savefig('india_precip_extremes_3cases_pdf.png', bbox_inches='tight')
print("\nSuccess! Saved as 'india_precip_extremes_3cases_pdf.png'")
plt.show()
