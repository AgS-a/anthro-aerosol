import xarray as xr
import glob

# 1. Grab all the NetCDF files in your local directory
target_files = glob.glob('emissions-cmip6_*.nc')

for file_in in target_files:
    # Change the prefix so you know these are your global files
    file_out = f"clean_global_{file_in}"
    
    print(f"Processing: {file_in}...")
    
    # Open the dataset
    ds = xr.open_dataset(file_in, decode_times=False)
    
    # Dynamically find the emission variable
    emission_vars = [var for var in ds.data_vars if 'emiss' in var or 'SO2' in var or 'so4' in var]
    
    for var in emission_vars:
        # Capture the original attributes (units, long_name, etc.)
        original_attrs = ds[var].attrs.copy()
        
        # Apply the 90% reduction GLOBALLY across the entire array
        # No masking required!
        ds[var] = ds[var] * 0.1
        
        # Restore the attributes (CRITICAL: CESM will crash if these are missing)
        ds[var].attrs = original_attrs
            
    # Save the cleaned file
    ds.to_netcdf(file_out)
    print(f"✅ Saved: {file_out}\n")

print("Success: All files have been globally cleaned and are ready for upload!")
