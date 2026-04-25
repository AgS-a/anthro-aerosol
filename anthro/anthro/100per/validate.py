import xarray as xr
import glob
import os

# Grab all the ORIGINAL files
orig_files = glob.glob('emissions-cmip6_*.nc')
# Filter out any files that accidentally start with 'clean_' just to be safe
orig_files = [f for f in orig_files if not f.startswith('clean_')]

print(f"\n{'Target File':<55} | {'Reduction'}")
print("-" * 75)

for orig_file in orig_files:
    clean_file = f"clean_global_{orig_file}"
    
    # Make sure the cleaned file actually exists
    if not os.path.exists(clean_file):
        print(f"❌ Missing clean file for: {orig_file}")
        continue
        
    # Open both datasets
    ds_orig = xr.open_dataset(orig_file, decode_times=False)
    ds_clean = xr.open_dataset(clean_file, decode_times=False)
    
    # Find the emission variable dynamically
    emission_vars = [var for var in ds_orig.data_vars if 'emiss' in var or 'SO2' in var or 'so4' in var]
    
    for var in emission_vars:
        # Sum the entire global 3D/4D array
        orig_sum = ds_orig[var].sum().values
        clean_sum = ds_clean[var].sum().values
        
        # Calculate the percentage difference
        if orig_sum == 0:
             reduction = 0.0 # Prevent divide-by-zero if an array is completely empty
        else:
             reduction = 100 - ((clean_sum / orig_sum) * 100)
             
        # Format the file name so it fits neatly in the terminal
        display_name = orig_file if len(orig_file) < 52 else orig_file[:49] + "..."
        
        # Print the verdict
        if round(reduction, 1) == 100.0:
            print(f"✅ {display_name:<52} | {reduction:>6.2f}%")
        else:
            print(f"❌ {display_name:<52} | {reduction:>6.2f}% (FAILED)")
            
    ds_orig.close()
    ds_clean.close()

print("\nVerification complete!")
