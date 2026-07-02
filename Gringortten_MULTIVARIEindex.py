# -*- coding: utf-8 -*-
"""
@author: Sarah 
"""

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import norm
import pandas as pd
from functools import reduce

# Load NetCDF data
print("Opening NetCDF file...")
file_path = "/SIM/safran_1958-2024.nc"
ds = xr.open_dataset(file_path)

# Load France map background
print("Loading France map background...")
FRANCE = gpd.read_file("/DATA/FOND_de_CARTE/FRANCE.gpkg")
if FRANCE.crs != "EPSG:4326":
    FRANCE = FRANCE.to_crs("EPSG:4326")
france_shape = FRANCE.geometry.unary_union

# Create geographical mask for France
print("Creating geographical mask for France...")
lat2d = ds['latc'].values
lon2d = ds['lonc'].values
points_geo = np.array([Point(x, y) for x, y in zip(lon2d.ravel(), lat2d.ravel())])
mask = np.array([france_shape.contains(pt) for pt in points_geo])
indices_valides = np.where(mask)[0]
i_lat, i_lon = np.unravel_index(indices_valides, lat2d.shape)
spatial_mask = np.zeros(lat2d.shape, dtype=bool)
spatial_mask[i_lat, i_lon] = True

# Define periods
ref_start = "1991-01-01"
ref_end = "2020-12-31"
analysis_start = "2022-01-01"
analysis_end = "2024-12-31"

# Extract variables and periods
def extract_var_period(var_name):
    """Extracts a variable for both reference and analysis periods,
    and applies the spatial mask."""
    var_full = ds[var_name]
    spatial_mask_da = xr.DataArray(spatial_mask, coords=[var_full.lat, var_full.lon], dims=["lat", "lon"])
    
    var_ref = var_full.sel(time=slice(ref_start, ref_end)).where(spatial_mask_da)
    var_analysis = var_full.sel(time=slice(analysis_start, analysis_end)).where(spatial_mask_da)
    return var_ref, var_analysis

variables_to_process = ['T_Q', 'VPD', 'SWI_Q', 'ETP_Q', 'EVAP_Q', 'WG_RACINE_Q']
ref_data, analysis_data = {}, {}
for var in variables_to_process:
    print(f"Extracting variable: {var}")
    ref_data[var], analysis_data[var] = extract_var_period(var)

# Select 3 points
selected_points = [
    (i_lat[0], i_lon[0]),
    (i_lat[82], i_lon[82]),
    (i_lat[35], i_lon[35])
]

lonc_points = [lon2d[i, j] for i, j in selected_points]
latc_points = [lat2d[i, j] for i, j in selected_points]

def select_points(data, points):
    """Selects time series for the specified points."""
    selected_list = [data.isel(lat=i, lon=j) for i, j in points]
    return xr.concat(selected_list, dim='points').transpose('time', 'points')

print("\nSelecting points for all variables...")
for var in variables_to_process:
    ref_data[var] = select_points(ref_data[var], selected_points)
    analysis_data[var] = select_points(analysis_data[var], selected_points)

# Monthly resampling
def resample_monthly_mean(data): 
    return data.resample(time="1MS").mean()

def resample_monthly_sum(data): 
    return data.resample(time="1MS").sum()

monthly_mean_vars = ['T_Q', 'VPD', 'SWI_Q', 'WG_RACINE_Q']
monthly_sum_vars = ['ETP_Q', 'EVAP_Q']

print("\nResampling data to monthly scale...")
for var in monthly_mean_vars:
    ref_data[var] = resample_monthly_mean(ref_data[var])
    analysis_data[var] = resample_monthly_mean(analysis_data[var])

for var in monthly_sum_vars:
    ref_data[var] = resample_monthly_sum(ref_data[var])
    analysis_data[var] = resample_monthly_sum(analysis_data[var])

# Compute moisture indices
print("\nComputing moisture indices (DEF, AET_PET_ratio)...")
ref_data["DEF"] = ref_data["ETP_Q"] - ref_data["EVAP_Q"]
analysis_data["DEF"] = analysis_data["ETP_Q"] - analysis_data["EVAP_Q"]

ref_data["AET_PET_ratio"] = ref_data["EVAP_Q"] / ref_data["ETP_Q"]
analysis_data["AET_PET_ratio"] = analysis_data["EVAP_Q"] / analysis_data["ETP_Q"]

ref_data["AET_PET_ratio"] = ref_data["AET_PET_ratio"].where(np.isfinite(ref_data["AET_PET_ratio"]))
analysis_data["AET_PET_ratio"] = analysis_data["AET_PET_ratio"].where(np.isfinite(analysis_data["AET_PET_ratio"]))


# Gringorten normalization function for multivariate indices
def normalize_gringorten_multivariate(ref_data_dict, analysis_data_dict, multivariate_combinations_list):
    """
    Computes standardized multivariate Gringorten indices by adjusting the direction
    of the inequalities based on the target anomaly type.
    """
    normalized_multivariate_data = {}

    for combo_spec in multivariate_combinations_list:
        combo_name = combo_spec['name']
        combo_vars = combo_spec['vars'] 
        print(f"\nProcessing multivariate combination: '{combo_name}'...")

        rolling_ref_series = []
        rolling_target_series = []
        
        # Validation and rolling mean calculation
        missing_var = False
        for var_name, scale, direction in combo_vars:
            if var_name not in ref_data_dict or var_name not in analysis_data_dict:
                print(f"  Error: Variable '{var_name}' not found. Combination '{combo_name}' ignored.")
                missing_var = True
                break
            
            rolling_ref_series.append(
                ref_data_dict[var_name].rolling(time=scale, center=True, min_periods=1).mean().transpose('time', 'points'))
            rolling_target_series.append(
                analysis_data_dict[var_name].rolling(time=scale, center=True, min_periods=1).mean().transpose('time', 'points'))

        if missing_var:
            continue

        time_coords_target = rolling_target_series[0].time
        points_coords = rolling_target_series[0].points
        ntime_target = len(time_coords_target)
        npoints = len(points_coords)

        z_combined_index = np.full((ntime_target, npoints), np.nan) 

        # Loop over each geographical point
        for i_point in range(npoints):
            series_ref_for_point = [s.isel(points=i_point).values for s in rolling_ref_series]
            series_target_for_point = [s.isel(points=i_point).values for s in rolling_target_series]

            ref_triplets = np.array(series_ref_for_point).T 
            target_triplets = np.array(series_target_for_point).T 

            valid_ref_mask = ~np.any(np.isnan(ref_triplets), axis=1)
            valid_ref_triplets = ref_triplets[valid_ref_mask]
            
            n_ref = len(valid_ref_triplets) 
            if n_ref == 0:
                print(f"  Point {i_point}: No valid reference data for '{combo_name}'. Point ignored.")
                continue

            z_vals_for_point = np.full(ntime_target, np.nan)

            # Iterate over each target time step
            for t_idx in range(ntime_target):
                current_target_triplet = target_triplets[t_idx, :]
                
                if np.any(np.isnan(current_target_triplet)):
                    continue

                # Dynamic condition building based on anomaly direction
                conditions = []
                for var_idx, (var_name, scale, direction) in enumerate(combo_vars):
                    if direction == 'less':
                        # For SWI (drought = historically lower or equal to the target crisis)
                        conditions.append(valid_ref_triplets[:, var_idx] <= current_target_triplet[var_idx])
                    elif direction == 'greater':
                        # For T_Q and VPD (heat/aridity = historically higher or equal)
                        conditions.append(valid_ref_triplets[:, var_idx] >= current_target_triplet[var_idx])
                    else:
                        raise ValueError(f"Unknown inequality direction: {direction}")
                
                # Intersection (logical AND) of all co-occurring crises
                combined_condition = reduce(np.logical_and, conditions)
                n_i = np.sum(combined_condition) 

                # Gringorten empirical plotting position formula
                p_i = (n_i - 0.44) / (n_ref + 0.12)
                
                # Invert probabilities so that Z-score increases with severity
                # (Rare historical event = small n_i -> small p_i -> 1 - p_i close to 1 -> highly positive Z-score)
                p_i = 1 - p_i
                
                p_i = np.clip(p_i, 1e-10, 1 - 1e-10) 
                z_score = norm.ppf(p_i)
                z_vals_for_point[t_idx] = z_score
            
            z_combined_index[:, i_point] = z_vals_for_point

        da_z = xr.DataArray(z_combined_index, 
                            coords={'time': time_coords_target, 'points': points_coords}, 
                            dims=['time', 'points'],
                            name=combo_name) 
        
        normalized_multivariate_data[combo_name] = da_z
        print(f"Normalization completed for combination '{combo_name}'.")

    return normalized_multivariate_data

# Definition of multivariate combinations to compute
# 'less'    -> Dry (Deficit)
# 'greater' -> Hot / Arid (Excess)
multivariate_combinations_list = [
    {
        'name': 'SMI6_STI3_SVPD3', 
        'vars': [
            ('SWI_Q', 6, 'less'),       # Soil drought (Deficit)
            ('T_Q', 3, 'greater'),      # Heatwave (Excess)
            ('VPD', 3, 'greater')       # Atmospheric evaporative demand (Excess)
        ]
    },
    {
        'name': 'SMI12_STI6_SVPD6', 
        'vars': [
            ('SWI_Q', 12, 'less'),
            ('T_Q', 6, 'greater'),
            ('VPD', 6, 'greater')
        ]
    },
    {
        'name': 'SMI3_STI1_SVPD1', 
        'vars': [
            ('SWI_Q', 3, 'less'),
            ('T_Q', 1, 'greater'),
            ('VPD', 1, 'greater')
        ]
    }
]

# Apply multivariate normalization
print("\nStarting multivariate normalization...")
normalized_multivariate_sets = normalize_gringorten_multivariate(ref_data, analysis_data, multivariate_combinations_list)
print("\nMultivariate normalization completed.")

# Function to convert DataArray to DataFrame with coordinates
def da_to_df_with_coords(da, lon_list, lat_list, var_name):
    df = da.to_pandas().copy()
    df.columns.name = None 
    df = df.reset_index() 
    df = df.melt(id_vars='time', var_name='point', value_name=var_name)
    
    df['lonc'] = df['point'].apply(lambda i: lon_list[i])
    df['latc'] = df['point'].apply(lambda i: lat_list[i])
    
    return df[['time', 'point', 'lonc', 'latc', var_name]]

# Convert each computed multivariate index to a DataFrame
dfs_multivariate = []

for combo_name, da in normalized_multivariate_sets.items():
    df = da_to_df_with_coords(da, lonc_points, latc_points, combo_name)
    dfs_multivariate.append(df)
    print(f"\nDataFrame for '{combo_name}' generated → {df.shape[0]} rows")
    print(df.head(2))

# Merge all multivariate DataFrames into a single one
if dfs_multivariate:
    print("\nMerging multivariate DataFrames...")
    df_merged_multivariate = reduce(
        lambda left, right: pd.merge(left, right, on=["time", "point", "lonc", "latc"], how="outer"), 
        dfs_multivariate
    )
    df_merged_multivariate = df_merged_multivariate.sort_values(by=["point", "time"]).reset_index(drop=True)

    print("\nMerged multivariate DataFrame:")
    print(df_merged_multivariate.head())

    # Save outputs
    output_path_multivariate = "/MULTIVARIE/df_merged_multivariate.csv"
    df_merged_multivariate.to_csv(output_path_multivariate, index=False)
    print(f"\nComplete multivariate DataFrame saved to: {output_path_multivariate}")
else:
    print("No multivariate indices were computed.")
