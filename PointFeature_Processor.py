import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio
import time
from tqdm import tqdm
import json
import random
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIGURATION AREA =================
DATA_ROOT = r"D:\Inseason_mapping"
OUTPUT_DIR = r"D:\Inseason_mapping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Spectral Indices
STANDARD_INDICES = ['EVI', 'NDVI', 'NDWI', 'NPCI', 'PPR', 'RVI', 'SAVI']
ALL_INDICES = STANDARD_INDICES

# 2. Crop Class Mapping
CROP_MAPPING = {
    'Cotton': 0, 'SCorn': 1, 'SWheat': 2,
    'DWheat': 3, 'Agroforestry': 4, 'Other': 5
}
INV_CROP_MAPPING = {v: k for k, v in CROP_MAPPING.items()}
YEARS = range(2018, 2025)
FILE_SUFFIX = "_Norm"


# ================= 🛠️ SPATIAL GRID FUNCTIONS =================
def create_and_assign_spatial_grid(ref_tif_path, output_shp_dir):
    print("\n🌐 [Grid Init] Generating 5km x 5km Spatial Blocks...")
    with rasterio.open(ref_tif_path) as src:
        bounds = src.bounds
        target_crs = src.crs

    if target_crs.is_projected:
        grid_size = 5000
    else:
        grid_size = 0.045
        print("⚠️ Warning: CRS is geographic (degrees). Using 0.045 degrees for 5km grid.")

    grids = []
    for x in np.arange(bounds.left, bounds.right, grid_size):
        for y in np.arange(bounds.bottom, bounds.top, grid_size):
            grids.append(box(x, y, x + grid_size, y + grid_size))

    grid_gdf = gpd.GeoDataFrame({'geometry': grids}, crs=target_crs)


    grid_gdf['Grid_ID'] = range(1, len(grid_gdf) + 1)


    np.random.seed(910)
    grid_gdf['split'] = np.random.choice(['Train', 'Val'], size=len(grid_gdf), p=[0.7, 0.3])

    grid_shp_path = os.path.join(output_shp_dir, "Spatial_Holdout_Grid_5km.shp")
    grid_gdf.to_file(grid_shp_path)
    print(f"✅ Master Grid saved to: {grid_shp_path}")

    return grid_gdf


def get_points_gdf_from_shp(year, target_crs):
    shp_pattern = os.path.join(DATA_ROOT, f"Export_Train_*_{year}.shp")
    shp_files = glob.glob(shp_pattern)
    gdf_list = []

    if not shp_files: return None

    for shp_path in shp_files:
        filename = os.path.basename(shp_path)
        try:
            crop_name = filename.split('Train_')[1].split(f'_{year}')[0]
        except:
            continue
        if crop_name not in CROP_MAPPING: continue

        gdf = gpd.read_file(shp_path)
        if gdf.crs != target_crs: gdf = gdf.to_crs(target_crs)
        gdf['Label'] = CROP_MAPPING[crop_name]
        gdf['CropType'] = crop_name
        gdf['Year'] = year
        gdf_list.append(gdf)

    if not gdf_list: return None
    merged_gdf = pd.concat(gdf_list, ignore_index=True)
    return merged_gdf


def get_file_path(year, index_name):
    if index_name in STANDARD_INDICES:
        return os.path.join(DATA_ROOT, f"{year}_{index_name}.tif")
    return None


# ================= 🚀 CORE PIPELINE =================
def calculate_stats(target_crs):
    print("\n🚀 [Step 1] Scanning feature distribution (RVI: Max)...")
    collected_rvi = []
    MAX_SAMPLES = 50000

    for year in tqdm(YEARS, desc="Sampling Stats"):
        pts_gdf = get_points_gdf_from_shp(year, target_crs)
        if pts_gdf is None: continue

        coords = [(p.x, p.y) for p in pts_gdf.geometry]
        if len(coords) > MAX_SAMPLES:
            coords = random.sample(coords, MAX_SAMPLES)

        path_rvi = get_file_path(year, 'RVI')
        if os.path.exists(path_rvi):
            with rasterio.open(path_rvi) as src:
                val = np.array(list(src.sample(coords)))
                if val.shape[1] > 18: val = val[:, :18]
                collected_rvi.append(val.flatten())

    stats = {}
    if collected_rvi:
        all_rvi = np.concatenate(collected_rvi)
        stats['RVI'] = float(np.max(all_rvi))
    else:
        return None
    return stats


if __name__ == "__main__":
    start = time.time()

    ref_path = get_file_path(2018, 'EVI')
    with rasterio.open(ref_path) as src:
        target_crs = src.crs

    master_grid = create_and_assign_spatial_grid(ref_path, OUTPUT_DIR)

    stats = calculate_stats(target_crs)
    if not stats: exit()

    print("\n🚀 [Step 2] Executing data extraction & normalization...")

    gee_export_list = []

    for year in tqdm(YEARS, desc="Processing Years"):
        pts_gdf = get_points_gdf_from_shp(year, target_crs)
        if pts_gdf is None: continue

        # ================= 🔥 SPATIAL JOIN 🔥 =================
        pts_with_split = gpd.sjoin(pts_gdf, master_grid[['geometry', 'split', 'Grid_ID']], how='inner',
                                   predicate='within')

        coords = [(p.x, p.y) for p in pts_with_split.geometry]
        labels = pts_with_split['Label'].values

        features_dict = {}
        valid_year = True

        for idx_name in ALL_INDICES:
            tif_path = get_file_path(year, idx_name)
            if not os.path.exists(tif_path): valid_year = False; break
            with rasterio.open(tif_path) as src:
                val = np.array(list(src.sample(coords)))
                if val.shape[1] > 18: val = val[:, :18]
                features_dict[idx_name] = val

        if not valid_year: continue

        X_raw = np.stack([features_dict[idx] for idx in ALL_INDICES], axis=1)
        y_raw = np.array(labels)

        # ================= 🔥 DATA FILTERING 🔥 =================
        evi_idx = ALL_INDICES.index('EVI')
        mask_evi_abnormal = np.any((X_raw[:, evi_idx, :] > 1) | (X_raw[:, evi_idx, :] < -1), axis=1)
        mask_keep = ~mask_evi_abnormal

        X_clean = X_raw[mask_keep]
        y_clean = y_raw[mask_keep]

        if len(y_clean) == 0: continue


        clean_pts_gdf = pts_with_split.iloc[mask_keep].copy()
        gee_export_list.append(clean_pts_gdf[['geometry', 'CropType', 'Label', 'Year', 'Grid_ID', 'split']])

        # ================= 🔥 NORMALIZATION & CLIPPING 🔥 =================
        X_norm = X_clean.copy()
        rvi_idx = ALL_INDICES.index('RVI')
        X_norm[:, rvi_idx, :] = np.clip(X_clean[:, rvi_idx, :] / (stats['RVI'] + 1e-6), 0.0, 1.0)
        for name in ALL_INDICES:
            if name != 'RVI':
                idx = ALL_INDICES.index(name)
                X_norm[:, idx, :] = np.clip(X_clean[:, idx, :], -1.0, 1.0)

        # ================= 🔥 SAVE FULL ARRAYS & ALIGNED METADATA 🔥 =================
        np.save(os.path.join(OUTPUT_DIR, f"X_{year}{FILE_SUFFIX}.npy"), X_norm.astype(np.float32))
        np.save(os.path.join(OUTPUT_DIR, f"y_{year}{FILE_SUFFIX}.npy"), y_clean.astype(np.int64))

        meta_df = clean_pts_gdf[['CropType', 'Label', 'Year', 'Grid_ID', 'split']].copy()
        meta_df.to_csv(os.path.join(OUTPUT_DIR, f"Meta_{year}{FILE_SUFFIX}.csv"), index=False)

    # ================= 🔥 SAVE GEE SHAPEFILE 🔥 =================
    if gee_export_list:
        final_gee_gdf = pd.concat(gee_export_list, ignore_index=True)
        gee_shp_path = os.path.join(OUTPUT_DIR, "All_Cleaned_Samples_For_GEE.shp")
        final_gee_gdf.to_file(gee_shp_path)

    # ================= 🔥 SAVE PARAMETERS JSON 🔥 =================
    norm_params = {name: {'divisor': stats['RVI'] if name == 'RVI' else 1.0,
                          'action': 'div_max_clip_0_1' if name == 'RVI' else 'clip_minus1_1'}
                   for name in ALL_INDICES}
    with open(os.path.join(OUTPUT_DIR, "Sample_2018_2024.json"), 'w') as f:
        json.dump(norm_params, f, indent=4)

    print(f"\n🎉 Done! Time: {(time.time() - start) / 60:.2f} min")