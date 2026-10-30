import rasterio
import numpy as np
import os
from rasterio.windows import Window
from tqdm import tqdm

# ================= CONFIGURATION AREA =================
DATA_ROOT = r"D:\Inseason_mapping"
OUTPUT_DIR = r"D:\Inseason_mapping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RF_FILE_TEMPLATE = "Export_RF_Map_{year}.tif"

WEIGHT_SIMPLE = 1.0
WEIGHT_ABAB = 0.8
WEIGHT_AABB = 0.5
WEIGHT_2_UNORDERED = 0.3
WEIGHT_COMPLEX = 0.1

Agroforestry_CLASS_ID = 4
WEIGHT_Agroforestry_FORCE = 1.0


# ================= CORE CALCULATION LOGIC =================
def calculate_complexity_weight(y1, y2, y3, y4):
    weights = np.full_like(y1, WEIGHT_COMPLEX, dtype=np.float32)

    # --- Step 1: Base Rotation Rules ---
    stack = np.stack([y1, y2, y3, y4], axis=0)
    sorted_stack = np.sort(stack, axis=0)
    diffs = np.diff(sorted_stack, axis=0)
    unique_counts = np.sum(diffs != 0, axis=0) + 1

    mask_1 = (unique_counts == 1)
    weights[mask_1] = WEIGHT_SIMPLE

    mask_2 = (unique_counts == 2)
    is_abab = (y1 == y3) & (y2 == y4)
    is_aabb = (y1 == y2) & (y3 == y4)

    weights[mask_2 & is_abab] = WEIGHT_ABAB
    weights[mask_2 & is_aabb] = WEIGHT_AABB
    weights[mask_2 & (~is_abab) & (~is_aabb)] = WEIGHT_2_UNORDERED

    # --- Step 2: Agroforestry Priority Rules ---
    is_Agro_1 = (y1 == Agroforestry_CLASS_ID)
    is_Agro_2 = (y2 == Agroforestry_CLASS_ID)
    is_Agro_3 = (y3 == Agroforestry_CLASS_ID)
    is_Agro_4 = (y4 == Agroforestry_CLASS_ID)

    Agroforestry_cnt = (is_Agro_1.astype(np.int8) + is_Agro_2.astype(np.int8) +
                   is_Agro_3.astype(np.int8) + is_Agro_4.astype(np.int8))

    mask_Agroforestry_priority = (Agroforestry_cnt >= 3) & is_Agro_4
    weights[mask_Agroforestry_priority] = WEIGHT_Agroforestry_FORCE

    # --- Step 3: Handle NoData ---
    mask_nodata = (y1 == 0) | (y2 == 0) | (y3 == 0) | (y4 == 0)
    weights[mask_nodata] = 0

    return weights


# ================= PROCESSING FLOW =================
def generate_weight_map(target_year):
    print(f"\n[Process] Generating interannual weight map for: {target_year}")

    years = [target_year - 4, target_year - 3, target_year - 2, target_year - 1]
    files = [os.path.join(DATA_ROOT, RF_FILE_TEMPLATE.format(year=y)) for y in years]

    for f in files:
        if not os.path.exists(f):
            print(f"Error: Missing file {f}")
            return

    srcs = [rasterio.open(f) for f in files]
    ref_src = srcs[0]
    meta = ref_src.meta.copy()

    meta.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'nodata': 0,
        'compress': 'lzw'
    })

    out_name = f"Interannual_Weight_{target_year}.tif"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    block_width, block_height = 2048, 2048

    with rasterio.open(out_path, 'w', **meta) as dst:
        if not ref_src.is_tiled:
            windows = []
            for col_off in range(0, ref_src.width, block_width):
                for row_off in range(0, ref_src.height, block_height):
                    width = min(block_width, ref_src.width - col_off)
                    height = min(block_height, ref_src.height - row_off)
                    windows.append(Window(col_off, row_off, width, height))
        else:
            windows = [window for ij, window in ref_src.block_windows(1)]

        print(f"Processing {len(windows)} blocks...")
        for win in tqdm(windows):
            try:
                d1 = srcs[0].read(1, window=win)
                d2 = srcs[1].read(1, window=win)
                d3 = srcs[2].read(1, window=win)
                d4 = srcs[3].read(1, window=win)

                w_map = calculate_complexity_weight(d1, d2, d3, d4)
                dst.write(w_map, 1, window=win)
            except Exception as e:
                print(f"Error processing block: {e}")

    for src in srcs:
        src.close()
    print(f"Success: {out_path}")


if __name__ == "__main__":
    generate_weight_map(2023)
    generate_weight_map(2024)