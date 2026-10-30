import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import ExitStack
import geopandas as gpd
from shapely.geometry import box
import json
import gc
import time
from datetime import timedelta
import rasterio
from rasterio.windows import Window
from rasterio import features

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

# ================= ⚙️ CONFIGURATION =================
BASE_DIR = r"D:\Inseason_mapping"
MODEL_DIR = r"D:\Inseason_mapping"
STATS_JSON_PATH = r"D:\Inseason_mapping\Sample_2018_2024.json"

YEARS_TO_PROCESS = [2023, 2024]

STANDARD_INDICES = ['EVI', 'NDVI', 'NDWI', 'NPCI', 'PPR', 'RVI', 'SAVI']
ALL_INDICES = STANDARD_INDICES
SPECIAL_NORM_INDICES = ['RVI']

ROI_SHP_PATH = r"D:\Inseason_mapping\Export_Base_ROI_Yarkant.shp"
USE_ROI_MASK = True
GENERATE_NO_PRIOR = True
GENERATE_WITH_PRIOR = True

TEST_MODE_SINGLE_BLOCK = False

PRIOR_INJECTION_STEPS = list(range(4, 19))
PRIOR_CONFIG = {
    2023: {
        1: [r"D:\Inseason_mapping\Export_Markov_bachu_2018_2022.csv"],
        2: [r"D:\Inseason_mapping\Export_Markov_zepu_yecheng_2018_2022.csv"],
        3: [r"D:\Inseason_mapping\Export_Markov_maigaiti_2018_2022.csv"],
        4: [r"D:\Inseason_mapping\Export_Markov_shache_2018_2022.csv"]
    },
    2024: {
        1: [r"D:\Inseason_mapping\Export_Markov_bachu_2019_2023.csv"],
        2: [r"D:\Inseason_mapping\Export_Markov_zepu_yecheng_2019_2023.csv"],
        3: [r"D:\Inseason_mapping\Export_Markov_maigaiti_2019_2023.csv"],
        4: [r"D:\Inseason_mapping\Export_Markov_shache_2019_2023.csv"]
    }
}

RF_FILE_TEMPLATE = "Export_RF_Map_{year}.tif"
ZONE_FILE = "Export_Base_Farmland_Mask.tif"
SPATIAL_WEIGHT_TEMPLATE = "Spatial_Weight_{year}.tif"

NUM_CLASSES = 6
AGRO_ID = 4
SCORN_ID = 1
COTTON_ID = 0
SWHEAT_ID = 2
DWHEAT_ID = 3
OTHER_ID = 5

INPUT_DIM, SEQ_LEN = 7, 18
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.15
BLOCK_SIZE = 1024

INFERENCE_BATCH_SIZE = 16384

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} | Batch Size: {INFERENCE_BATCH_SIZE}")

if os.path.exists(STATS_JSON_PATH):
    with open(STATS_JSON_PATH, 'r') as f:
        HYBRID_STATS = json.load(f)
else:
    HYBRID_STATS = {}


# ================= 🧠 MODEL DEFINITION (DS TRANSFORMER) =================
class PositionalEncoding(nn.Module):
    """Standard sine-wave positional encoding"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DSTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, nhead=4):
        super(DSTransformerModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        B, T, _ = x.size()

        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        attn_out = self.transformer_encoder(x_pos, mask=causal_mask, is_causal=True)

        normed_out = self.layer_norm(attn_out)
        out_steps = self.fc(self.dropout_layer(normed_out))
        out_final = out_steps[:, -1, :]
        return out_steps, out_final, None


# ================= 🌟 TRACKER & HELPER FUNCTIONS =================
class CropStateTracker:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.current_class = np.full((height, width), 255, dtype=np.uint8)
        self.current_prob = np.zeros((height, width), dtype=np.float32)
        self.update_time = np.zeros((height, width), dtype=np.uint8)

    def update(self, k, probs, valid_classes):
        working_probs = probs.copy()
        if k < 12:
            if 2 in valid_classes and 3 in valid_classes:
                working_probs[:, :, 2] += working_probs[:, :, 3]
                working_probs[:, :, 3] = 0.0

        mask_invalid_cols = np.ones((self.height, self.width, NUM_CLASSES), dtype=bool)
        effective_valid = list(valid_classes)
        if k < 12 and 3 in effective_valid: effective_valid.remove(3)
        mask_invalid_cols[:, :, effective_valid] = False
        working_probs[mask_invalid_cols] = -1.0

        step_cls = np.argmax(working_probs, axis=2).astype(np.uint8)
        step_prob = np.max(working_probs, axis=2)

        mask_valid_update = (step_prob > 0.5) if k < 12 else (step_prob > -1.0)
        if np.any(mask_valid_update):
            self.current_class[mask_valid_update] = step_cls[mask_valid_update]
            self.current_prob[mask_valid_update] = step_prob[mask_valid_update]
            self.update_time[mask_valid_update] = k


def linear_fusion(tans_probs, prior_probs, time_weight, spatial_weight):
    w_prior = time_weight * spatial_weight
    w_tans = 1.0 - w_prior
    w_tans_expanded = w_tans[..., np.newaxis]
    w_prior_expanded = w_prior[..., np.newaxis]
    fused_unnormalized = (w_tans_expanded * tans_probs) + (w_prior_expanded * prior_probs)
    sum_probs = np.sum(fused_unnormalized, axis=-1, keepdims=True) + 1e-9
    return fused_unnormalized / sum_probs


class PriorManager:
    def __init__(self, year_config):
        self.lookup = {}
        print("📊 Loading Prior CSV tables...")
        for zone_id, file_list in year_config.items():
            for csv_path in file_list:
                if not os.path.exists(csv_path): continue
                try:
                    df = pd.read_csv(csv_path)
                    grouped = df.groupby(['a', 'b', 'c', 'd'])
                    for (a, b, c, d), group in grouped:
                        py_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
                        for _, row in group.iterrows():
                            cls_idx = int(row['next_cls'])
                            prob = float(row['prob_sym'])
                            if cls_idx == 1:
                                py_probs[5] = prob
                            elif cls_idx == 2:
                                py_probs[0] = prob
                            elif cls_idx == 3:
                                py_probs[2] = prob
                            elif cls_idx == 4:
                                py_probs[4] = prob
                            elif cls_idx == 5:
                                py_probs[1] = prob
                            elif cls_idx == 6:
                                py_probs[3] = prob
                        self.lookup[(zone_id, int(a), int(b), int(c), int(d))] = py_probs
                except Exception:
                    pass
        print(f"✅ Prior loaded. Rules: {len(self.lookup)}")

    def get_prior_map(self, zone_arr, hist_arrs):
        h, w = zone_arr.shape
        default_prob = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        stack = np.dstack([zone_arr] + hist_arrs)
        rows = stack.reshape(-1, 5).astype(np.int32)
        unique_rows, inverse_indices = np.unique(rows, axis=0, return_inverse=True)
        unique_probs = np.array([self.lookup.get(tuple(row), default_prob) for row in unique_rows])
        return unique_probs[inverse_indices].reshape(h, w, NUM_CLASSES)


def get_valid_classes(k):
    valid = []
    if k >= 4: valid.extend([2, 3])
    if k >= 5: valid.append(0)
    if k >= 6: valid.append(4)
    if k >= 7: valid.extend([1, 5])
    return valid


def load_and_stack_data(year, window, height, width):
    stacked_data = np.zeros((len(ALL_INDICES), SEQ_LEN, height, width), dtype=np.float32)
    for idx, index_name in enumerate(ALL_INDICES):
        filepath = os.path.join(BASE_DIR, f"EMD_{year}_{index_name}.tif")
        if os.path.exists(filepath):
            with rasterio.open(filepath) as src:
                data = src.read(window=window)
                if data.shape[0] > SEQ_LEN: data = data[:SEQ_LEN]
                data = data.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)

                if index_name in SPECIAL_NORM_INDICES:
                    if index_name in HYBRID_STATS:
                        divisor = HYBRID_STATS[index_name]['divisor']
                        if divisor != 0: data = data / divisor
                    data = np.clip(data, 0.0, 1.0)
                else:
                    data = np.clip(data, -1.0, 1.0)

                if data.shape[0] < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN - data.shape[0], height, width), dtype=np.float32)
                    data = np.concatenate([data, pad], axis=0)
                stacked_data[idx] = data
    stacked_data = np.transpose(stacked_data, (1, 0, 2, 3))
    flat_data = stacked_data.reshape(SEQ_LEN, INPUT_DIM, -1)

    return np.transpose(flat_data, (2, 1, 0))


def load_roi_geometry(tif_crs):
    if not os.path.exists(ROI_SHP_PATH): return None
    gdf = gpd.read_file(ROI_SHP_PATH)
    if gdf.crs != tif_crs: gdf = gdf.to_crs(tif_crs)
    return gdf.union_all()


def get_roi_mask(shape, transform, roi_geom):
    if roi_geom is None: return np.ones(shape, dtype=bool)
    return features.geometry_mask([roi_geom], transform=transform, out_shape=shape, invert=True)


def filter_windows_with_roi(windows, transform, roi_geom):
    if roi_geom is None: return windows
    valid = []
    for win in tqdm(windows, desc="Filtering Windows", unit="blk", dynamic_ncols=True):
        minx, miny, maxx, maxy = rasterio.windows.bounds(win, transform)
        if roi_geom.intersects(box(minx, miny, maxx, maxy)): valid.append(win)
    return valid


def save_checkpoint(ckpt_path, next_idx):
    with open(ckpt_path, 'w') as f:
        f.write(str(next_idx))


# ================= 🚀 MAIN PROCESSING FLOW =================
def process_year(year):
    start_time = time.time()
    print(f"\n🚀 Processing {year} (DS Transformer + Auto Resume)")

    out_dir_no_prior = os.path.join(BASE_DIR, f"Steps_{year}_NoPrior_DSTransformer")
    out_dir_prior = os.path.join(BASE_DIR, f"Steps_{year}_Prior_DSTransformer")
    out_dir_rollback = os.path.join(BASE_DIR, f"Steps_{year}_Rollback_StatsDSTransformer")
    os.makedirs(out_dir_no_prior, exist_ok=True)
    os.makedirs(out_dir_prior, exist_ok=True)
    os.makedirs(out_dir_rollback, exist_ok=True)

    checkpoint_file = os.path.join(BASE_DIR, f"Resume_Checkpoint_{year}.txt")
    resume_block = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                resume_block = int(f.read().strip())
            print(f"🔄 Checkpoint found! Resuming from block {resume_block}...")
        except ValueError:
            print("⚠️ Checkpoint file corrupted, starting from scratch.")
            resume_block = 0

    if GENERATE_WITH_PRIOR:
        year_config = PRIOR_CONFIG.get(year)
        prior_manager = PriorManager(year_config)

    model_name = f"Final_Model_DSTransformer_{year}.pth"
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"⚠️ Model weight not found: {model_path}")
        return

    model_net = DSTransformerModel(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT_RATE, nhead=4).to(device)
    model_net.load_state_dict(torch.load(model_path, map_location=device))
    model_net.eval()

    ref_file = os.path.join(BASE_DIR, f"EMD_{year}_{STANDARD_INDICES[0]}.tif")
    path_zone = os.path.join(BASE_DIR, ZONE_FILE)
    path_sp_w = os.path.join(BASE_DIR, SPATIAL_WEIGHT_TEMPLATE.format(year=year))
    hist_paths = [os.path.join(BASE_DIR, RF_FILE_TEMPLATE.format(year=y)) for y in range(year - 4, year)]

    roi_geom = None
    with rasterio.open(ref_file) as src:
        meta = src.meta.copy()
        transform = src.transform
        if USE_ROI_MASK: roi_geom = load_roi_geometry(src.crs)
        windows = [Window(c, r, min(BLOCK_SIZE, src.width - c), min(BLOCK_SIZE, src.height - r))
                   for c in range(0, src.width, BLOCK_SIZE) for r in range(0, src.height, BLOCK_SIZE)]

    if USE_ROI_MASK and roi_geom: windows = filter_windows_with_roi(windows, transform, roi_geom)

    meta.update(count=2, dtype='float32', compress='lzw', driver='GTiff', BIGTIFF='YES', nodata=255)
    meta_rb = meta.copy()
    meta_rb.update(count=5, nodata=0)

    open_mode = 'r+' if resume_block > 0 else 'w'

    with ExitStack() as stack:
        src_zone = stack.enter_context(rasterio.open(path_zone)) if GENERATE_WITH_PRIOR else None
        src_sp_w = stack.enter_context(rasterio.open(path_sp_w)) if GENERATE_WITH_PRIOR else None
        srcs_hist = [stack.enter_context(rasterio.open(p)) for p in hist_paths] if GENERATE_WITH_PRIOR else []

        files_nop, files_pri, files_rb = {}, {}, {}
        for k in range(4, 19):
            if GENERATE_NO_PRIOR:
                path = os.path.join(out_dir_no_prior, f"Result_{year}_Step_{k:02d}_NoPrior.tif")
                files_nop[k] = stack.enter_context(
                    rasterio.open(path, open_mode if os.path.exists(path) else 'w', **meta))
            if GENERATE_WITH_PRIOR:
                path = os.path.join(out_dir_prior, f"Result_{year}_Step_{k:02d}_Prior.tif")
                files_pri[k] = stack.enter_context(
                    rasterio.open(path, open_mode if os.path.exists(path) else 'w', **meta))
                path_rb = os.path.join(out_dir_rollback, f"RollbackLog_{year}_Step_{k:02d}.tif")
                files_rb[k] = stack.enter_context(
                    rasterio.open(path_rb, open_mode if os.path.exists(path_rb) else 'w', **meta_rb))

        pbar = tqdm(windows, desc=f"Processing {year}", unit="blk", dynamic_ncols=True)

        if resume_block > 0:
            pbar.update(resume_block)

        valid_block_count = 0

        for idx, window in enumerate(pbar):
            if idx < resume_block:
                continue

            win_h, win_w = int(window.height), int(window.width)
            win_transform = rasterio.windows.transform(window, transform)

            pixel_mask = get_roi_mask((win_h, win_w), win_transform, roi_geom) if USE_ROI_MASK else np.ones(
                (win_h, win_w), dtype=bool)

            if not np.any(pixel_mask):
                save_checkpoint(checkpoint_file, idx + 1)
                continue

            pixel_data = load_and_stack_data(year, window, win_h, win_w)
            if np.max(np.abs(pixel_data)) == 0:
                save_checkpoint(checkpoint_file, idx + 1)
                continue

            if GENERATE_WITH_PRIOR:
                zone_data = src_zone.read(1, window=window)
                sp_w_data = src_sp_w.read(1, window=window).astype(np.float32)
                hist_data = [src.read(1, window=window) for src in srcs_hist]
                prior_map = prior_manager.get_prior_map(zone_data, hist_data)

            tracker_nop = CropStateTracker(win_h, win_w)
            tracker_pri = CropStateTracker(win_h, win_w)

            pixel_tensor_cpu = torch.tensor(pixel_data, dtype=torch.float32)
            num_pixels = pixel_tensor_cpu.shape[0]

            for k in range(4, 19):
                curr_input_cpu = pixel_tensor_cpu.clone()
                if k < 18: curr_input_cpu[:, :, k:] = 0.0
                all_logits = torch.zeros((num_pixels, NUM_CLASSES), dtype=torch.float32)
                current_step_idx = min(k - 1, 17)

                with torch.inference_mode():
                    for i in range(0, num_pixels, INFERENCE_BATCH_SIZE):
                        batch_in = curr_input_cpu[i: min(i + INFERENCE_BATCH_SIZE, num_pixels)].to(device,
                                                                                                   non_blocking=True)
                        out_steps, out_final, _ = model_net(batch_in)
                        all_logits[i: i + INFERENCE_BATCH_SIZE] = out_steps[:, current_step_idx, :].cpu()

                probs_tans = F.softmax(all_logits, dim=1).numpy().reshape(win_h, win_w, NUM_CLASSES)
                valid_classes = get_valid_classes(k)

                if GENERATE_NO_PRIOR:
                    tracker_nop.update(k, probs_tans.copy(), valid_classes)
                    out_nop_data = np.stack([tracker_nop.current_class, tracker_nop.current_prob]).astype('float32')
                    files_nop[k].write(out_nop_data, window=window)

                if GENERATE_WITH_PRIOR:
                    if k in PRIOR_INJECTION_STEPS:
                        w_current_linear = max(0.0, 1.0 - ((k - 1) / 17.0))
                        tans_max_conf = np.max(probs_tans, axis=2)
                        w_time_matrix = np.zeros((win_h, win_w), dtype=np.float32)

                        if 4 <= k <= 11:
                            mask_high_conf = tans_max_conf > 0.80
                            w_time_matrix[:] = w_current_linear
                            w_time_matrix[mask_high_conf] = 0.0

                        y1, y2, y3, y4 = hist_data[0], hist_data[1], hist_data[2], hist_data[3]
                        AGRO_cnt = (y1 == AGRO_ID).astype(np.int8) + (y2 == AGRO_ID).astype(np.int8) + \
                                      (y3 == AGRO_ID).astype(np.int8) + (y4 == AGRO_ID).astype(np.int8)
                        mask_stable_AGRO = (AGRO_cnt >= 3) & (y4 == AGRO_ID)

                        w_time_matrix[mask_stable_AGRO] = w_current_linear

                        if k <= 11:
                            mask_zone_active = np.isin(zone_data, [2, 4]) if k == 6 else np.isin(zone_data,
                                                                                                 [1, 2, 3, 4])
                            w_time_matrix[(~mask_zone_active) & (~mask_stable_AGRO)] = 0.0
                        else:
                            w_time_matrix[~mask_stable_AGRO] = 0.0

                        probs_fused = linear_fusion(probs_tans, prior_map, w_time_matrix, sp_w_data)

                        fused_pred_before_rb = np.argmax(probs_fused, axis=2)
                        fused_prob_before_rb = np.max(probs_fused, axis=2)
                        tans_pred = np.argmax(probs_tans, axis=2)

                        mask_force_AGRO = mask_stable_AGRO & (tans_pred != AGRO_ID) & (tans_max_conf < 0.80)
                        if np.any(mask_force_AGRO):
                            probs_fused[mask_force_AGRO, :] = 0.0
                            probs_fused[mask_force_AGRO, AGRO_ID] = 1.0

                        fused_pred = np.argmax(probs_fused, axis=2)
                        diff = np.max(probs_fused, axis=2) - np.max(probs_tans, axis=2)

                        mask_rollback_A = (tans_pred == SCORN_ID) & (fused_pred == COTTON_ID) & (diff <= 0.30)
                        mask_w_swap = ((tans_pred == 2) & (fused_pred == 3)) | ((tans_pred == 3) & (fused_pred == 2))
                        mask_rollback_C = mask_w_swap & (diff <= 0.12)
                        mask_handled = ((tans_pred == SCORN_ID) & (fused_pred == COTTON_ID)) | \
                                       ((tans_pred == OTHER_ID) & ((fused_pred == 2) | (fused_pred == 3))) | \
                                       mask_w_swap | (((tans_pred == 2) | (tans_pred == 3)) & (fused_pred == 4))
                        mask_rollback_E = (tans_pred != fused_pred) & (~mask_handled) & (diff <= 0.12)

                        mask_rollback_all = mask_rollback_A | mask_rollback_C | mask_rollback_E
                        if np.any(mask_rollback_all):
                            probs_fused[mask_rollback_all] = probs_tans[mask_rollback_all]

                        total_rb_mask = mask_rollback_all

                        if np.any(total_rb_mask):
                            out_rb = np.zeros((5, win_h, win_w), dtype=np.float32)
                            out_rb[0, total_rb_mask] = 1.0
                            out_rb[1, total_rb_mask] = fused_pred_before_rb[total_rb_mask]
                            out_rb[2, total_rb_mask] = tans_pred[total_rb_mask]
                            out_rb[3, total_rb_mask] = fused_prob_before_rb[total_rb_mask]
                            out_rb[4, total_rb_mask] = tans_max_conf[total_rb_mask]
                            files_rb[k].write(out_rb, window=window)
                            del out_rb

                        probs_for_update = probs_fused
                    else:
                        probs_for_update = probs_tans.copy()

                    probs_for_update[~pixel_mask] = 0
                    tracker_pri.update(k, probs_for_update, valid_classes)
                    out_pri_data = np.stack([tracker_pri.current_class, tracker_pri.current_prob]).astype('float32')
                    files_pri[k].write(out_pri_data, window=window)

            if idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            save_checkpoint(checkpoint_file, idx + 1)

            valid_block_count += 1
            if TEST_MODE_SINGLE_BLOCK and valid_block_count >= 1:
                print(f"\n🛑 [TEST MODE] Successfully processed 1 valid block (Global Index {idx}), breaking loop!")
                break

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("🗑️ Processing completed for the year, checkpoint file removed.")

    end_time = time.time()
    print(f"\n✅ Finished {year}. Total time: {str(timedelta(seconds=int(end_time - start_time)))}")


if __name__ == "__main__":
    for year in YEARS_TO_PROCESS:
        process_year(year)