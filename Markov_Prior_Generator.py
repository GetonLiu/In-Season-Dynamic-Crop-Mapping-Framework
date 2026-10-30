import os
import numpy as np
import rasterio
import pandas as pd

# ===================== CONFIGURATION: PATHS & PARAMS =====================
DATA_ROOT = r"D:\Inseason_mapping"
RF_DATA_ROOT = r"D:\Inseason_mapping"
OUTPUT_DIR = r"D:\Inseason_mapping"

# Input templates and zone file
RF_FILE_TEMPLATE = "Export_RF_Map_{year}.tif"
ZONE_FILE = os.path.join(DATA_ROOT, "Export_Base_Farmland_Mask.tif")

zone_ids = [1, 2, 3, 4]
zone_names = ['bachu', 'zepu_yecheng', 'maigaiti', 'shache']

S = 6  # Number of classes (1..6)
alpha6 = 0.5  # 6-class Dirichlet/Laplace smoothing
alphaBin = 0.5  # Beta smoothing for alternating events
eps = 1e-12  # Prevent zero division

# 5-year window: predict 'e' using (a,b,c,d)
windows = [
    {'label': '2018_2022', 'years': [2018, 2019, 2020, 2021, 2022]},
    {'label': '2019_2023', 'years': [2019, 2020, 2021, 2022, 2023]}
]


# ===================== UTILITY FUNCTIONS =====================
def decode_ctx4(ctx_key):
    """Decode 4-tuple ctx key (0..1295) -> (a, b, c, d) in range 0..5"""
    a0 = ctx_key // 216
    rem1 = ctx_key % 216
    b0 = rem1 // 36
    rem2 = rem1 % 36
    c0 = rem2 // 6
    d0 = rem2 % 6
    return a0, b0, c0, d0


def partner_ctx_key4(ctx_key):
    """Partner ctx for ABAB <-> BABA (swap a<->b, c<->d)"""
    a0, b0, c0, d0 = decode_ctx4(ctx_key)
    return b0 * 216 + a0 * 36 + d0 * 6 + c0


def is_abab(ctx_key):
    """Check if ctx pattern is ABAB (a==c, b==d, a!=b)"""
    a0, b0, c0, d0 = decode_ctx4(ctx_key)
    return (a0 == c0) and (b0 == d0) and (a0 != b0)


def read_tif(file_path):
    """Read single-band TIF as 2D numpy array"""
    with rasterio.open(file_path) as src:
        return src.read(1)


# ===================== MAIN PROCESSING LOGIC =====================
def main():
    print("Loading zone image...")
    zone_arr = read_tif(ZONE_FILE)

    for win in windows:
        label = win['label']
        years = win['years']
        print(f"\n========== Processing window {label} ==========")

        # 1. Load 5-year RF maps
        print(f"Reading raster data for {years}...")
        img_arrs = []
        for y in years:
            tif_path = os.path.join(RF_DATA_ROOT, RF_FILE_TEMPLATE.format(year=y))
            img_arrs.append(read_tif(tif_path))

        a_arr, b_arr, c_arr, d_arr, e_arr = img_arrs

        # 2. Iterate through each zone
        for zid, z_name in zip(zone_ids, zone_names):
            print(f"  > Calculating for zone: {z_name} (ID: {zid})")

            # Mask valid pixels (1..6) and match zone ID
            valid_mask = (
                    (a_arr > 0) & (b_arr > 0) & (c_arr > 0) & (d_arr > 0) & (e_arr > 0) &
                    (zone_arr == zid)
            )

            # Extract valid pixels, shift to 0..5 (强制转换为 int32 避免 uint8 溢出)
            a_v = a_arr[valid_mask].astype(np.int32) - 1
            b_v = b_arr[valid_mask].astype(np.int32) - 1
            c_v = c_arr[valid_mask].astype(np.int32) - 1
            d_v = d_arr[valid_mask].astype(np.int32) - 1
            e_v = e_arr[valid_mask].astype(np.int32) - 1

            # Vectorized context and pair keys
            ctx_v = a_v * 216 + b_v * 36 + c_v * 6 + d_v
            pair_v = ctx_v * S + e_v

            # Frequency histogram via bincount
            pair_counts = np.bincount(pair_v, minlength=1296 * S)

            rows = []

            # 3. Iterate over all context keys (0..1295)
            for ctx_key in range(1296):
                a0, b0, c0, d0 = decode_ctx4(ctx_key)
                is_abab_flag = is_abab(ctx_key)
                partner_key = partner_ctx_key4(ctx_key)

                # 6-dim count vector for current ctx
                start_idx = ctx_key * S
                n_ctx = pair_counts[start_idx: start_idx + S]
                total_ctx = np.sum(n_ctx)

                # Base smoothed probabilities
                denom = total_ctx + alpha6 * S
                probs_base = (n_ctx + alpha6) / denom

                # ABAB paired alternating logic
                if is_abab_flag:
                    p_start = partner_key * S
                    n_partner = pair_counts[p_start: p_start + S]
                    total_partner = np.sum(n_partner)

                    alt_idx_ctx = a0
                    alt_idx_par = b0

                    cnt_alt_ctx = n_ctx[alt_idx_ctx]
                    cnt_alt_par = n_partner[alt_idx_par]
                    succ_pair = cnt_alt_ctx + cnt_alt_par
                    tot_pair = total_ctx + total_partner

                    p_alt_pair = (succ_pair + alphaBin) / (tot_pair + alphaBin * 2)
                else:
                    alt_idx_ctx = a0
                    alt_idx_par = b0
                    total_partner = 0
                    succ_pair = 0
                    tot_pair = total_ctx
                    p_alt_pair = 0.0

                # Scaling factor for non-alternating classes
                p_base_alt = probs_base[alt_idx_ctx]
                scale_others = (1.0 - p_alt_pair) / (1.0 - p_base_alt + eps)

                # Symmetric probabilities and validation sum
                prob_syms = []
                for s in range(S):
                    if is_abab_flag:
                        psym = p_alt_pair if s == alt_idx_ctx else probs_base[s] * scale_others
                    else:
                        psym = probs_base[s]
                    prob_syms.append(psym)

                prob_sum_ctx = sum(prob_syms)

                # 4. Construct feature rows
                for s in range(S):
                    seq5 = (a0 + 1) * 10000 + (b0 + 1) * 1000 + (c0 + 1) * 100 + (d0 + 1) * 10 + (s + 1)

                    rows.append({
                        'zone_id': int(zid),
                        'zone': z_name,
                        'window': label,
                        'ctx': int(ctx_key),
                        'is_ABA': 1 if is_abab_flag else 0,
                        'partner_ctx': int(partner_key),
                        'a': int(a0 + 1),
                        'b': int(b0 + 1),
                        'c': int(c0 + 1),
                        'd': int(d0 + 1),
                        'next_cls': int(s + 1),
                        'seq5': int(seq5),
                        'count_raw': int(n_ctx[s]),
                        'total_ctx_raw': int(total_ctx),
                        'prob_base': float(probs_base[s]),
                        'alt_idx_ctx': int(alt_idx_ctx),
                        'alt_idx_partner': int(alt_idx_par),
                        'succ_alt_pair': int(succ_pair),
                        'tot_pair': int(tot_pair),
                        'p_alt_pair': float(p_alt_pair),
                        'prob_sym': float(prob_syms[s]),
                        'prob_sum_ctx': float(prob_sum_ctx)
                    })

            # 5. Export to CSV
            df = pd.DataFrame(rows)
            out_name = f"Export_Markov_{z_name}_{label}.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            df.to_csv(out_path, index=False)
            print(f"Exported: {out_name}")


if __name__ == "__main__":
    main()
    print("\n✅ All tasks completed successfully!")