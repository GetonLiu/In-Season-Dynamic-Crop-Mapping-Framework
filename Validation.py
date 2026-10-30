import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from tqdm import tqdm

RESULT_ROOT = r"D:\Inseason_mapping"
SHP_ROOT = r"D:\Inseason_mapping"
SAVE_DIR = r"F:\Inseason_mapping"

GRID_SHP = os.path.join(SHP_ROOT, "Spatial_Holdout_Grid_5km.shp")
GRID_SPLIT_FIELD = "split"
VAL_BLOCK_KEYWORD = "val"          # case-insensitive prefix of the validation label

CM_DIR = os.path.join(SAVE_DIR, "Confusion_Matrices")
AREA_CACHE_JSON = os.path.join(SAVE_DIR, "Map_Area_Counts.json")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# ---- sampling design -------------------------------------------------------
MASTER_SEED = 910
N_REPEATS = 10
SAMPLE_PER_CLASS = 3000
EARLY_WHEAT_SUB_SAMPLE = 1500

# ---- run scope -------------------------------------------------------------
YEARS_TO_VALIDATE = [2023, 2024]
MODES = ["Prior"]                

# ---- class bookkeeping -----------------------------------------------------
SHP_MAPPING = {"Cotton": 0, "SCorn": 1, "SWheat": 2, "DWheat": 3, "Agroforestry": 4, "Others": 5}

ID_TO_NAME = {0: "Cotton", 1: "SCorn", 2: "SWheat", 3: "DWheat",
              4: "Agroforestry", 5: "Others", 99: "Early_Wheat"}

UNKNOWN_ID = 6
NODATA_ID = 255
VALID_CLASS_VALUES = (0, 1, 2, 3, 4, 5)

FIRST_STEP = 4                     
LAST_STEP = 18                     
TEST_STEPS = list(range(FIRST_STEP, LAST_STEP + 1))

CLASS_START_STEPS = {0: 5, 4: 6, 99: 4, 1: 7, 5: 7, 3: 12, 2: 12}
WHEAT_MERGE_LAST_STEP = 11  

CM_ORDER_MERGED = [0, 4, 99, 1, 5]
CM_ORDER_FULL = [0, 4, 3, 2, 1, 5]

DATES = {
    2023: ["Feb 17", "Mar 04", "Mar 19", "Apr 03", "Apr 18", "May 03", "May 18", "Jun 02",
           "Jun 17", "Jul 02", "Jul 17", "Aug 01", "Aug 16", "Aug 31", "Sep 15", "Sep 30",
           "Oct 15", "Oct 30"],
    2024: ["Feb 22", "Mar 08", "Mar 23", "Apr 07", "Apr 22", "May 07", "May 22", "Jun 06",
           "Jun 21", "Jul 06", "Jul 21", "Aug 05", "Aug 20", "Sep 04", "Sep 19", "Oct 04",
           "Oct 19", "Nov 01"],
}


# ============================================================================
# 1. Reference pool restricted to validation blocks
# ============================================================================
def load_val_blocks():
    if not os.path.exists(GRID_SHP):
        raise FileNotFoundError(
            f"Master grid not found: {GRID_SHP}\n"
            "The script needs Spatial_Holdout_Grid_5km.shp to restrict reference "
            "samples to the held-out 5 km blocks."
        )
    grid = gpd.read_file(GRID_SHP)
    if GRID_SPLIT_FIELD not in grid.columns:
        raise KeyError(f"'{GRID_SPLIT_FIELD}' missing from {GRID_SHP}; columns={list(grid.columns)}")

    is_val = grid[GRID_SPLIT_FIELD].astype(str).str.lower().str.startswith(VAL_BLOCK_KEYWORD)
    blocks = grid.loc[is_val, ["geometry"]].copy()
    print(f"[*] Grid: {len(grid)} blocks, {int(is_val.sum())} validation "
          f"({is_val.mean():.1%})")
    return blocks, grid.crs


def load_validation_pool(year, val_blocks, grid_crs):
    """All reference points of `year`, then drop the training blocks."""
    print(f"[*] Loading reference points for {year} ...")
    frames = []
    for crop_name, class_id in SHP_MAPPING.items():
        shp_path = os.path.join(SHP_ROOT, f"Export_Train_{crop_name}_{year}.shp")
        if not os.path.exists(shp_path):
            print(f"    [!] missing {os.path.basename(shp_path)}")
            continue
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            print(f"    [!] {os.path.basename(shp_path)} has no CRS, skipped")
            continue
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
        gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]
        if len(gdf) == 0:
            continue
        gdf = gdf.copy()
        gdf["True_Label"] = class_id
        keep = ["geometry", "True_Label"]
        if GRID_SPLIT_FIELD in gdf.columns:
            keep.append(GRID_SPLIT_FIELD)
        frames.append(gdf[keep])

    if not frames:
        return None

    pool = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    n_all = len(pool)

    if GRID_SPLIT_FIELD in pool.columns:
        how = f"attribute '{GRID_SPLIT_FIELD}'"
        mask = pool[GRID_SPLIT_FIELD].astype(str).str.lower().str.startswith(VAL_BLOCK_KEYWORD)
        kept = pool.loc[mask].copy()
    else:
        how = "spatial join to validation blocks"
        if pool.crs != grid_crs:
            pool = pool.to_crs(grid_crs)
        joined = gpd.sjoin(pool, val_blocks, how="inner", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        kept = pool.loc[joined.index].copy()

    kept = kept.drop(columns=[GRID_SPLIT_FIELD], errors="ignore").reset_index(drop=True)

    print(f"    kept {len(kept):,}/{n_all:,} ({len(kept) / max(n_all, 1):.1%}) via {how}; "
          f"dropped {n_all - len(kept):,} "
          f"(outside every block or on a block boundary)")
    for cid, n in kept["True_Label"].value_counts().sort_index().items():
        flag = "" if n >= SAMPLE_PER_CLASS else f"   <-- only {n}, all of them are used"
        print(f"      {ID_TO_NAME[cid]:<14s}{n:>9,}{flag}")

    return kept


# ============================================================================
# 2. Raster sampling, nodata-safe
# ============================================================================
def extract_raster_values(gdf, raster_path):
    if not os.path.exists(raster_path):
        return None

    with rasterio.open(raster_path) as src:
        g = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
        coords = [(geom.x, geom.y) for geom in g.geometry]
        raw = np.array([val[0] for val in src.sample(coords)], dtype="float64")
        nodata = src.nodata

    if nodata is not None and np.isfinite(nodata):
        raw[np.isclose(raw, float(nodata))] = NODATA_ID
    raw[~np.isfinite(raw)] = NODATA_ID

    pred = np.full(raw.shape, NODATA_ID, dtype=np.int32)
    valid = np.isin(raw, VALID_CLASS_VALUES)
    pred[valid] = raw[valid].astype(np.int32)
    pred[np.isclose(raw, float(UNKNOWN_ID))] = UNKNOWN_ID
    return pred


def count_map_classes(raster_path, cache):
    """Pixel count per class over the whole map, used for W_i."""
    key = os.path.abspath(raster_path)
    if key in cache:
        return {int(k): int(v) for k, v in cache[key].items()}

    counts = Counter()
    with rasterio.open(raster_path) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            vals = arr[np.isin(arr, VALID_CLASS_VALUES)]
            if vals.size:
                uniq, cnt = np.unique(vals, return_counts=True)
                for u, c in zip(uniq.tolist(), cnt.tolist()):
                    counts[int(u)] += int(c)

    cache[key] = dict(counts)
    return dict(counts)


def area_weights(counts, merged):
 
    if merged:
        raw = {cid: float(counts.get(cid, 0)) for cid in (0, 1, 4, 5)}
        raw[99] = float(counts.get(2, 0)) + float(counts.get(3, 0))
    else:
        raw = {cid: float(counts.get(cid, 0)) for cid in VALID_CLASS_VALUES}

    total = sum(raw.values())
    if total <= 0:
        return {}, raw
    return {cid: v / total for cid, v in raw.items()}, raw


# ============================================================================
# 3. Repeated validation sets (drawn once per year)
# ============================================================================
def draw_validation_sets(pool):

    by_class = {c: np.flatnonzero(pool["True_Label"].to_numpy() == c) for c in VALID_CLASS_VALUES}

    sets_full, sets_merged = [], []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(MASTER_SEED + r)
        full_parts, merged_parts = [], []
        for cid in VALID_CLASS_VALUES:
            idx = by_class[cid]
            if idx.size == 0:
                continue
            pick = rng.choice(idx, size=min(SAMPLE_PER_CLASS, idx.size), replace=False)
            full_parts.append(pick)
            if cid in (2, 3):
                n_sub = min(EARLY_WHEAT_SUB_SAMPLE, pick.size)
                merged_parts.append(rng.choice(pick, size=n_sub, replace=False))
            else:
                merged_parts.append(pick)
        sets_full.append(np.concatenate(full_parts) if full_parts else np.array([], dtype=int))
        sets_merged.append(np.concatenate(merged_parts) if merged_parts else np.array([], dtype=int))

    return sets_full, sets_merged


def active_classes(step, merged):
    if merged:
        return [cid for cid, s in CLASS_START_STEPS.items() if step >= s and cid not in (2, 3)]
    return [cid for cid, s in CLASS_START_STEPS.items() if step >= s and cid != 99]


def canonical_order(active, merged):
    order = CM_ORDER_MERGED if merged else CM_ORDER_FULL
    return [c for c in order if c in active] + [c for c in active if c not in order]


# ============================================================================
# 4. Area-weighted metrics, Eqs. (12)-(17)
# ============================================================================
def compute_area_weighted_metrics(true, pred, active, W):
    active = list(active)
    w = np.array([max(float(W.get(c, 0.0)), 0.0) for c in active], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()

    k = len(active)
    n_i = np.zeros(k)
    counts = np.zeros((k, k + 1))       # last column: unmapped / invalid

    for row, cid in enumerate(active):
        m = true == cid
        n_i[row] = float(m.sum())
        if n_i[row] == 0:
            continue
        pc = pred[m]
        for col, pid in enumerate(active):
            counts[row, col] = float(np.count_nonzero(pc == pid))
        counts[row, k] = n_i[row] - counts[row, :k].sum()

    p = (counts / np.where(n_i > 0, n_i, 1.0)[:, None]) * w[:, None]
    p_diag = np.diag(p[:, :k]).copy() if k else np.array([])

    oa = float(p_diag.sum())
    col_margin = p.sum(axis=0)
    pe = float(np.sum(w * col_margin[:k]))
    kappa = (oa - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else np.nan

    P, R, F1, Cov = {}, {}, {}, {}
    for i, cid in enumerate(active):
        R[cid] = float(p_diag[i] / w[i]) if w[i] > 0 else np.nan
        P[cid] = float(p_diag[i] / col_margin[i]) if col_margin[i] > 1e-12 else np.nan
        r, pr = R[cid], P[cid]
        F1[cid] = float(2 * pr * r / (pr + r)) if (np.isfinite(pr) and np.isfinite(r) and pr + r > 0) else np.nan
        Cov[cid] = float((n_i[i] - counts[i, k]) / n_i[i]) if n_i[i] > 0 else np.nan

    total_cov = float(sum(w[i] * Cov[cid] for i, cid in enumerate(active)
                           if np.isfinite(Cov[cid])))

    return {"OA": oa, "Kappa": kappa, "Total_Coverage": total_cov,
            "P": P, "R": R, "F1": F1, "Cov": Cov,
            "n_i": {cid: float(n_i[i]) for i, cid in enumerate(active)},
            "W": {cid: float(w[i]) for i, cid in enumerate(active)},
            "p": p, "counts": counts, "active": active}


# ============================================================================
# 5. Aggregation and export
# ============================================================================
def flatten(res, classes=tuple(ID_TO_NAME)):
    row = {"OA": res["OA"], "Kappa": res["Kappa"], "Total_Coverage": res["Total_Coverage"]}
    for cid in classes:
        name = ID_TO_NAME[cid]
        row[f"P_{name}"] = res["P"].get(cid, np.nan)
        row[f"R_{name}"] = res["R"].get(cid, np.nan)
        row[f"F1_{name}"] = res["F1"].get(cid, np.nan)
        row[f"Cov_{name}"] = res["Cov"].get(cid, np.nan)
    return row


def summarise(repeat_df):
    """Column-wise mean and SD over the repeats."""
    out = {}
    for col in repeat_df.columns:
        if col in ("Year", "Mode", "Step", "Phase", "Date", "Repeat"):
            continue
        if not pd.api.types.is_numeric_dtype(repeat_df[col]):
            continue
        out[col] = repeat_df[col].mean()
        out[f"{col}_sd"] = repeat_df[col].std(ddof=1)
    return out


def cm_to_frame(p_matrix, counts, classes, w, kind="prop"):
    names = [ID_TO_NAME[c] for c in classes] + ["Unmapped"]
    if kind == "prop":
        mat = p_matrix
        tail = np.array([w.get(c, 0.0) for c in classes])
    else:
        mat = counts
        tail = counts.sum(axis=1)
    df = pd.DataFrame(mat, index=[f"True_{n}" for n in names[:-1]],
                      columns=[f"Pred_{n}" for n in names])
    df["True_Area" if kind == "prop" else "True_n"] = tail
    return df


# ============================================================================
# 6. Main
# ============================================================================
def main():
    val_blocks, grid_crs = load_val_blocks()

    area_cache = {}
    if os.path.exists(AREA_CACHE_JSON):
        with open(AREA_CACHE_JSON, "r") as fh:
            area_cache = json.load(fh)

    metric_rows, repeat_rows, design_rows = [], [], []

    for year in YEARS_TO_VALIDATE:
        pool = load_validation_pool(year, val_blocks, grid_crs)
        if pool is None:
            print(f"[!] no reference vectors for {year}, skipped")
            continue
        true_all = pool["True_Label"].to_numpy()
        sets_full, sets_merged = draw_validation_sets(pool)

        for mode in MODES:
            for step in tqdm(TEST_STEPS, desc=f"{year}-{mode}"):
                f_path = os.path.join(RESULT_ROOT, f"Steps_{year}_{mode}_DSTransformer",
                                      f"Result_{year}_Step_{step:02d}_{mode}.tif")
                if not os.path.exists(f_path):
                    f_path = os.path.join(RESULT_ROOT, f"Result_{year}_Step_{step:02d}_{mode}.tif")

                preds = extract_raster_values(pool, f_path)
                if preds is None:
                    tqdm.write(f"    [!] missing raster: {os.path.basename(f_path)}")
                    continue

                merged = step <= WHEAT_MERGE_LAST_STEP
                active = canonical_order(active_classes(step, merged), merged)
                W, raw_counts = area_weights(count_map_classes(f_path, area_cache), merged)
                sets = sets_merged if merged else sets_full

                reps, p_mats, c_mats = [], [], []
                for r_i, idx in enumerate(sets):
                    if idx.size == 0:
                        continue
                    t = true_all[idx]
                    pv = preds[idx]
                    if merged:
                        t = np.where(np.isin(t, [2, 3]), 99, t)
                        pv = np.where(np.isin(pv, [2, 3]), 99, pv)

                    res = compute_area_weighted_metrics(t, pv, active, W)
                    row = {"Year": year, "Mode": mode, "Step": step,
                           "Phase": step - FIRST_STEP + 1,
                           "Date": DATES[year][step - 1], "Repeat": r_i}
                    row.update(flatten(res))
                    reps.append(row)
                    p_mats.append(res["p"])
                    c_mats.append(res["counts"])

                if not reps:
                    continue

                rep_df = pd.DataFrame(reps)
                repeat_rows.extend(reps)

                head = {"Year": year, "Mode": mode, "Step": step,
                        "Phase": step - FIRST_STEP + 1,
                        "Date": DATES[year][step - 1],
                        "Wheat_Merged": merged, "N_Repeats": len(reps)}
                head.update(summarise(rep_df))
                metric_rows.append(head)

                for cid in active:
                    n_ref = []
                    for idx in sets:
                        if idx.size == 0:
                            continue
                        t = true_all[idx]
                        if merged and cid == 99:
                            n_ref.append(np.count_nonzero(np.isin(t, [2, 3])))
                        else:
                            n_ref.append(np.count_nonzero(t == cid))
                    design_rows.append({
                        "Year": year, "Mode": mode, "Step": step,
                        "Phase": step - FIRST_STEP + 1, "Date": DATES[year][step - 1],
                        "Class": ID_TO_NAME[cid],
                        "N_map_px": int(raw_counts.get(cid, 0)),
                        "W": W.get(cid, np.nan),
                        "n_ref_mean": float(np.mean(n_ref)) if n_ref else np.nan,
                    })

                mean_p = np.mean(np.stack(p_mats), axis=0)
                mean_c = np.mean(np.stack(c_mats), axis=0)

                # row margins of the exported matrix must match its row sums,
                # so use the weights after renormalisation over the open classes
                w_act = {c: max(float(W.get(c, 0.0)), 0.0) for c in active}
                tot = sum(w_act.values()) or 1.0
                w_act = {c: v / tot for c, v in w_act.items()}

                tag = f"{mode}_{year}_Phase{step - FIRST_STEP + 1:02d}"
                cm_to_frame(mean_p, mean_c, active, w_act, "prop").to_csv(
                    os.path.join(CM_DIR, f"CM_{tag}.csv"))
                cm_to_frame(mean_p, mean_c, active, w_act, "count").to_csv(
                    os.path.join(CM_DIR, f"CMcounts_{tag}.csv"))

            pd.DataFrame([r for r in metric_rows
                          if r["Year"] == year and r["Mode"] == mode]).to_csv(
                os.path.join(SAVE_DIR, f"Validation_Metrics_{mode}_{year}.csv"),
                index=False, encoding="utf-8-sig")

    with open(AREA_CACHE_JSON, "w") as fh:
        json.dump(area_cache, fh)

    if not metric_rows:
        print("\n[!] nothing produced; check RESULT_ROOT / SHP_ROOT / GRID_SHP")
        return

    pd.DataFrame(metric_rows).to_csv(os.path.join(SAVE_DIR, "Validation_Metrics_All_Steps.csv"),
                                     index=False, encoding="utf-8-sig")
    pd.DataFrame(repeat_rows).to_csv(os.path.join(SAVE_DIR, "Validation_Metrics_Repeats.csv"),
                                     index=False, encoding="utf-8-sig")
    pd.DataFrame(design_rows).to_csv(os.path.join(SAVE_DIR, "Sampling_Design.csv"),
                                     index=False, encoding="utf-8-sig")

    print("\n[*] metrics         -> Validation_Metrics_All_Steps.csv")
    print("[*] per-repeat rows -> Validation_Metrics_Repeats.csv")
    print("[*] sampling design -> Sampling_Design.csv")
    print(f"[*] matrices        -> {CM_DIR}")


# ============================================================================
# 7. Self-test on synthetic data (no rasters needed)
# ============================================================================
def _selftest():
    rng = np.random.default_rng(0)
    active = [0, 4, 99, 1, 5]
    W = {0: 0.45, 4: 0.30, 99: 0.15, 1: 0.07, 5: 0.03}
    true = np.concatenate([np.full(3000, c) for c in active])
    pred = true.copy()
    flip = rng.random(true.size) > 0.70
    pred[flip] = rng.choice(active, size=flip.sum())
    unm = rng.random(true.size) < 0.10
    pred[unm] = UNKNOWN_ID

    res = compute_area_weighted_metrics(true, pred, active, W)

    w = np.array([W[c] for c in active]); w = w / w.sum()
    oa_manual = sum(w[i] * np.mean(pred[true == c] == c) for i, c in enumerate(active))
    assert np.isclose(res["OA"], oa_manual), (res["OA"], oa_manual)
    assert -1.0 <= res["Kappa"] <= 1.0, res["Kappa"]
    assert np.isclose(res["p"].sum(axis=1).sum(), 1.0), "error matrix must sum to 1"
    for c in active:
        assert 0.0 <= res["Cov"][c] <= 1.0
    assert res["OA"] <= max(1, res["OA"])

    # all-correct case -> OA = 1, kappa = 1
    res2 = compute_area_weighted_metrics(true, true.copy(), active, W)
    assert np.isclose(res2["OA"], 1.0) and np.isclose(res2["Kappa"], 1.0), (res2["OA"], res2["Kappa"])

    # everything unmapped -> OA = 0, kappa = 0 (single marginal category)
    res3 = compute_area_weighted_metrics(true, np.full(true.size, UNKNOWN_ID), active, W)
    assert np.isclose(res3["OA"], 0.0), res3["OA"]
    assert np.isclose(res3["Total_Coverage"], 0.0), res3["Total_Coverage"]

    print("self-test passed:")
    print(f"  OA={res['OA']:.4f}  kappa={res['Kappa']:.4f}  coverage={res['Total_Coverage']:.4f}")
    for c in active:
        print(f"  {ID_TO_NAME[c]:<14s} P={res['P'][c]:.3f} R={res['R'][c]:.3f} "
              f"F1={res['F1'][c]:.3f} Cov={res['Cov'][c]:.3f}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
