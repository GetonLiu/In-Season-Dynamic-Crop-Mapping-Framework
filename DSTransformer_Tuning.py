import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import itertools
import time
import torch.nn.functional as F

# ================= CONFIGURATION =================
DATA_ROOT = r"D:\Inseason_mapping"
FILE_SUFFIX = "_GlobalNorm"
META_SUFFIX = FILE_SUFFIX  # spatial block split is stored in Meta_{year}{META_SUFFIX}.csv
MODEL_SAVE_DIR = r"D:\Inseason_mapping\Tuning"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

RESULT_CSV_PATH = os.path.join(MODEL_SAVE_DIR, "Tuning_Results_DSTransformer.csv")

# [FEATURE SELECTION]
SELECTED_FEATURES = [0, 1, 2, 3, 4, 5, 6]
INPUT_FEATURES = len(SELECTED_FEATURES)
print(f"Feature config: Retaining indices {SELECTED_FEATURES} (Input Dim={INPUT_FEATURES})")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

TASKS = [
    {'target': 2023, 'train_years': [2018, 2019, 2020, 2021, 2022]}
]

PARAM_GRID = {
    'hidden_dim': [64, 128, 256],
    'dropout': [0.15, 0.25, 0.35],
    'learning_rate': [1e-4, 3e-4, 5e-4, 8e-4],
    'batch_size': [256, 512, 1024, 2048]
}

NUM_LAYERS = 1
EPOCHS = 100
SAMPLES_PER_CLASS_TRAIN = 12000
SAMPLES_PER_CLASS_VAL = 3000
SAMPLES_PER_CLASS_TEST = 3000
NUM_CLASSES = 6
WARMUP_EPOCHS = 10


# ================= MODEL DEFINITION (PURE TRANSFORMER) =================
class PositionalEncoding(nn.Module):
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


class TransformerModelDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, nhead=4):
        super(TransformerModelDS, self).__init__()
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
        B, T, _ = x.size()
        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        attn_out = self.transformer_encoder(x_pos, mask=causal_mask, is_causal=True)

        normed_out = self.layer_norm(attn_out)
        out_steps = self.fc(self.dropout_layer(normed_out))
        out_final = out_steps[:, -1, :]
        return out_steps, out_final, None


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss


class CropDataset(Dataset):
    def __init__(self, X, y, is_training=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.is_training = is_training
        self.enable_masking = False

    def set_masking(self, status: bool):
        self.enable_masking = status

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.is_training and self.enable_masking:
            if torch.rand(1).item() < 0.5:
                max_step = x.shape[0]
                if max_step > 4:
                    cut_step = torch.randint(4, max_step, (1,)).item()
                    x[cut_step:, :] = 0.0
        return x, y


# ================= DATA LOADING LOGIC =================
ALL_DATA_CACHE = {}


def locate_meta_file(year):
    """Resolve the Meta csv that carries the spatial block split."""
    expected = os.path.join(DATA_ROOT, f"Meta_{year}{META_SUFFIX}.csv")
    if os.path.exists(expected):
        return expected

    candidates = sorted(glob.glob(os.path.join(DATA_ROOT, f"Meta_{year}_*.csv")))
    if len(candidates) == 1:
        print(f"  [WARN] {os.path.basename(expected)} not found, falling back to "
              f"{os.path.basename(candidates[0])}")
        return candidates[0]

    raise FileNotFoundError(
        f"Spatial split metadata missing for {year}. Expected {expected}. "
        f"Without it the spatial block hold-out would be silently discarded."
    )


def load_split_flags(year, y, n_rows):
    """Read the Train/Val block assignment and verify it matches the arrays."""
    meta_path = locate_meta_file(year)
    meta = pd.read_csv(meta_path)

    if len(meta) != n_rows:
        raise ValueError(
            f"Row mismatch for {year}: Meta has {len(meta)} rows but y has {n_rows}. "
            f"Metadata and arrays do not come from the same extraction run."
        )

    if 'Label' in meta.columns:
        if not np.array_equal(meta['Label'].to_numpy(), np.asarray(y)):
            raise ValueError(
                f"Label column in {os.path.basename(meta_path)} does not match y_{year}. "
                f"Unique labels in meta: {np.unique(meta['Label'].to_numpy())}, in y: {np.unique(y)}."
            )

    if 'split' not in meta.columns:
        raise ValueError(f"'split' column not found in {os.path.basename(meta_path)}.")

    split = meta['split'].astype(str).str.strip().str.lower().to_numpy()
    unknown = set(np.unique(split)) - {'train', 'val'}
    if unknown:
        raise ValueError(f"Unexpected split values in {os.path.basename(meta_path)}: {sorted(unknown)}")

    is_train = split == 'train'
    is_val = split == 'val'
    if is_train.sum() == 0 or is_val.sum() == 0:
        raise ValueError(f"Empty split for {year}: train={int(is_train.sum())}, val={int(is_val.sum())}")

    return is_train, is_val


def preload_all_years():
    years_needed = set()
    for t in TASKS:
        years_needed.add(t['target'])
        years_needed.update(t['train_years'])
    print(f"Preloading data ...")
    for year in sorted(years_needed):
        x_p = os.path.join(DATA_ROOT, f"X_{year}{FILE_SUFFIX}.npy")
        y_p = os.path.join(DATA_ROOT, f"y_{year}{FILE_SUFFIX}.npy")
        if not (os.path.exists(x_p) and os.path.exists(y_p)):
            raise FileNotFoundError(f"Missing arrays for {year}: {x_p} / {y_p}")

        X = np.load(x_p)
        y = np.load(y_p).astype(np.int64)
        X = np.nan_to_num(X, nan=0.0)

        if X.shape[1] <= 12 and X.shape[2] > 12:
            if X.shape[1] > INPUT_FEATURES: X = X[:, SELECTED_FEATURES, :]
            X = X.transpose(0, 2, 1)
        elif X.shape[1] > 12 and X.shape[2] <= 12:
            if X.shape[2] > INPUT_FEATURES: X = X[:, :, SELECTED_FEATURES]

        if X.shape[1] > 18: X = X[:, :18, :]

        is_train, is_val = load_split_flags(year, y, len(y))

        ALL_DATA_CACHE[year] = (X, y, is_train, is_val)
        print(f"  Loaded {year}: {X.shape} | train blocks {int(is_train.sum())} px | val blocks {int(is_val.sum())} px")


def perform_stratified_sampling(X, y, samples_per_class, seed_offset=0, mask=None):
    X_res, y_res = [], []
    for cls_id in range(NUM_CLASSES):
        if mask is None:
            indices = np.where(y == cls_id)[0]
        else:
            indices = np.where(mask & (y == cls_id))[0]
        if len(indices) == 0: continue
        current_samples = min(len(indices), samples_per_class)
        if current_samples < samples_per_class:
            print(f"  [WARN] class {cls_id} has only {len(indices)} samples (< {samples_per_class})")
        np.random.seed(seed_offset + cls_id)
        selected_idx = np.random.choice(indices, current_samples, replace=False)
        X_res.append(X[selected_idx])
        y_res.append(y[selected_idx])
    if not X_res: return np.array([]), np.array([])
    return np.concatenate(X_res), np.concatenate(y_res)


def split_val_test_blocks(X_blk, y_blk, target):
    """Per-class disjoint draw from validation blocks: first the reporting set, then the dev set."""
    X_eval, y_eval, X_dev, y_dev = [], [], [], []
    for cls_id in range(NUM_CLASSES):
        indices = np.where(y_blk == cls_id)[0]
        if len(indices) == 0:
            print(f"  [WARN] no validation-block samples for class {cls_id}")
            continue
        np.random.seed(target + cls_id + 42)
        np.random.shuffle(indices)

        eval_idx = indices[:min(SAMPLES_PER_CLASS_TEST, len(indices))]
        rest = indices[eval_idx.size:]
        dev_idx = rest[:min(SAMPLES_PER_CLASS_VAL, rest.size)]

        if eval_idx.size < SAMPLES_PER_CLASS_TEST:
            print(f"  [WARN] class {cls_id} has only {len(indices)} validation-block samples "
                  f"(< {SAMPLES_PER_CLASS_TEST} required by the paper protocol)")

        X_eval.append(X_blk[eval_idx]); y_eval.append(y_blk[eval_idx])
        if dev_idx.size > 0:
            X_dev.append(X_blk[dev_idx]); y_dev.append(y_blk[dev_idx])

    if not X_eval:
        return None, None, None, None
    return np.concatenate(X_eval), np.concatenate(y_eval), np.concatenate(X_dev), np.concatenate(y_dev)


def get_data_pooled_stratified(task_config):
    train_years = task_config['train_years']
    X_pool_list, y_pool_list = [], []
    for yr in train_years:
        if yr in ALL_DATA_CACHE:
            X_y, y_y, is_train, _ = ALL_DATA_CACHE[yr]
            X_pool_list.append(X_y[is_train])   # training blocks only
            y_pool_list.append(y_y[is_train])

    if not X_pool_list: return None, None, None, None, None, None
    X_pool = np.concatenate(X_pool_list)
    y_pool = np.concatenate(y_pool_list)

    X_train, y_train = perform_stratified_sampling(
        X_pool, y_pool, SAMPLES_PER_CLASS_TRAIN, seed_offset=task_config['target'] + 42
    )
    if len(y_train) == 0: return None, None, None, None, None, None

    tgt = task_config['target']
    if tgt in ALL_DATA_CACHE:
        X_tgt, y_tgt, _, is_val = ALL_DATA_CACHE[tgt]
        X_test, y_test, X_val, y_val = split_val_test_blocks(X_tgt[is_val], y_tgt[is_val], tgt)
        if X_val is None or len(y_val) == 0:
            print("  [WARN] no disjoint dev samples left, reusing the evaluation set for early stopping")
            X_val, y_val = X_test, y_test
    else:
        X_test, y_test, X_val, y_val = np.array([]), np.array([]), np.array([]), np.array([])

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_accuracy(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for X_b, y_b in dataloader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            _, out_final, _ = model(X_b)
            _, predicted = torch.max(out_final, 1)
            total += y_b.size(0)
            correct += (predicted == y_b).sum().item()
    return (correct / total * 100) if total > 0 else 0


# ================= TRAINING AND EVALUATION =================
def train_and_eval(params, X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = CropDataset(X_train, y_train, is_training=True)
    val_ds = CropDataset(X_val, y_val, is_training=False)
    test_ds = CropDataset(X_test, y_test, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=0)

    model = TransformerModelDS(INPUT_FEATURES, params['hidden_dim'], NUM_CLASSES,
                               NUM_LAYERS, params['dropout'], nhead=4).to(DEVICE)

    criterion_train = FocalLoss(gamma=2.0, reduction='none').to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)

    best_val_acc = 0.0
    patience, no_improve_epoch = 6, 0
    best_weights = None
    final_train_loss = 0.0

    VALID_EARLY_CLASSES = torch.tensor([0, 1, 2, 4, 5], device=DEVICE)

    BETA = 0.3
    steps_tensor = torch.arange(0, 14, device=DEVICE).float()
    time_weights = torch.exp(-BETA * steps_tensor)
    time_weights = time_weights / time_weights.sum()

    for epoch in range(EPOCHS):
        is_warmup = epoch < WARMUP_EPOCHS
        train_ds.set_masking(not is_warmup)
        model.train()
        running_loss = 0.0
        num_batches = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out_steps, out_final, _ = model(X_b)

            step_losses_early = []
            for idx, t in enumerate(range(3, 17)):
                if t >= out_steps.shape[1]: break
                pred_t = out_steps[:, t, :]
                raw_loss_vec = criterion_train(pred_t, y_b)

                if t < 11:
                    mask = torch.isin(y_b, VALID_EARLY_CLASSES)
                    step_loss = (raw_loss_vec * mask.float()).sum() / (
                                mask.sum() + 1e-6) if mask.sum() > 0 else torch.tensor(0.0, device=DEVICE)
                else:
                    step_loss = torch.mean(raw_loss_vec)

                weighted_step_loss = step_loss * time_weights[idx]
                step_losses_early.append(weighted_step_loss)

            weighted_early_loss = torch.stack(step_losses_early).sum() if step_losses_early else torch.tensor(0.0,
                                                                                                              device=DEVICE)
            loss_last_step = criterion_train(out_final, y_b).mean()

            total_loss = 0.9 * weighted_early_loss + 0.1 * loss_last_step
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            num_batches += 1

        final_train_loss = running_loss / num_batches if num_batches > 0 else 0
        val_acc = evaluate_accuracy(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epoch = 0
            # state_dict() holds references to the live parameters, which optimizer.step()
            # mutates in place; a detached clone is required for the snapshot to persist.
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            if not is_warmup: no_improve_epoch += 1

        if no_improve_epoch >= patience: break

    if best_weights: model.load_state_dict(best_weights)
    test_acc = evaluate_accuracy(model, test_loader)
    return final_train_loss, best_val_acc, test_acc


# ================= GRID SEARCH AND RESUME =================
def load_existing_results():
    if os.path.exists(RESULT_CSV_PATH):
        try:
            df = pd.read_csv(RESULT_CSV_PATH)
            existing_params = []
            for _, row in df.iterrows():
                p = {
                    'hidden_dim': int(row['hidden_dim']),
                    'dropout': float(row['dropout']),
                    'learning_rate': float(row['learning_rate']),
                    'batch_size': int(row['batch_size'])
                }
                existing_params.append(p)
            return existing_params
        except:
            return []
    return []


def is_param_done(params, existing_params):
    for ep in existing_params:
        if (ep['hidden_dim'] == params['hidden_dim'] and
                abs(ep['dropout'] - params['dropout']) < 1e-6 and
                abs(ep['learning_rate'] - params['learning_rate']) < 1e-6 and
                ep['batch_size'] == params['batch_size']):
            return True
    return False


if __name__ == "__main__":
    try:
        preload_all_years()
    except Exception as e:
        print(f"Error: {e}")
        exit()

    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    existing = load_existing_results()
    tasks_to_run = [p for p in param_combinations if not is_param_done(p, existing)]

    total = len(tasks_to_run)
    start_time = time.time()

    for idx, params in enumerate(tasks_to_run):
        print(f"\n[{idx + 1}/{total}] Running Trial: {params}")
        res = params.copy()
        test_scores = []
        valid_run = True

        for task in TASKS:
            X_tr, y_tr, X_val, y_val, X_te, y_te = get_data_pooled_stratified(task)
            if X_tr is None: valid_run = False; break

            if len(y_te) == 0: print(f"  [WARN] Test set is empty for task {task['target']}")
            final_loss, best_val, test_acc = train_and_eval(params, X_tr, y_tr, X_val, y_val, X_te, y_te)

            res[f'Train_Loss_{task["target"]}'] = final_loss
            res[f'Val_Acc_{task["target"]}'] = best_val
            res[f'Test_Acc_{task["target"]}'] = test_acc
            test_scores.append(test_acc)
            print(f"  {task['target']} | Loss: {final_loss:.4f} | Val: {best_val:.2f}% | Test: {test_acc:.2f}%")

        if valid_run:
            final_score = test_scores[0]
            res['Final_Acc'] = final_score

            new_df = pd.DataFrame([res])
            if os.path.exists(RESULT_CSV_PATH):
                existing_df = pd.read_csv(RESULT_CSV_PATH)
                existing_cols = existing_df.columns.tolist()
                new_cols = [c for c in new_df.columns if c not in existing_cols]

                if new_cols:
                    backup_name = RESULT_CSV_PATH.replace(".csv", f"_backup_{int(time.time())}.csv")
                    try:
                        os.rename(RESULT_CSV_PATH, backup_name)
                        new_df.to_csv(RESULT_CSV_PATH, index=False)
                    except OSError:
                        new_df.to_csv(RESULT_CSV_PATH, mode='a', header=False, index=False)
                else:
                    new_df = new_df[existing_cols]
                    new_df.to_csv(RESULT_CSV_PATH, mode='a', header=False, index=False)
            else:
                new_df.to_csv(RESULT_CSV_PATH, mode='w', header=True, index=False)
            elapsed = time.time() - start_time
            print(f"  Result: {final_score:.2f}%. (Elapsed: {elapsed / 60:.1f}m)")
        else:
            print("  Failed.")

    print(f"\nAll Tuning Done! Grid Results Saved to: {RESULT_CSV_PATH}")
