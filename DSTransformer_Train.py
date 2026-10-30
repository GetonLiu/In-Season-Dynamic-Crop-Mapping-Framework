import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================= 1. GLOBAL CONFIGURATION =================
DATA_ROOT = r"D:\Inseason_mapping"
FILE_SUFFIX = "_GlobalNorm"
META_SUFFIX = FILE_SUFFIX  # spatial block split lives in Meta_{year}{META_SUFFIX}.csv
MODEL_SAVE_DIR = r"D:\Inseason_mapping"
STATS_SAVE_DIR = r"D:\Inseason_mapping\Training_Stats"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)

MODE_SWITCH = "ALL"

# [FEATURE SELECTION]
SELECTED_FEATURES = [0, 1, 2, 3, 4, 5, 6]
INPUT_FEATURES = len(SELECTED_FEATURES)
print(f"Feature Configuration: {SELECTED_FEATURES} (Input Dim={INPUT_FEATURES})")

# [MODEL HYPERPARAMETERS]

HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT_RATE = 0.15
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024


# --- TRAINING CONTROL ---
EPOCHS = 100
WARMUP_EPOCHS = 10
NUM_CLASSES = 6
PRINT_INTERVAL = 1

SAMPLES_PER_CLASS_TRAIN = 12000
SAMPLES_PER_CLASS_VAL = 3000
SAMPLES_PER_CLASS_TEST = 3000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

TASKS = {
    2023: [2018, 2019, 2020, 2021, 2022],
    2024: [2019, 2020, 2021, 2022, 2023]
}

CLASS_NAMES = ['Cotton', 'SCorn', 'SWheat', 'DWheat', 'Agroforestry', 'Other']


# ================= 2. MODEL AND LOSS =================
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


# ================= MODEL DEFINITION (PURE TRANSFORMER, ADDED POSITIONAL ENCODING) =================
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


def evaluate_and_stats(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_attn = []

    with torch.no_grad():
        for X_b, y_b in dataloader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            out_steps, out_full, attn_weights = model(X_b)

            loss = criterion(out_full, y_b)
            total_loss += loss.item()
            probs = F.softmax(out_full, dim=1)
            _, pred = torch.max(probs, 1)

            correct += (pred == y_b).sum().item()
            total += y_b.size(0)

            if attn_weights is not None:
                all_attn.append(attn_weights[:, -1, :].cpu())

    acc = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    avg_attn = torch.cat(all_attn, dim=0).mean(dim=0).numpy() if all_attn else np.zeros(18)
    return avg_loss, acc, avg_attn


# ================= 3. SPATIAL BLOCK SPLIT =================
_SPLIT_CACHE = {}


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


def get_block_mask(year, y, block):
    """Boolean mask of the requested spatial block type ('train' or 'val')."""
    if year not in _SPLIT_CACHE:
        meta_path = locate_meta_file(year)
        meta = pd.read_csv(meta_path)

        if len(meta) != len(y):
            raise ValueError(
                f"Row mismatch for {year}: Meta has {len(meta)} rows but y has {len(y)}. "
                f"Metadata and arrays do not come from the same extraction run."
            )
        if 'Label' in meta.columns and not np.array_equal(meta['Label'].to_numpy(), np.asarray(y)):
            raise ValueError(f"Label column in {os.path.basename(meta_path)} does not match y_{year}.")
        if 'split' not in meta.columns:
            raise ValueError(f"'split' column not found in {os.path.basename(meta_path)}.")

        split = meta['split'].astype(str).str.strip().str.lower().to_numpy()
        unknown = set(np.unique(split)) - {'train', 'val'}
        if unknown:
            raise ValueError(f"Unexpected split values in {os.path.basename(meta_path)}: {sorted(unknown)}")

        is_train, is_val = split == 'train', split == 'val'
        if is_train.sum() == 0 or is_val.sum() == 0:
            raise ValueError(f"Empty split for {year}: train={int(is_train.sum())}, val={int(is_val.sum())}")
        _SPLIT_CACHE[year] = (is_train, is_val)

    is_train, is_val = _SPLIT_CACHE[year]
    return is_train if block == 'train' else is_val


def _load_year(yr, block=None):
    xp = os.path.join(DATA_ROOT, f"X_{yr}{FILE_SUFFIX}.npy")
    yp = os.path.join(DATA_ROOT, f"y_{yr}{FILE_SUFFIX}.npy")
    if not (os.path.exists(xp) and os.path.exists(yp)):
        return None, None

    xt, yt = np.load(xp), np.load(yp).astype(np.int64)
    xt = np.nan_to_num(xt, nan=0.0)

    if xt.shape[1] <= 12 and xt.shape[2] > 12:
        if xt.shape[1] > INPUT_FEATURES: xt = xt[:, SELECTED_FEATURES, :]
        xt = xt.transpose(0, 2, 1)
    elif xt.shape[1] > 12 and xt.shape[2] <= 12:
        if xt.shape[2] > INPUT_FEATURES: xt = xt[:, :, SELECTED_FEATURES]

    if xt.shape[1] > 18: xt = xt[:, :18, :]

    if block is not None:
        mask = get_block_mask(yr, yt, block)
        xt, yt = xt[mask], yt[mask]

    return xt, yt


def load_data_list(years, sample_limit=None, seed_base=0, block='train'):
    X_list, y_list = [], []
    for yr in years:
        xt, yt = _load_year(yr, block=block)
        if xt is None:
            print(f"  [WARN] missing arrays for {yr}")
            continue
        X_list.append(xt)
        y_list.append(yt)

    if not X_list: return None, None
    X_pool, y_pool = np.concatenate(X_list), np.concatenate(y_list)

    X_final, y_final = [], []
    for cls_id in range(NUM_CLASSES):
        idx = np.where(y_pool == cls_id)[0]
        if len(idx) == 0: continue
        if sample_limit and len(idx) > sample_limit:
            np.random.seed(seed_base + cls_id)
            idx = np.random.choice(idx, sample_limit, replace=False)
        X_final.append(X_pool[idx])
        y_final.append(y_pool[idx])

    X_out, y_out = np.concatenate(X_final), np.concatenate(y_final)
    p = np.random.permutation(len(y_out))
    return X_out[p], y_out[p]


def load_val_test_split(target_year):
    """Disjoint per-class draws from the target year's validation blocks."""
    xt, yt = _load_year(target_year, block='val')
    if xt is None:
        raise FileNotFoundError(f"Missing arrays for target year {target_year}")

    X_val, y_val, X_test, y_test = [], [], [], []
    for cls_id in range(NUM_CLASSES):
        idx = np.where(yt == cls_id)[0]
        if len(idx) == 0:
            print(f"  [WARN] no validation-block samples for class {cls_id}")
            continue
        np.random.seed(target_year + cls_id + 7)
        np.random.shuffle(idx)

        val_idx = idx[:min(SAMPLES_PER_CLASS_VAL, len(idx))]
        rest = idx[val_idx.size:]
        test_idx = rest[:min(SAMPLES_PER_CLASS_TEST, rest.size)]

        X_val.append(xt[val_idx]); y_val.append(yt[val_idx])
        if test_idx.size > 0:
            X_test.append(xt[test_idx]); y_test.append(yt[test_idx])

    if not X_val or not X_test:
        raise ValueError(f"Validation blocks of {target_year} are too small to build val/test sets")

    X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)
    X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

    p1 = np.random.permutation(len(y_val))
    X_val, y_val = X_val[p1], y_val[p1]
    p2 = np.random.permutation(len(y_test))
    X_test, y_test = X_test[p2], y_test[p2]
    return X_val, y_val, X_test, y_test


def prepare_datasets(target_year, train_years):
    print(f"\nLoading Train {train_years} (training blocks only)...")
    X_pool, y_pool = load_data_list(train_years, block='train')
    if X_pool is None:
        return None, None, None, None, None, None

    X_tr, y_tr = [], []
    for cls_id in range(NUM_CLASSES):
        idx = np.where(y_pool == cls_id)[0]
        if len(idx) == 0: continue
        np.random.seed(target_year + cls_id)
        np.random.shuffle(idx)

        n_train = min(SAMPLES_PER_CLASS_TRAIN, len(idx))
        X_tr.append(X_pool[idx[:n_train]])
        y_tr.append(y_pool[idx[:n_train]])

    X_train, y_train = np.concatenate(X_tr), np.concatenate(y_tr)
    p1 = np.random.permutation(len(y_train))
    X_train, y_train = X_train[p1], y_train[p1]

    print(f"Loading Val/Test {target_year} (validation blocks only)...")
    X_val, y_val, X_test, y_test = load_val_test_split(target_year)

    print(f"  Train {len(y_train)} | Val {len(y_val)} | Test {len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ================= 4. STEP-BY-STEP DYNAMIC EVALUATION FUNCTION =================
def evaluate_step_by_step(model, dataloader, target_year):
    print(f"\nPerforming Step-by-Step Evaluation for year {target_year}...")
    model.eval()

    step_valid_classes = {
        4: [2, 3],
        5: [0, 2, 3],
        6: [0, 2, 3, 4],
    }

    with torch.no_grad():
        for step in range(3, 17):
            valid_classes = step_valid_classes.get(step, [0, 1, 2, 3, 4, 5])
            valid_tensor = torch.tensor(valid_classes, device=DEVICE)

            limit_per_class = {c: 3000 for c in valid_classes}
            if step <= 11:
                if 2 in limit_per_class: limit_per_class[2] = 1500
                if 3 in limit_per_class: limit_per_class[3] = 1500

            class_counts = {c: 0 for c in valid_classes}
            correct, total = 0, 0

            for X_b, y_b in dataloader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

                X_masked = X_b.clone()
                if step < 18:
                    X_masked[:, step:, :] = 0.0

                out_steps, _, _ = model(X_masked)
                current_step_idx = step - 1
                pred_logits = out_steps[:, current_step_idx, :].clone()

                _, pred = torch.max(pred_logits, 1)

                pred_is_valid = torch.isin(pred, valid_tensor)
                pred[~pred_is_valid] = -1

                final_y, final_pred = [], []
                for i in range(len(y_b)):
                    c = y_b[i].item()
                    if c in valid_classes and class_counts[c] < limit_per_class[c]:
                        final_y.append(y_b[i])
                        final_pred.append(pred[i])
                        class_counts[c] += 1

                if len(final_y) == 0:
                    continue

                filtered_y = torch.stack(final_y)
                filtered_pred = torch.stack(final_pred)

                if step <= 11:
                    filtered_y[filtered_y == 3] = 2
                    filtered_pred[filtered_pred == 3] = 2

                correct += (filtered_pred == filtered_y).sum().item()
                total += len(filtered_y)

            acc = 100 * correct / total if total > 0 else 0.0

            display_classes = []
            for c in valid_classes:
                if step <= 11 and c in [2, 3]:
                    if "Wheat_Merged" not in display_classes:
                        display_classes.append("Wheat_Merged")
                else:
                    display_classes.append(CLASS_NAMES[c])

            class_names_str = ",".join(display_classes)
            print(f"Step {step:02d} | [{class_names_str}] | Evaluation Base: {total} | OA: {acc:.2f}%")


# ================= 5. MAIN TRAINING LOGIC =================
def train_final_model(target_year):
    train_years = TASKS[target_year]
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(target_year, train_years)
    if X_train is None: return

    train_ds = CropDataset(X_train, y_train, is_training=True)
    val_ds = CropDataset(X_val, y_val, is_training=False)
    test_ds = CropDataset(X_test, y_test, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TransformerModelDS(INPUT_FEATURES, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT_RATE, nhead=4).to(DEVICE)

    criterion_train = FocalLoss(gamma=2.0, reduction='none').to(DEVICE)
    criterion_val = FocalLoss(gamma=2.0, reduction='mean').to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_acc, best_epoch = 0.0, 0
    best_path = os.path.join(MODEL_SAVE_DIR, f"Final_Model_DSTransformer_{target_year}.pth")

    VALID_EARLY_CLASSES = torch.tensor([0, 1, 2, 4, 5], device=DEVICE)
    training_history = []

    # =========================================================
    # TIME DECAY WEIGHTS (PRE-COMPUTED)
    # =========================================================
    BETA = 0.3
    steps_tensor = torch.arange(0, 14, device=DEVICE).float()
    time_weights = torch.exp(-BETA * steps_tensor)
    time_weights = time_weights / time_weights.sum()


    for epoch in range(EPOCHS):
        is_warmup = epoch < WARMUP_EPOCHS
        train_ds.set_masking(not is_warmup)
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out_steps, out_final, _ = model(X_b)

            step_losses_early = []
            for idx, t in enumerate(range(4, 18)):
                if t < out_steps.shape[1]:
                    pred_t = out_steps[:, t, :]
                    raw_loss_vec = criterion_train(pred_t, y_b)

                    if t < 12:
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
            _, pred = torch.max(out_final, 1)
            correct += (pred == y_b).sum().item()
            total += y_b.size(0)

        scheduler.step()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, avg_attn = evaluate_and_stats(model, val_loader, criterion_val)

        save_msg = ""
        if not is_warmup and val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), best_path)
            save_msg = "BEST"

        if (epoch + 1) % PRINT_INTERVAL == 0 or save_msg:
            status = "WARMUP" if is_warmup else "TRAIN"
            print(
                f"Ep {epoch + 1:03d} [{status}] Loss:{train_loss:.4f} Tr:{train_acc:.1f}% Val:{val_acc:.2f}% {save_msg}")

        epoch_record = {
            'Epoch': epoch + 1, 'Train_Loss': round(train_loss, 4), 'Train_Acc': round(train_acc, 2),
            'Val_Loss': round(val_loss, 4), 'Val_Acc': round(val_acc, 2)
        }
        for i in range(len(avg_attn)): epoch_record[f'Attn_Step_{i + 1}'] = round(float(avg_attn[i]), 4)
        training_history.append(epoch_record)

    print(f"\nDone {target_year}. Best Val Acc: {best_acc:.2f}% (Ep {best_epoch})")
    history_df = pd.DataFrame(training_history)
    history_csv_path = os.path.join(STATS_SAVE_DIR, f"Training_History_DSTransformer_{target_year}.csv")
    history_df.to_csv(history_csv_path, index=False)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        evaluate_step_by_step(model, test_loader, target_year)


if __name__ == "__main__":
    train_final_model(2023)
    train_final_model(2024)