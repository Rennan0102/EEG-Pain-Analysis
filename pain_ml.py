# pain_ml.py
# Labonno: basic ML pipeline for pain vs non-pain on MNE FIF files

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mne
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, cross_val_predict

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = os.path.join(os.getcwd(), "Cleaned")  # folder with ID*_cleaned_raw.fif
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "ID*_cleaned_raw.fif")))
SHOW = True  # set False to only save PNG files
VERBOSE = True

RESULTS_DIR = Path("./Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Event codes (from your earlier runs)
CODE_NONPAIN = 33024
CODE_PAIN    = 33025

# Epoching & PSD
TMIN, TMAX = 0.0, 2.0               # 2-second windows per event
FMIN, FMAX = 0.5, 50.0              # Hz
PSD_METHOD = "multitaper"           # robust default

# Bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 50),
}

# -----------------------------
# UTILITIES
# -----------------------------
def subject_id_from_path(fname: str) -> int:
    """Parse integer subject ID from filename like '.../Cleaned/ID7_cleaned_raw.fif'."""
    base = os.path.basename(fname)          # e.g., "ID7_cleaned_raw.fif"
    first = base.split("_")[0]              # "ID7"
    return int(first.replace("ID", ""))     # 7

def bandpower(psd, freqs, lo, hi):
    """Integrate power between lo..hi Hz (across freqs axis=-1)."""
    idx = (freqs >= lo) & (freqs < hi)
    # integrate per channel then average channels → shape (n_epochs,)
    # psd shape: (n_epochs, n_channels, n_freqs)
    p = np.trapz(psd[:, :, idx], freqs[idx], axis=-1)   # (n_epochs, n_channels)
    return p.mean(axis=1)                                # average channels

def features_from_epochs(epochs: mne.Epochs):
    """Compute PSD and return (X, y) features for this epochs object."""
    # PSD: (n_epochs, n_channels, n_freqs)
    spec = epochs.compute_psd(method=PSD_METHOD, fmin=FMIN, fmax=FMAX, verbose=False)
    psd, freqs = spec.get_data(return_freqs=True)

    # total power for normalization (0.5–50 Hz)
    total = bandpower(psd, freqs, FMIN, FMAX) + 1e-12  # avoid zero

    # per-band relative power
    cols = []
    for name, (lo, hi) in BANDS.items():
        bp = bandpower(psd, freqs, lo, hi) / total
        cols.append(bp[:, None])
    X = np.hstack(cols)  # (n_epochs, 5)
    return X

def make_confusion_and_roc(y_true, y_score, y_pred, title_prefix, model_folder):
    target_dir = RESULTS_DIR / model_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-pain", "Pain"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{title_prefix} — Confusion Matrix")

    fig_cm.savefig(target_dir / "confusion_matrix(pain_ml).png", dpi=300)
    if SHOW:
        plt.show()
    plt.close(fig_cm)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{title_prefix} — ROC")
    ax_roc.legend()

    fig_roc.savefig(target_dir / "roc_curve(pain_ml).png", dpi=300)
    if SHOW:
        plt.show()
    plt.close(fig_roc)

# -----------------------------
# LOAD ALL SUBJECTS → X, y, groups
# -----------------------------
if not FILES:
    raise RuntimeError(f"No *_cleaned_raw.fif files found in {DATA_DIR}")

print(f"FILES used: {[os.path.basename(f) for f in FILES]}\n")
X_all, y_all, groups = [], [], []

for f in FILES:
    if VERBOSE:
        print(f"Loading {os.path.relpath(f)} ...")
    raw = mne.io.read_raw_fif(f, preload=True, verbose=False)

    # events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # find codes present
    id_np = event_id.get(str(CODE_NONPAIN))
    id_pn = event_id.get(str(CODE_PAIN))
    if id_np is None or id_pn is None:
        # Try numeric keys (depending on how they were stored)
        id_np = event_id.get(CODE_NONPAIN, id_np)
        id_pn = event_id.get(CODE_PAIN, id_pn)

    if id_np is None or id_pn is None:
        print(f"  Skipping {os.path.basename(f)}: missing event codes.")
        continue

    # Build epochs per class
    ep_np = mne.Epochs(raw, events, id_np, tmin=TMIN, tmax=TMAX,
                       preload=True, baseline=None, verbose=False)
    ep_pn = mne.Epochs(raw, events, id_pn, tmin=TMIN, tmax=TMAX,
                       preload=True, baseline=None, verbose=False)

    if len(ep_np) == 0 and len(ep_pn) == 0:
        print(f"  Skipping {os.path.basename(f)}: no epochs found.")
        continue

    X_np = features_from_epochs(ep_np) if len(ep_np) else np.empty((0, len(BANDS)))
    X_pn = features_from_epochs(ep_pn) if len(ep_pn) else np.empty((0, len(BANDS)))
    y_np = np.zeros(len(X_np), dtype=int)
    y_pn = np.ones(len(X_pn), dtype=int)

    X = np.vstack([X_np, X_pn])
    y = np.concatenate([y_np, y_pn])

    # subject group label (for GroupKFold)
    sid = subject_id_from_path(f)
    group = np.full(len(y), fill_value=sid, dtype=int)

    X_all.append(X)
    y_all.append(y)
    groups.append(group)

# Final dataset
if not X_all:
    raise RuntimeError("No data accumulated—check event codes and files.")
X = np.vstack(X_all)
y = np.concatenate(y_all)
groups = np.concatenate(groups)
n_groups = len(np.unique(groups))
print(f"\nFinal dataset: X={X.shape}, y={y.shape}, subjects={n_groups}")

# -----------------------------
# MODELS
# -----------------------------
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=2.0, gamma="scale",
                probability=True, class_weight="balanced", random_state=42)),
])

rf_pipe = Pipeline([
    ("rf", RandomForestClassifier(
        n_estimators=300, random_state=42,
        class_weight="balanced_subsample"))
])

# CV strategy: group-wise if ≥2 subjects, else stratified on labels
if n_groups >= 2:
    n_splits = max(2, min(5, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups))
else:
    # fallback: only 1 subject → use label-stratified CV
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    min_class = max(1, min(n_pos, n_neg))
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))

def eval_model(name, pipe):
    print(f"\n=== {name} ===")
    scores = cross_val_score(pipe, X, y, cv=splits, scoring="accuracy")
    print(f"CV accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")

    # Predictions for diagnostics (use same splits)
    try:
        y_score = cross_val_predict(pipe, X, y, cv=splits, method="decision_function")
    except Exception:
        y_proba = cross_val_predict(pipe, X, y, cv=splits, method="predict_proba")
        y_score = y_proba[:, 1]
    y_pred = cross_val_predict(pipe, X, y, cv=splits, method="predict")

    prefix = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    folder_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    make_confusion_and_roc(y, y_score, y_pred, title_prefix=name, model_folder=folder_name)
    print(f"Saved in {RESULTS_DIR / folder_name}")

# -----------------------------
# TRAIN & EVALUATE
# -----------------------------
eval_model("SVM (RBF)", svm_pipe)
eval_model("Random Forest", rf_pipe)

print("\nDone! Check the saved confusion/ROC PNGs in this folder.")
