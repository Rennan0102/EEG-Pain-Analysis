import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------
# 0) CONFIG
# ------------------------------------------
DATA_DIR = Path(r"C:\Users\Renan\Pictures\Universidade\8periodo\EEG-Pain-Analysis\Cleaned")

RESULTS_DIR = Path("./Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# match both spellings: *_cleaned_raw.fif and *clean_raw.fif
FILES = sorted(map(str, [
    *DATA_DIR.glob("*_cleaned_raw.fif"),
    *DATA_DIR.glob("*clean_raw.fif"),
]))

if not FILES:
    avail = "\n".join([p.name for p in DATA_DIR.glob("*.fif")])
    raise RuntimeError(f"No cleaned FIFs found in {DATA_DIR}.\nFound:\n{avail}")

print("FILES used:", [Path(f).name for f in FILES])

CODE_NONPAIN = "33024"
CODE_PAIN = "33025"
TMIN, TMAX = 0.0, 2.0
FMIN, FMAX = 0.5, 50.0
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

# ------------------------------------------
# 1) HELPERS
# ------------------------------------------
def _find_event_id(event_id_dict, code):
    for k, v in event_id_dict.items():
        if str(k) == str(code):
            return v
    return None

def bandpower_from_psd(psd, freqs, band_lo, band_hi):
    idx = np.logical_and(freqs >= band_lo, freqs <= band_hi)
    psd_mean = psd.mean(axis=1)
    return np.trapz(psd_mean[:, idx], freqs[idx], axis=1)

def extract_features(epochs):
    psd_obj = epochs.compute_psd(fmin=FMIN, fmax=FMAX)
    psd, freqs = psd_obj.get_data(return_freqs=True)
    total_power = np.trapz(psd.mean(axis=1), freqs, axis=1)
    feats = []
    for bname, (lo, hi) in BANDS.items():
        bp = bandpower_from_psd(psd, freqs, lo, hi)
        feats.append(bp / total_power)
    return np.vstack(feats).T

def make_confusion_and_roc(y_true, y_score, y_pred, title_prefix, model_folder):
    target_dir = RESULTS_DIR / model_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    SHOW = True  # 👈 Add this toggle to control showing vs. saving

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-pain", "Pain"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{title_prefix} — Confusion Matrix")
    fig_cm.savefig(target_dir / "confusion(pain_classify).png", dpi=300)
    if SHOW:
        plt.show()  # 👈 Show confusion matrix
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{title_prefix} — ROC")
    ax_roc.legend()

    fig_roc.savefig(target_dir / "roc(pain_classify).png", dpi=300)
    if SHOW:
        plt.show()  # 👈 Show ROC curve
    plt.close()

# ------------------------------------------
# 2) LOAD DATA
# ------------------------------------------
X_all, y_all, groups = [], [], []
for gi, fname in enumerate(FILES):
    print(f"\nLoading {fname} ...")
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    id_np = _find_event_id(event_id, CODE_NONPAIN)
    id_pn = _find_event_id(event_id, CODE_PAIN)
    if id_np is None or id_pn is None:
        print(f"  Skipping {fname}: missing event codes.")
        continue
    ep_np = mne.Epochs(raw, events, id_np, tmin=TMIN, tmax=TMAX, preload=True, baseline=None, verbose=False)
    ep_pn = mne.Epochs(raw, events, id_pn, tmin=TMIN, tmax=TMAX, preload=True, baseline=None, verbose=False)
    if len(ep_np) == 0 or len(ep_pn) == 0:
        continue
    X_np = extract_features(ep_np)
    X_pn = extract_features(ep_pn)
    y_np = np.zeros(len(X_np), dtype=int)
    y_pn = np.ones(len(X_pn), dtype=int)
    X_all.append(np.vstack([X_np, X_pn]))
    y_all.append(np.concatenate([y_np, y_pn]))
    groups.append(np.full(len(X_np) + len(X_pn), gi, dtype=int))

X = np.vstack(X_all)
y = np.concatenate(y_all)
groups = np.concatenate(groups)
print(f"\nFinal dataset: X={X.shape}, y={y.shape}, subjects={len(FILES)}")

# ------------------------------------------
# 3) MODELS
# ------------------------------------------
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=42))
])

rf_pipe = Pipeline([
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))
])

from sklearn.model_selection import GroupKFold, StratifiedKFold

n_groups = len(np.unique(groups))

if n_groups >= 2:
    # Use GroupKFold when you have ≥2 subjects
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups))
else:
    # Fall back to StratifiedKFold when only one subject is loaded
    min_class = min(np.bincount(y))
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))

def eval_model(name, pipe):
    print(f"\n=== {name} ===")
    scores = cross_val_score(pipe, X, y, cv=splits, scoring="accuracy")
    print(f"CV accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")
    y_pred = cross_val_predict(pipe, X, y, cv=splits, method="predict")
    try:
        y_score = cross_val_predict(pipe, X, y, cv=splits, method="decision_function")
    except Exception:
        y_proba = cross_val_predict(pipe, X, y, cv=splits, method="predict_proba")
        y_score = y_proba[:, 1]
    prefix = name.lower().replace(" ", "_")

    folder_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    make_confusion_and_roc(y, y_score, y_pred, title_prefix=name, model_folder=folder_name)
    print(f"Saved in {RESULTS_DIR / folder_name}")

# ------------------------------------------
# 4) TRAIN & EVALUATE
# ------------------------------------------
eval_model("SVM (RBF)", svm_pipe)
eval_model("Random Forest", rf_pipe)
print("\nDone! Check confusion and ROC PNG files in the folder.")
