import os
import numpy as np
import mne
from pathlib import Path
from mne.export import export_raw

# ========= SETTINGS (edit these) =========
FILE     = "ID1.gdf"   # change to any ID*.gdf
DATA_DIR = r"C:\Users\Renan\Pictures\Universidade\8periodo\EEG-Pain-Analysis\data"
DUR      = 10.0         # seconds to show in each viewer
SCALE_UV = 150e-6       # trace scale (try 200e-6 or 500e-6 if lines look tiny)
SAVE_EPOCHS = False     # set True if you want 2s epochs saved to disk
# ========================================

# Use matplotlib viewer (stable on macOS)
# mne.viz.set_browser_backend("matplotlib")
mne.viz.set_browser_backend("qt")

# ------------- small helpers -------------
def mark_bad_by_std(r: mne.io.BaseRaw, z: float = 4.0) -> None:
    """Auto-mark EEG channels whose std dev is an outlier (robust z-score)."""
    picks = mne.pick_types(r.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return
    data, _ = r.get_data(picks=picks, return_times=True)
    ch_std = data.std(axis=1)
    med = np.median(ch_std)
    mad = np.median(np.abs(ch_std - med))
    if mad == 0:
        return
    zscore = 0.6745 * (ch_std - med) / mad
    bad_idx = np.where(np.abs(zscore) >= z)[0]
    bads = [r.ch_names[picks[i]] for i in bad_idx]
    if bads:
        r.info["bads"] = sorted(set(r.info.get("bads", [])) | set(bads))
        print(f"Auto-marked bad channels (STD >={z}): {r.info['bads']}")

def apply_ica_blinks(r: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Try ICA blink removal. Falls back to no-ICA if anything fails."""
    try:
        # prefer actual EOG channel if present
        eog_ch = None
        for cand in ("Fp1", "Fp2", "AFz"):
            if cand in r.ch_names:
                eog_ch = cand
                break

        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
        ica.fit(r.copy().filter(1., 40., fir_design="firwin"), picks="eeg")

        if eog_ch is not None:
            inds, scores = ica.find_bads_eog(r, ch_name=eog_ch)
        else:
            inds, scores = ica.find_bads_eog(r)  # template method
        if inds:
            ica.exclude = list(set(inds))
            print(f"ICA: excluding components (blink-related): {ica.exclude}")
            return ica.apply(r.copy())
        else:
            print("ICA found no blink components; returning original filtered signal.")
            return r
    except Exception as e:
        print(f"ICA skipped ({e}). Using band-passed signal only.")
        return r

def find_eo_ec_ranges(raw: mne.io.BaseRaw, dur: float) -> tuple[tuple[float,float]|None, tuple[float,float]|None]:
    """Return (EO_range, EC_range) preferring annotations; else fallback windows."""
    EO = EC = None
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        labels = {v: k for k, v in event_id.items()}
        sf = raw.info["sfreq"]
        for e in events:
            t = e[0] / sf
            label = labels.get(e[2], "")
            if "EO" in label and EO is None:
                EO = (t, min(t + dur, raw.times[-1]))
            if "EC" in label and EC is None:
                EC = (t, min(t + dur, raw.times[-1]))
    except Exception:
        pass

    if EO is None:
        EO = (0.0, min(dur, raw.times[-1]))
    if EC is None and raw.times[-1] > 300:
        EC = (300.0, min(300.0 + dur, raw.times[-1]))
    elif EC is None:
        EC = EO  # last resort
    return EO, EC

def make_clean_copy(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Band-pass 1-40 Hz, 60 Hz notch (+harmonic), auto-mark bads, ICA blinks, avg re-ref."""
    clean = raw.copy()
    clean.load_data()

    # band-pass
    clean.filter(l_freq=1., h_freq=40., fir_design="firwin")

    # remove line noise (and first harmonic if present)
    try:
        clean.notch_filter(freqs=[60, 120], fir_design="firwin")
    except Exception:
        pass

    # auto bad channels + ICA
    mark_bad_by_std(clean, z=4.0)
    clean = apply_ica_blinks(clean)

    # average re-reference at end
    clean.set_eeg_reference("average")
    return clean

browsers = []

def show_pair(raw_full: mne.io.BaseRaw, clean_full: mne.io.BaseRaw, t0: float, t1: float, tag: str):
    """Open two non-blocking viewers (RAW vs CLEAN) for same time span."""
    r_raw   = raw_full.copy().crop(t0, t1)
    r_clean = clean_full.copy().crop(t0, t1)

    b1 = r_raw.plot(title=f"{FILE} - {tag} - RAW", scalings=dict(eeg=SCALE_UV), show=False)
    b2 = r_clean.plot(title=f"{FILE} - {tag} - CLEANED", scalings=dict(eeg=SCALE_UV), show=False)
    
    browsers.extend([b1, b2])

def save_epochs(clean: mne.io.BaseRaw, eo_rng, ec_rng, outdir: str):
    """Save 2-second fixed-length epochs for EO and EC (optional)."""
    os.makedirs(outdir, exist_ok=True)
    for rng, tag in [(eo_rng, "EO"), (ec_rng, "EC")]:
        t0, t1 = rng
        piece = clean.copy().crop(t0, t1)
        epochs = mne.make_fixed_length_epochs(piece, duration=2.0, overlap=0.0, preload=True)
        fname = os.path.join(outdir, f"{os.path.splitext(FILE)[0]}_{tag}_epochs-epo.fif")
        epochs.save(fname, overwrite=True)
        print(f"Saved {tag} epochs -> {fname}  (n_epochs={len(epochs)})")

# ================= main =================
if __name__ == "__main__":
    path = os.path.join(DATA_DIR, FILE)
    print(f"Loading: {path}")
    raw = mne.io.read_raw_gdf(path, preload=True)

    # keep consistent 10–20 names so PSD/topomap stay pretty
    mont = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(mont, match_case=False, on_missing="ignore")

    # initial average ref before PSD/ICA
    raw.set_eeg_reference("average", projection=False)

    # build CLEAN version once
    clean = make_clean_copy(raw)

    # find EO / EC ranges (10 s chunks by default)
    EO_rng, EC_rng = find_eo_ec_ranges(raw, DUR)
    print(f"EO range: {EO_rng}, EC range: {EC_rng}")

    show_pair(raw, clean, *EO_rng, "EO")
    show_pair(raw, clean, *EC_rng, "EC")

    # Exibir todas as janelas criadas
    for b in browsers:
        b.show()

    print("\n[BACKEND QT] Visualizadores abertos.")
    print("Feche as janelas para salvar os dados e encerrar o script.")

    # Iniciar loop do Qt (ESSENCIAL)
    try:
        from mne.viz.backends.qt import _qt_app_exec
        _qt_app_exec()
    except:
        from PyQt5.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        app.exec_()

    # --- Salvar dados (após fechar as janelas) ---
    out_dir = Path(DATA_DIR) / "processed"
    out_dir.mkdir(exist_ok=True)

    fname_base = Path(FILE).stem
    
    clean_fif = out_dir / f"{Path(FILE).stem}_clean_raw.fif"
    clean.save(clean_fif, overwrite=True)

    clean_edf = out_dir / f"{fname_base}_clean_raw.edf"
    export_raw(str(clean_edf), clean, fmt="edf", overwrite=True)
    
    print(f"Arquivos salvos em: {out_dir}")
    # optional epochs for your deliverable
    if SAVE_EPOCHS:
        save_epochs(clean, EO_rng, EC_rng, outdir=str(out_dir / "epochs"))

    print("Done. To run another subject, edit FILE at the top and re-run.")

# pip install edfio