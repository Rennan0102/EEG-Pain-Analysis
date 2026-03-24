import mne
import matplotlib.pyplot as plt
import numpy as np

# ---------- settings ----------
FILE = r"C:\Users\Renan\Pictures\Universidade\8periodo\EEG-Pain-Analysis\data\ID11.gdf"   # change to any ID*.gdf
PICK = ["F3","F4","C3","C4","P3","P4","O1","O2"]      # channels to show
SPACING_UV = 120                                      # vertical spacing in µV
WIN_SEC = 10                                          # seconds to display per panel
# --------------------------------

raw = mne.io.read_raw_gdf(FILE, preload=True)
raw.filter(1.0, 40.0)           # band-pass
try:
    raw.notch_filter(60.0)      # remove mains (US); ignore if fails
except Exception:
    pass

# Start after 5s per dataset note
if raw.times[-1] > 10:
    raw.crop(tmin=5.0)

# Try to split EO/EC from annotations; otherwise assume 0–300s = EO, 300–600s = EC
EO, EC = None, None
events, event_id = mne.events_from_annotations(raw)
labels = {v: k for k, v in event_id.items()}
if len(events) and any(("EO" in labels[e[2]] or "EC" in labels[e[2]]) for e in events):
    sfreq = raw.info["sfreq"]
    eo_on = [e[0]/sfreq for e in events if "EO" in labels[e[2]]]
    ec_on = [e[0]/sfreq for e in events if "EC" in labels[e[2]]]
    if eo_on: EO = raw.copy().crop(eo_on[0], min(eo_on[0]+300, raw.times[-1]))
    if ec_on: EC = raw.copy().crop(ec_on[0], min(ec_on[0]+300, raw.times[-1]))
else:
    T = raw.times[-1]
    EO = raw.copy().crop(0, min(300, T))
    if T > 300:
        EC = raw.copy().crop(300, min(600, T))

def stack_plot(r, title):
    # pick desired channels; fall back to first 8 if names missing
    picks = [ch for ch in PICK if ch in r.ch_names] or r.ch_names[:8]
    data, times = r.copy().pick(picks).get_data(return_times=True)  # volts
    data_uv = data * 1e6  # to µV

    # center each channel (demean) so it wiggles around 0
    data_uv = data_uv - data_uv.mean(axis=1, keepdims=True)

    # take a window to make it compact
    n_samp = int(WIN_SEC * r.info["sfreq"])
    if data_uv.shape[1] > n_samp:
        data_uv = data_uv[:, :n_samp]
        times = times[:n_samp]

    # build vertical offsets
    offsets = np.arange(len(picks))[::-1] * SPACING_UV

    plt.figure(figsize=(10, 5))
    for i, ch in enumerate(picks):
        plt.plot(times, data_uv[i] + offsets[i], label=ch)  # color auto

    # dashed baselines (optional)
    for off in offsets:
        plt.hlines(off, times[0], times[-1], linestyles="dashed", linewidth=0.5)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.yticks(offsets, picks)  # channel names on the left
    plt.legend(loc="upper right", ncol=1, fontsize=8)
    plt.tight_layout()
    plt.show()

if EO is not None:
    stack_plot(EO, f"{FILE.split('/')[-1]} - Eyes Open (EO)")
if EC is not None:
    stack_plot(EC, f"{FILE.split('/')[-1]} - Eyes Closed (EC)")
