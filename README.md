# ✂️ Frequency Scissors

**An interactive, frequency-domain audio editing workstation — built with Streamlit.**

> Started as a project to "cut" frequency bands out of an audio clip with
> draggable scissors. It has since grown into a small audio workstation:
> frequency editing, synthetic-noise testing, noise removal, multi-track
> mixing, and a versioned edit history with automatic before/after proof —
> all in one app. We kept the original name for continuity; rename freely if
> the team prefers something broader.

---

## ✨ Features

| Area | What it does |
|---|---|
| 🎛️ **Frequency editing** | Visualize a clip's waveform + spectrum, pick a frequency band, and **remove / attenuate / amplify / isolate** it. Reconstructed live with an inverse FFT. |
| 🧪 **Synthetic noise (for testing)** | Inject **stationary** (white / pink / hum / hum+hiss), **drifting** (slowly rising/falling level), or **burst/click** noise onto any clip. |
| 🧹 **Noise detection & removal** | Estimate a noise profile (fixed or adaptive) and clean it out via **STFT spectral subtraction**; detect and repair clicks/pops with a flux + spectral-flatness transient detector. |
| 🎧 **Multi-track mixing** | Load several clips, set per-track gain, auto-resample to a common rate, and mix down with peak normalization. |
| 🕓 **Versioning & proof panel** | Every operation creates a new named version linked to its parent. A **Before / After** panel overlays spectra, shows a dB-delta plot, overlays waveforms, and gives you two mini players — so viewers can see *and* hear that the claimed change really happened. |
| 📜 **Session history** | Every action is logged (CSV-backed) with a human-readable description; filter, browse, and export it from the History page. |

---

## 🖼️ Screenshots

<!-- Add real screenshots here, e.g.:
![Audio Studio — Edit tab](docs/screenshots/edit_tab.png)
![Before/After comparison panel](docs/screenshots/before_after.png)
-->

_Screenshots coming soon — see `docs/screenshots/` (create this folder and
drop your PNGs in, then link them above)._

---

## 🚀 Getting Started

### Requirements
- Python 3.10+
- (Optional) `ffmpeg` on your `PATH` if you want MP3 export/import via `pydub`

### Install & Run
```bash
git clone <your-repo-url>
cd frequency_scissors
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser (default: `http://localhost:8501`).

---

## 🧭 Using the App

1. **Landing page (`app.py`)** — project intro and team info.
2. **Audio Studio** — the main workspace, split into four tabs:
   - **Edit** — pick a frequency band and remove/attenuate/amplify/isolate it.
   - **Add Noise** — layer synthetic stationary/drifting/burst noise on top of
     the currently selected version, for testing the tools below.
   - **Denoise** — estimate a noise profile (auto / adaptive / manual region)
     and remove hiss/hum and/or clicks.
   - **Mix** — combine multiple uploaded tracks into one.

   Every button press creates a new **version** in the sidebar version list.
   Right below the player, the **🔬 Before / After** panel automatically
   compares whatever version is selected against the version it was built
   from — no extra steps needed.
3. **History** — a full, filterable timeline of every action taken this
   session, with a downloadable CSV.

---

## 📁 Project Structure

```
frequency_scissors/
├── app.py                     # Landing page (Streamlit multipage entry point)
├── requirements.txt
├── ABOUT_PROJECT.txt          # Longer-form project background/summary
├── pages/
│   ├── audio_studio.py        # Main workspace: edit / add-noise / denoise / mix + versioning
│   └── history.py             # Session action log, filters, CSV export
├── utils/
│   ├── audio_utils.py         # Core DSP: I/O, spectrum, band filtering, STFT/ISTFT,
│   │                          # noise synthesis, noise-profile estimation,
│   │                          # spectral subtraction, transient detection, mixing
│   ├── audio_player.py        # Custom waveform-aware <audio> player component
│   ├── logging_utils.py       # CSV-backed action log + human-readable descriptions
│   └── storage.py             # Upload / processed / log directory management
├── helper/                    # Earlier standalone prototype pages (kept for reference;
│   │                          # superseded by pages/audio_studio.py)
│   ├── frequency_editing.py
│   ├── noise_reduction.py
│   └── signal_generator.py
└── data/
    ├── uploads/                # User-uploaded clips
    ├── processed/              # Saved processed clips
    └── logs/                   # session_log.csv
```

> `helper/` contains the original, single-purpose pages this project began
> with. `pages/audio_studio.py` is the actively developed, consolidated
> version of that same functionality — check there first.

---

## 🧠 How the Noise Removal Works (short version)

All noise handling shares one Short-Time Fourier Transform (STFT) engine —
frame the signal (default **2048** samples, **512**-sample hop, Hann window),
FFT each frame, modify only the **magnitude**, keep the original **phase**,
then reconstruct with IFFT + overlap-add.

- **Stationary noise** (constant hiss/hum): the quietest ~20% of frames are
  averaged into one fixed noise profile, then subtracted from every frame.
- **Drifting noise** (level rises/falls over time): the same rule is applied
  in a rolling ~40-frame window, so the profile tracks the changing level
  instead of averaging it away.
- **Bursts/clicks**: a separate, finer STFT (512 / 128 samples) flags frames
  that are *both* a sudden spectral-flux spike *and* broadband ("flat") —
  flagged frames get their magnitude replaced with the median of nearby
  clean frames.

_(Exact parameter values — SNR targets, thresholds, frame sizes — live as
named constants/UI defaults in `utils/audio_utils.py` and
`pages/audio_studio.py`; see `ABOUT_PROJECT.txt` for a quick-reference
table.)_

---

## 🛠️ Tech Stack

Streamlit · NumPy · Pandas · SciPy · Plotly · SoundFile
_(pydub, optional, used for MP3 export/import if `ffmpeg` is available)_

---

## 🗺️ Roadmap / Ideas

- [ ] Persist versions/history across sessions (currently in-memory per session)
- [ ] Add real screenshots to this README
- [ ] Expand automated tests around the DSP functions in `utils/audio_utils.py`
- [ ] Retire or merge the legacy pages in `helper/`
- [ ] _(add your team's next milestone here)_

---

## 👋 Team

| Name | ID | Role |
|---|---|---|
| Abdur Razzak | 2305110 | Undergraduate, CSE |
| Dipto Debnath | 2305111 | Undergraduate, CSE |

---

## 📄 License

No license has been chosen yet. If this is a public/course repo, consider
adding one (MIT is a common, permissive default for university projects) —
add a `LICENSE` file and update this section.
