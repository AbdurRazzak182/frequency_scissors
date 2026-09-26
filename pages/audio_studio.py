from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import audio_utils, storage, audio_player, logging_utils

st.set_page_config(page_title="Audio Studio", page_icon="🎚️", layout="wide")

# ----------------------------------------------------------------------
# THEME — global look.
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root{
  --c-teal:#2dd4bf; --c-blue:#38bdf8; --c-pink:#f43f5e;
  --c-amber:#f59e0b; --c-purple:#a78bfa;
}

html, body, [class*="css"]{ font-family:'Space Grotesk', sans-serif; }

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 12% 8%, rgba(45,212,191,0.14), transparent 40%),
    radial-gradient(circle at 88% 15%, rgba(167,139,250,0.14), transparent 42%),
    radial-gradient(circle at 50% 100%, rgba(56,189,248,0.10), transparent 55%),
    #0a0f1e;
}
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0d1424 0%, #0a0f1e 100%);
  border-right:1px solid rgba(148,163,184,0.12);
}

/* page spacing */
[data-testid="stAppViewContainer"] .main .block-container{
  padding-top: 1.6rem;
  padding-bottom: 2rem;
}

/* ---- glowing gradient title ---- */
.studio-hero{
  font-family:'Poppins', sans-serif; font-weight:800; font-size:2.4rem;
  background:linear-gradient(90deg, var(--c-teal), var(--c-blue), var(--c-purple), var(--c-pink));
  background-size:300% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:studio-shine 8s linear infinite;
  margin-bottom:0.1rem; letter-spacing:0.5px;
}
@keyframes studio-shine{ to{ background-position:300% center; } }
.studio-sub{ color:#94a3b8; font-size:0.95rem; margin-bottom:1.2rem; }

/* ---- glassy cards for containers ---- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(160deg, rgba(30,41,59,0.55), rgba(15,23,42,0.55));
  border:1px solid rgba(148,163,184,0.15) !important;
  border-radius:18px !important;
  box-shadow:0 8px 30px rgba(0,0,0,0.35);
  backdrop-filter:blur(6px);
}

/* ---- tabs styled as glowing pills ---- */
.stTabs [data-baseweb="tab-list"]{ gap:8px; border-bottom:none; }
.stTabs [data-baseweb="tab"]{
  background:rgba(30,41,59,0.6); border-radius:999px !important;
  padding:8px 20px; border:1px solid rgba(148,163,184,0.18);
  color:#cbd5e1; font-weight:600; transition:all .25s ease;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(90deg, rgba(45,212,191,0.25), rgba(56,189,248,0.25)) !important;
  border:1px solid rgba(56,189,248,0.6) !important;
  box-shadow:0 0 18px rgba(56,189,248,0.35);
  color:#f8fafc !important;
}

/* ---- buttons: gradient, glow on hover ---- */
.stButton>button, .stDownloadButton>button{
  border-radius:12px !important; border:1px solid rgba(148,163,184,0.25) !important;
  background:linear-gradient(135deg, rgba(45,212,191,0.15), rgba(56,189,248,0.15)) !important;
  color:#e2e8f0 !important; font-weight:600 !important; transition:all .2s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  border-color:rgba(56,189,248,0.8) !important;
  box-shadow:0 0 16px rgba(56,189,248,0.45);
  transform:translateY(-1px);
}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg, var(--c-teal), var(--c-blue)) !important;
  color:#04121a !important; border:none !important;
}
.stButton>button[kind="primary"]:hover{ box-shadow:0 0 22px rgba(45,212,191,0.6); }

/* ---- selectbox / slider accents ---- */
[data-testid="stSlider"] [role="slider"]{ background:var(--c-blue) !important; }
[data-baseweb="select"]>div{
  background:rgba(30,41,59,0.6) !important; border-radius:10px !important;
  border:1px solid rgba(148,163,184,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state — VERSION TREE of the working clip. Editing tools act
# ONLY on whichever version is currently selected in the player below —
# pick any version there and the next operation branches off of it.
# Version names accumulate the chain of operations applied, e.g.
#   alarm.mp3(v_0) -> +white noise -> alarm+white_noise.mp3(v_1)
#                   -> +amplify 300-500Hz -> alarm+white_noise+300-500hz_amplified.mp3(v_2)
# and selecting an earlier version (say v_1) and adding pink noise branches
# a new version off of it: alarm+white_noise+pink_noise.mp3(v_3).
# ----------------------------------------------------------------------
defaults = {
    "studio_filename": None,
    "studio_base_name": None,       # original clip name, no extension, e.g. "alarm"
    "studio_ext": "",               # original clip extension, e.g. ".mp3"
    "studio_versions": [],          # list of {label, name, audio, sr, bytes, version_num, parent_label}
    "studio_next_version_num": 0,
    "noise_report": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

NOISE_SUFFIX = {
    "white": "white_noise", "pink": "pink_noise", "hum": "hum_noise",
    "white+hum": "white_hum_noise", "drift": "drift_noise", "burst": "burst_noise",
}
BAND_OP_SUFFIX = {"remove": "removed", "isolate": "isolated", "attenuate": "attenuated", "amplify": "amplified"}


def resample_log_uniform(freqs, magnitude_db, n_points=2048, f_min=20.0):
    f_max = freqs[-1]
    log_freqs = np.logspace(np.log10(f_min), np.log10(max(f_max, f_min * 2)), n_points)
    return log_freqs, np.interp(log_freqs, freqs, magnitude_db)


def _selected():
    """The version currently chosen in the player — this is the ONLY
    version the next editing operation will act on."""
    label = st.session_state.get("studio_version_select")
    for v in st.session_state.studio_versions:
        if v["label"] == label:
            return v
    return st.session_state.studio_versions[-1]


def _version_by_label(label):
    for v in st.session_state.studio_versions:
        if v["label"] == label:
            return v
    return None


def _add_version(suffix: str, audio: np.ndarray, sr: int, action_type: str, details: dict, parent: dict = None):
    parent = parent or _selected()
    num = st.session_state.studio_next_version_num
    ext = st.session_state.studio_ext
    new_name = f"{parent['name']}+{suffix}"
    label = f"{new_name}{ext}(v_{num})"

    entry = {
        "label": label, "name": new_name, "audio": audio, "sr": sr,
        "bytes": audio_utils.samples_to_wav_bytes(audio, sr),
        "version_num": num, "parent_label": parent["label"],
        "action_type": action_type, "action_details": dict(details or {}),
    }
    st.session_state.studio_versions.append(entry)
    st.session_state.studio_next_version_num += 1

    logging_utils.log_action(
        action_type=action_type,
        source_filename=st.session_state.studio_filename,
        sample_rate=sr,
        duration_sec=audio_utils.get_duration(audio, sr),
        details={**details, "based_on": parent["label"]},
    )
    st.session_state["studio_version_select"] = label
    st.session_state.noise_report = None


def _load_clip(path, display_name):
    samples, sr = audio_utils.load_audio(path)
    base_name, ext = Path(display_name).stem, (Path(display_name).suffix or ".wav")
    v0_label = f"{base_name}{ext}(v_0)"

    st.session_state.studio_filename = display_name
    st.session_state.studio_base_name = base_name
    st.session_state.studio_ext = ext
    st.session_state.studio_versions = [{
        "label": v0_label, "name": base_name, "audio": samples, "sr": sr,
        "bytes": audio_utils.samples_to_wav_bytes(samples, sr),
        "version_num": 0, "parent_label": None,
        "action_type": "load", "action_details": {},
    }]
    st.session_state.studio_next_version_num = 1
    st.session_state.noise_report = None
    st.session_state["studio_version_select"] = v0_label
    logging_utils.log_action("load", display_name, sr, audio_utils.get_duration(samples, sr), {})


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _rms_db(samples: np.ndarray) -> float:
    if samples is None or len(samples) == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(np.square(samples.astype(np.float64))) + 1e-12)
    return float(20.0 * np.log10(rms + 1e-12))


def _band_energy_db(samples: np.ndarray, sr: int, low: float, high: float) -> float:
    freqs, mag_db = audio_utils.compute_spectrum(samples, sr)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return float("nan")
    return float(np.mean(mag_db[mask]))


def _overlay_spectrum_fig(before_audio, before_sr, after_audio, after_sr, band=None):
    """Before vs after spectrum, overlaid, with a delta(dB) panel underneath
    so a viewer can SEE exactly which frequencies were cut/boosted/removed."""
    bf, bmag = audio_utils.compute_spectrum(before_audio, before_sr)
    af, amag = audio_utils.compute_spectrum(after_audio, after_sr)
    bf_log, bmag_log = resample_log_uniform(bf, bmag)
    af_log, amag_log = resample_log_uniform(af, amag)
    # common grid so the delta is well defined even if sample rates differ
    common_freqs = bf_log if len(bf_log) <= len(af_log) else af_log
    b_on_common = np.interp(common_freqs, bf_log, bmag_log)
    a_on_common = np.interp(common_freqs, af_log, amag_log)
    delta = a_on_common - b_on_common

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34], vertical_spacing=0.06,
        subplot_titles=("Spectrum — before vs after", "Δ change (after − before, dB)"),
    )
    fig.add_trace(go.Scattergl(x=bf_log, y=bmag_log, mode="lines", name="Before",
                                line=dict(color="#94a3b8", width=1.6, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scattergl(x=af_log, y=amag_log, mode="lines", name="After",
                                line=dict(color="#2dd4bf", width=2.0),
                                fill="tonexty", fillcolor="rgba(45,212,191,0.10)"), row=1, col=1)

    pos = np.where(delta > 0, delta, 0.0)
    neg = np.where(delta < 0, delta, 0.0)
    fig.add_trace(go.Scattergl(x=common_freqs, y=pos, mode="lines", name="Boosted",
                                line=dict(color="#22c55e", width=0.5), fill="tozeroy",
                                fillcolor="rgba(34,197,94,0.45)"), row=2, col=1)
    fig.add_trace(go.Scattergl(x=common_freqs, y=neg, mode="lines", name="Cut / removed",
                                line=dict(color="#f43f5e", width=0.5), fill="tozeroy",
                                fillcolor="rgba(244,63,94,0.45)"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(148,163,184,0.35)", width=1), row=2, col=1)

    if band is not None:
        low, high = band
        for r in (1, 2):
            fig.add_vrect(x0=max(low, 1.0), x1=max(high, 1.0), row=r, col=1,
                          fillcolor="rgba(56,189,248,0.12)", line_width=0)

    fig.update_xaxes(type="log", dtick=1, tickformat="~s", gridcolor="rgba(148,163,184,0.1)", row=2, col=1,
                      title_text="Frequency (Hz)")
    fig.update_xaxes(type="log", dtick=1, tickformat="~s", gridcolor="rgba(148,163,184,0.1)", row=1, col=1)
    fig.update_yaxes(title_text="dB", gridcolor="rgba(148,163,184,0.1)", row=1, col=1)
    fig.update_yaxes(title_text="Δ dB", gridcolor="rgba(148,163,184,0.1)", row=2, col=1)
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=36, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
    )
    return fig


def _overlay_waveform_fig(before_audio, before_sr, after_audio, after_sr):
    """Before vs after waveform envelopes, overlaid on a shared time axis —
    useful for SEEING added bursts/hum or a hiss floor dropping out."""
    bt, bv = audio_utils.downsample_waveform(before_audio, before_sr, max_points=2000)
    at, av = audio_utils.downsample_waveform(after_audio, after_sr, max_points=2000)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=bt, y=bv, mode="lines", name="Before",
                                line=dict(color="#94a3b8", width=1.0)))
    fig.add_trace(go.Scattergl(x=at, y=av, mode="lines", name="After",
                                line=dict(color="#f59e0b", width=1.0)))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", range=[-1.05, 1.05]),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _spectrum_fig(samples, sr, color):
    freqs, mag_db = audio_utils.compute_spectrum(samples, sr)
    freqs, mag_db = resample_log_uniform(freqs, mag_db)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=freqs, y=mag_db, mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Frequency (Hz)", yaxis_title="dB",
        xaxis=dict(type="log", dtick=1, tickformat="~s", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------------------------------------------------
# Sidebar — audio input only
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("🎵 Audio Input")
    previous = storage.list_uploaded_files()
    choice = st.selectbox("Previously uploaded clips", options=["— select —"] + previous, index=0)
    if choice != "— select —" and st.button("Load selected clip", use_container_width=True):
        _load_clip(storage.uploaded_file_path(choice), choice)
        st.rerun()

    st.divider()
    uploaded = st.file_uploader("...or upload new", type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "aiff"])
    if uploaded is not None and st.button("Save & load upload", use_container_width=True):
        saved_path = storage.save_uploaded_file(uploaded)
        _load_clip(saved_path, saved_path.name)
        st.rerun()

    if st.session_state.studio_filename is not None:
        st.divider()
        st.caption(f"Clip: **{st.session_state.studio_filename}**")
        st.caption(f"Versions this session: {len(st.session_state.studio_versions)}")
        if st.button("↺ Start over", use_container_width=True):
            _load_clip(storage.uploaded_file_path(st.session_state.studio_filename), st.session_state.studio_filename)
            st.rerun()

    st.divider()
    st.page_link("pages/history.py", label="📜 Full history", icon="📜")

st.markdown('<div class="studio-hero">🎚️ Audio Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="studio-sub">Edit, denoise, and mix in any order — every version stays one click away.</div>', unsafe_allow_html=True)

if not st.session_state.studio_versions:
    st.info("Upload or choose a clip from the sidebar to open the studio.")
    st.stop()

version_labels = [v["label"] for v in st.session_state.studio_versions]
if st.session_state.get("studio_version_select") not in version_labels:
    st.session_state["studio_version_select"] = version_labels[-1]

# ----------------------------------------------------------------------
# MAIN CONTENT — live spectrum of whatever's selected in the player,
# then the loop-able tool tabs, then the player.
# ----------------------------------------------------------------------
selected = _selected()

with st.container(border=True):
    st.markdown(f"**📊 Spectrum — `{selected['label']}`**")
    st.plotly_chart(_spectrum_fig(selected["audio"], selected["sr"], "#38bdf8"), use_container_width=True, key="studio_top_spectrum")

st.info(
    f"🎯 Editing base: **`{selected['label']}`** — every tool below acts on this version. "
    "Pick a different version in the player under the tools to branch a new edit off of it."
)

tab_edit, tab_add_noise, tab_denoise, tab_mix = st.tabs(
    ["✂️ Frequency Edit", "🔊 Add Noise", "🧹 Detect & Remove Noise", "🎚️ Mix"]
)

# ============================ TAB 1: Frequency Edit ============================
with tab_edit:
    cur = _selected()
    st.caption(f"Editing: `{cur['label']}`")
    samples, sr = cur["audio"], cur["sr"]
    nyquist = sr / 2.0

    freqs, mag_db = audio_utils.compute_spectrum(samples, sr)
    log_freqs, log_mag_db = resample_log_uniform(freqs, mag_db)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=log_freqs, y=log_mag_db, mode="lines", line=dict(color="#2dd4bf", width=1.5)))
    fig.update_layout(
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Frequency (Hz)", yaxis_title="dB",
        xaxis=dict(type="log", dtick=1, tickformat="~s", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        dragmode="select", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    sel_event = st.plotly_chart(fig, use_container_width=True, key="edit_spectrum", on_select="rerun", selection_mode=("box",))

    low_sel, high_sel = None, None
    if sel_event is not None:
        boxes = sel_event.get("selection", {}).get("box", [])
        if boxes:
            x_range = boxes[0].get("x", [])
            if len(x_range) == 2:
                low_sel, high_sel = sorted(float(v) for v in x_range)
                low_sel, high_sel = max(low_sel, 0.0), min(high_sel, nyquist)

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1], vertical_alignment="bottom")
    with c1:
        if low_sel is not None:
            st.caption(f"Band: `{low_sel:,.0f}`–`{high_sel:,.0f}` Hz")
        operation = st.selectbox("Operation", ["Remove band", "Isolate band", "Attenuate band", "Amplify band"], key="e_op", label_visibility="collapsed" if low_sel is not None else "visible")
    op_key = {"Remove band": "remove", "Isolate band": "isolate", "Attenuate band": "attenuate", "Amplify band": "amplify"}[operation]
    gain = 1.0
    with c2:
        if op_key == "attenuate":
            gain = st.slider("Attenuation", 0.0, 1.0, 0.3, 0.05, key="e_atten", label_visibility="collapsed")
        elif op_key == "amplify":
            gain = st.slider("Amplify ×", 1.0, 10.0, 2.0, 0.5, key="e_amp", label_visibility="collapsed")
        else:
            st.caption("No gain needed")
    with c4:
        apply_clicked = st.button("✂️ Apply", use_container_width=True, type="primary", key="e_apply", disabled=low_sel is None)

    if not low_sel:
        st.info("Drag a box on the spectrum above to pick a band.")

    if apply_clicked and low_sel is not None:
        processed = audio_utils.apply_band_operation(samples, sr, low_sel, high_sel, op_key, gain=gain)
        suffix = f"{low_sel:.0f}-{high_sel:.0f}hz_{BAND_OP_SUFFIX[op_key]}"
        _add_version(suffix, processed, sr,
                     "frequency_edit", {"operation": op_key, "low": low_sel, "high": high_sel, "gain": gain},
                     parent=cur)
        st.rerun()

# ============================ TAB 2: Add Noise ============================
with tab_add_noise:
    NOISE_TYPES = {
        "White (hiss)": "white", "Pink": "pink", "Hum": "hum", "White + Hum": "white+hum",
        "Drifting": "drift", "Bursts/clicks": "burst",
    }
    n1, n2, n3, n4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    with n1:
        noise_label = st.selectbox("Noise type", list(NOISE_TYPES.keys()), key="n_type")
    noise_type = NOISE_TYPES[noise_label]
    with n2:
        snr_db = st.slider("SNR (dB)", -10.0, 40.0, 15.0, 1.0, key="n_snr")
    hum_freq, n_bursts, burst_ms = 50.0, 6, 30.0
    with n3:
        if noise_type in ("hum", "white+hum"):
            hum_freq = st.select_slider("Hum Hz", [50.0, 60.0], value=50.0, key="n_hum")
        elif noise_type == "burst":
            n_bursts = st.number_input("Bursts", 1, 30, 6, key="n_nb")
        else:
            st.caption(" ")
    with n4:
        add_clicked = st.button("🔊 Add", use_container_width=True, type="primary", key="n_add")

    cur = _selected()
    st.caption(f"Editing: `{cur['label']}`")

    if add_clicked:
        noisy = audio_utils.add_noise(cur["audio"], cur["sr"], noise_type=noise_type, snr_db=snr_db,
                                       hum_freq=hum_freq, n_bursts=n_bursts, burst_duration_ms=burst_ms)
        _add_version(NOISE_SUFFIX[noise_type], noisy, cur["sr"], "noise_add",
                     {"noise_type": noise_type, "snr_db": snr_db, "hum_freq": hum_freq}, parent=cur)
        st.rerun()

    st.caption("Only needed for testing — adds noise on top of the version selected in the player.")

# ============================ TAB 3: Detect & Remove Noise ============================
with tab_denoise:
    cur = _selected()
    st.caption(f"Editing: `{cur['label']}`")

    method_label = st.radio("Noise estimate", ["Automatic", "Adaptive (drifting)", "Manual region"],
                             horizontal=True, key="d_method")
    profile_method = {"Automatic": "auto", "Adaptive (drifting)": "adaptive", "Manual region": "region"}[method_label]

    noise_start, noise_end = 0.0, None
    if profile_method == "region":
        total_dur = audio_utils.get_duration(cur["audio"], cur["sr"])
        rc1, rc2 = st.columns(2)
        with rc1:
            noise_start = st.number_input("Start (s)", 0.0, total_dur, 0.0, 0.1, key="d_start")
        with rc2:
            noise_end = st.number_input("End (s)", 0.0, total_dur, min(0.5, total_dur), 0.1, key="d_end")

    d1, d2, d3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with d1:
        sensitivity = st.slider("Click sensitivity", 1.5, 6.0, 3.0, 0.5, key="d_sens")
    with d2:
        analyze_clicked = st.button("🔍 Analyze", use_container_width=True, key="d_analyze")

    if analyze_clicked:
        st.session_state.noise_report = audio_utils.analyze_noise(
            cur["audio"], cur["sr"], method=profile_method, noise_start=noise_start,
            noise_end=noise_end, transient_sensitivity=sensitivity,
        )

    if st.session_state.noise_report is not None:
        r = st.session_state.noise_report
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Floor", f"{r['noise_floor_db']:.1f} dB")
        m2.metric("SNR", f"{r['estimated_snr_db']:.1f} dB")
        m3.metric("Hiss/hum", "🔴" if r["is_noisy"] else "🟢")
        m4.metric("Clicks", r["transient_count"])

    rm1, rm2, rm3, rm4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    with rm1:
        strength = st.slider("Strength", 0.0, 3.0, 1.3, 0.1, key="d_strength")
    with rm2:
        floor = st.slider("Floor kept", 0.0, 0.3, 0.02, 0.01, key="d_floor")
    with rm3:
        suppress_transients = st.checkbox("Also suppress clicks", value=True, key="d_suppress")
    with rm4:
        remove_clicked = st.button("🧹 Remove", use_container_width=True, type="primary", key="d_remove")

    if remove_clicked:
        denoised = audio_utils.reduce_noise(
            cur["audio"], cur["sr"], profile_method=profile_method, noise_start=noise_start, noise_end=noise_end,
            reduction_strength=strength, floor=floor, suppress_transients=suppress_transients,
            transient_sensitivity=sensitivity,
        )
        _add_version(f"denoised_{profile_method}", denoised, cur["sr"], "noise_remove",
                     {"profile_method": profile_method, "reduction_strength": strength, "floor": floor,
                      "suppress_transients": suppress_transients}, parent=cur)
        st.rerun()

# ============================ TAB 4: Mix ============================
with tab_mix:
    cur = _selected()
    st.caption(f"Editing: `{cur['label']}`")

    other_files = [f for f in storage.list_uploaded_files() if f != st.session_state.studio_filename]
    if not other_files:
        st.info("No other clips in the workspace to mix in — upload one from the sidebar.")
    else:
        m1, m2, m3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        with m1:
            mix_choice = st.selectbox("Track to mix in", other_files, key="m_choice")
        with m2:
            mix_gain = st.slider("Gain", 0.0, 2.0, 1.0, 0.1, key="m_gain")
        with m3:
            mix_clicked = st.button("🎚️ Mix", use_container_width=True, type="primary", key="m_apply")

        if mix_clicked:
            tmp_path = storage.PROCESSED_DIR / "_studio_working_tmp.wav"
            audio_utils.save_wav(cur["audio"], cur["sr"], tmp_path)
            tracks = [{"path": tmp_path, "gain": 1.0}, {"path": storage.uploaded_file_path(mix_choice), "gain": mix_gain}]
            mixed = audio_utils.mix_audio_files(tracks, target_sr=44100)
            mix_base = Path(mix_choice).stem
            _add_version(f"mix_{mix_base}", mixed, 44100, "mix",
                         {"mixed_with": mix_choice, "gain": mix_gain}, parent=cur)
            st.rerun()

# ----------------------------------------------------------------------
# PLAYER — a normal section placed under the workstation tabs.
# It scrolls with the page, so nothing can hide behind it.
# ----------------------------------------------------------------------
version_labels = [v["label"] for v in st.session_state.studio_versions]
if st.session_state.get("studio_version_select") not in version_labels:
    st.session_state["studio_version_select"] = version_labels[-1]

with st.container(border=True, key="player_panel"):
    bar1, bar3 = st.columns([3, 1.4], vertical_alignment="bottom")
    with bar1:
        st.selectbox("🎧 Now Playing — also the editing base for the tools above",
                     options=version_labels, key="studio_version_select")
    selected = _selected()
    st.caption(f"Selected clip: **`{selected['label']}`**")
    version_tag = f"v_{selected['version_num']}"

    with bar3:
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button("⬇️ Download", data=selected["bytes"],
                                file_name=f"{selected['name']}_{version_tag}.wav",
                                mime="audio/wav", use_container_width=True, key="player_dl")
        with dcol2:
            if st.button("💾 Save", use_container_width=True, key="player_save"):
                target = storage.UPLOADS_DIR / f"{selected['name']}_{version_tag}.wav"
                with open(target, "wb") as f:
                    f.write(selected["bytes"])
                logging_utils.log_action("export", st.session_state.studio_filename, selected["sr"],
                                          audio_utils.get_duration(selected["audio"], selected["sr"]),
                                          {"format": "wav", "saved_as": target.name})
                st.toast(f"Saved {target.name}")

    peaks = audio_utils.compute_peaks(selected["audio"], num_points=300)
    audio_player.render_audio_player(selected["bytes"], selected["label"], peaks, key="studio_player", height=210)

# ----------------------------------------------------------------------
# BEFORE / AFTER COMPARISON — the "proof" box. Shows the version
# currently selected in the player against the version it was built
# from, so a viewer can visually (and audibly) confirm that whatever
# was just done — attenuate/amplify/isolate/remove a band, add a test
# noise, or clean noise out — actually happened.
# ----------------------------------------------------------------------
with st.container(border=True, key="compare_panel"):
    # st.markdown("**🔬 Before / After — Proof of the Edit**")

    parent = _version_by_label(selected.get("parent_label")) if selected.get("parent_label") else None
    action_type = selected.get("action_type")
    action_details = selected.get("action_details", {}) or {}

    if parent is None:
        st.caption(
            "This is the original, unedited clip — apply a frequency edit, add noise, "
            "remove noise, or mix in a track to see a before/after comparison here."
        )
    else:
        action_label = logging_utils.ACTION_LABELS.get(action_type, action_type or "Edit")
        action_desc = logging_utils.describe_action(action_type, action_details)
        st.caption(f"{action_label}: **{action_desc}**")

        # pcol, acol = st.columns(2)
        # with pcol:
        #     st.markdown(f"⬅️ **Before** — `{parent['label']}`")
        #     audio_player.render_audio_player(
        #         parent["bytes"], parent["label"],
        #         audio_utils.compute_peaks(parent["audio"], num_points=200),
        #         key="compare_before_player", height=160,
        #     )
        # with acol:
        #     st.markdown(f"➡️ **After** — `{selected['label']}`")
        #     audio_player.render_audio_player(
        #         selected["bytes"], selected["label"],
        #         audio_utils.compute_peaks(selected["audio"], num_points=200),
        #         key="compare_after_player", height=160,
        #     )

        band = None
        if action_type == "frequency_edit" and action_details.get("low") is not None:
            band = (float(action_details["low"]), float(action_details["high"]))

        cmp_spectrum_tab, cmp_wave_tab = st.tabs(["📊 Spectrum overlay", "🌊 Waveform overlay"])
        with cmp_spectrum_tab:
            st.plotly_chart(
                _overlay_spectrum_fig(parent["audio"], parent["sr"], selected["audio"], selected["sr"], band=band),
                use_container_width=True, key="compare_spectrum_fig",
            )
            if band is not None:
                before_band_db = _band_energy_db(parent["audio"], parent["sr"], *band)
                after_band_db = _band_energy_db(selected["audio"], selected["sr"], *band)
                st.caption(
                    f"Energy in edited band `{band[0]:,.0f}`–`{band[1]:,.0f}` Hz: "
                    f"**{before_band_db:.1f} dB → {after_band_db:.1f} dB** "
                    f"({after_band_db - before_band_db:+.1f} dB)"
                )
            else:
                st.caption(
                    f"Overall level: **{_rms_db(parent['audio']):.1f} dB → {_rms_db(selected['audio']):.1f} dB** "
                    f"({_rms_db(selected['audio']) - _rms_db(parent['audio']):+.1f} dB). "
                    "Green = frequencies that got louder, red = frequencies that got quieter/removed."
                )
        with cmp_wave_tab:
            st.plotly_chart(
                _overlay_waveform_fig(parent["audio"], parent["sr"], selected["audio"], selected["sr"]),
                use_container_width=True, key="compare_waveform_fig",
            )
            st.caption("Gray = before, amber = after — line up bursts, hum ripple, or hiss floor by eye.")