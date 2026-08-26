import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import audio_utils, storage, audio_player

st.set_page_config(page_title="Phase 1 · Frequency Cutting", page_icon="🎛️", layout="wide")
st.title("🎛️ Phase 1 — Frequency Cutting Workspace")

# ----------------------------------------------------------------------
# Session state initialisation 
# ----------------------------------------------------------------------
defaults = {
    "raw_audio": None,      # np.ndarray - the untouched clip, kept in memory
    "sr": None,              # int sample rate
    "filename": None,        # str, name of the currently loaded clip
    "audio_bytes": None,     # bytes of the original file, for playback
    "bands": [],              # list[dict(low, high, gain)] - the scissor cuts
    "processed_audio": None,  # np.ndarray, result of IFFT reconstruction
    "processed_bytes": None,  # bytes, wav-encoded processed audio
    "processed_path": None,   # Path to saved processed file
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_processed():
    st.session_state.processed_audio = None
    st.session_state.processed_bytes = None
    st.session_state.processed_path = None


def _load_clip(path, display_name):
    samples, sr = audio_utils.load_audio(path)
    st.session_state.raw_audio = samples
    st.session_state.sr = sr
    st.session_state.filename = display_name
    st.session_state.audio_bytes = storage.read_bytes(path)
    st.session_state.bands = []
    # _reset_processed()
    # logging_utils.log_entry(
    #     source_filename=display_name,
    #     sample_rate=sr,
    #     duration_sec=audio_utils.get_duration(samples, sr),
    #     bands=[],
    #     processed_filename="",
    # )


# ----------------------------------------------------------------------
# Sidebar — choose or upload a clip
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("🎵 Audio Source")

    previous = storage.list_uploaded_files()
    choice = st.selectbox(
        "Previously uploaded clips",
        options=["— select —"] + previous,
        index=0,
    )
    if choice != "— select —" and st.button("Load selected clip", use_container_width=True):
        _load_clip(storage.uploaded_file_path(choice), choice)
        st.rerun()

    st.divider()

    uploaded = st.file_uploader(
        "...or upload a new audio file",
        type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "aiff"],
    )
    if uploaded is not None and st.button("Save & load upload", use_container_width=True):
        saved_path = storage.save_uploaded_file(uploaded)
        _load_clip(saved_path, saved_path.name)
        st.rerun()

    if st.session_state.raw_audio is not None:
        st.divider()
        st.caption(f"Loaded: **{st.session_state.filename}**")
        st.caption(f"Sample rate: {st.session_state.sr} Hz")
        st.caption(
            f"Duration: {audio_utils.get_duration(st.session_state.raw_audio, st.session_state.sr):.2f} s"
        )

# ----------------------------------------------------------------------
# Waveform + Spectrum boxes (empty state when nothing is loaded)
# ----------------------------------------------------------------------
st.subheader("1. Waveform & Spectrum")


st.markdown("**Waveform**")
waveform_box = st.container(border=True)

st.markdown("**Frequency Spectrum**")
spectrum_box = st.container(border=True)

spectrum_selection_event = None 

def resample_log_uniform(freqs, magnitude_db, n_points=2048, f_min=20.0):
    """
    Resample a linearly-spaced FFT spectrum onto a log-uniform frequency grid.
    Useful for smooth, visually even plots on a log x-axis.

    f_min: lowest frequency to include (avoid 0 Hz, which breaks log scale)
    """
    f_max = freqs[-1]
    log_freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    # Interpolate magnitude (in dB) onto the new log-spaced grid
    magnitude_db_interp = np.interp(log_freqs, freqs, magnitude_db)

    return log_freqs, magnitude_db_interp 

if st.session_state.raw_audio is None:
    with waveform_box:
        st.caption("No audio selected. Upload or choose a clip from the sidebar.")
        st.empty()
    with spectrum_box:
        st.caption("No audio selected. Upload or choose a clip from the sidebar.")
        st.empty()
else:
    samples = st.session_state.raw_audio
    sr = st.session_state.sr

    t, wave_vals = audio_utils.downsample_waveform(samples, sr)
    wave_fig = go.Figure()
    wave_fig.add_trace(go.Scattergl(x=t, y=wave_vals, mode="lines",
                                     line=dict(color="#2dd4bf", width=1)))
    wave_fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        template="plotly_dark",
    )
    with waveform_box:
        st.plotly_chart(wave_fig, use_container_width=True, key="waveform_plot")

    freqs, mag_db = audio_utils.compute_spectrum(samples, sr)
    log_freqs, log_mag_db = resample_log_uniform(freqs, mag_db)
    spec_fig = go.Figure()
    spec_fig.add_trace(go.Scattergl(x=log_freqs, y=log_mag_db, mode="lines",
                                     line=dict(color="#f59e0b", width=1)))
    spec_fig.update_layout(
    height=280, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
    xaxis=dict(
        type="log",
        dtick=1,           # one labeled tick per decade (10, 100, 1000, 10000...)
        tickformat="~s",   # SI-style suffix: 100, 1k, 10k
    ),
    template="plotly_dark",
)
    with spectrum_box:
        st.caption("Tip: drag a box on the spectrum below to pick a frequency band to cut.")
        spectrum_selection_event = st.plotly_chart(
            spec_fig,
            use_container_width=True,
            key="spectrum_plot",
            on_select="rerun",
            selection_mode=("box",),
        )

# ----------------------------------------------------------------------
# 2. Playback of the ORIGINAL audio, with a playhead synced to the waveform
# ----------------------------------------------------------------------
if st.session_state.raw_audio is not None:
    st.subheader("2. Listen to the Original")
    peaks = audio_utils.compute_peaks(st.session_state.raw_audio, num_points=400)
    audio_player.render_audio_player(
        st.session_state.audio_bytes,
        st.session_state.filename,
        peaks,
        key="original",
    )
