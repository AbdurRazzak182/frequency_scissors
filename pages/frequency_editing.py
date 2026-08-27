import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import audio_utils, storage, audio_player 

st.set_page_config(page_title="Phase 1 · Frequency Cutting", page_icon="🎛️", layout="wide")
st.title("🎛️ Wave Visualization & Frequency Cutting Workspace") 

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

col1, col2 = st.columns([3, 5], vertical_alignment="bottom")
with col1:
    st.subheader("1. Waveform & Spectrum")
with col2:
    st.badge(choice, color="blue")


st.markdown("**Waveform**")
waveform_box = st.container(border=True)

st.markdown("**Frequency Spectrum**")
spectrum_box = st.container(border=True)
player_box = st.container(border=True)

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
with player_box:
    if st.session_state.raw_audio is not None:
        st.subheader("2. Listen to the Original")
        peaks = audio_utils.compute_peaks(st.session_state.raw_audio, num_points=400)
        audio_player.render_audio_player(
            st.session_state.audio_bytes,
            st.session_state.filename,
            peaks,
            key="original",
        )



# ----------------------------------------------------------------------
# 3. The Scissors — an editable spectrum you drag a box on, plus the
#    four cutting operations. Applying one runs the IFFT and shows the
#    modified spectrum + a player for the modified audio.
# ----------------------------------------------------------------------
if st.session_state.raw_audio is not None:
    st.subheader("3. Edit the Frequency Spectrum")
    st.caption(
        "Drag a box on the spectrum below to select a frequency band, "
        "choose an operation, then apply it."
    )

    edit_box = st.container(border=True)
    samples = st.session_state.raw_audio
    sr = st.session_state.sr
    nyquist = sr / 2.0

    with edit_box:
        # --- the editable, box-selectable spectrum ---
        edit_freqs, edit_mag_db = audio_utils.compute_spectrum(samples, sr)
        edit_log_freqs, edit_log_mag_db = resample_log_uniform(edit_freqs, edit_mag_db)

        edit_fig = go.Figure()
        edit_fig.add_trace(go.Scattergl(
            x=edit_log_freqs, y=edit_log_mag_db, mode="lines",
            line=dict(color="#38bdf8", width=1),
        ))
        edit_fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
            xaxis=dict(type="log", dtick=1, tickformat="~s"),
            dragmode="select",
            template="plotly_dark",
        )
        edit_selection_event = st.plotly_chart(
            edit_fig,
            use_container_width=True,
            key="edit_spectrum_plot",
            on_select="rerun",
            selection_mode=("box",),
        )

        # --- pull the dragged box's x-range (Hz) out of the select event ---
        low_sel, high_sel = None, None
        if edit_selection_event is not None:
            boxes = edit_selection_event.get("selection", {}).get("box", [])
            if boxes:
                x_range = boxes[0].get("x", [])
                if len(x_range) == 2:
                    low_sel, high_sel = sorted(float(v) for v in x_range)
                    low_sel = max(low_sel, 0.0)
                    high_sel = min(high_sel, nyquist)

        if low_sel is None:
            st.info("No band selected yet — drag a box on the spectrum above.")
        else:
            st.markdown(f"**Selected band:** `{low_sel:,.1f} Hz` → `{high_sel:,.1f} Hz`")

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1], vertical_alignment="bottom")

            with ctrl_col1:
                operation = st.selectbox(
                    "Operation",
                    options=["Remove band", "Isolate band", "Attenuate band", "Amplify band"],
                    key="scissor_operation",
                )

            op_key = {
                "Remove band": "remove",
                "Isolate band": "isolate",
                "Attenuate band": "attenuate",
                "Amplify band": "amplify",
            }[operation]

            gain = 1.0
            with ctrl_col2:
                if op_key == "attenuate":
                    gain = st.slider(
                        "Attenuation (fraction of original amplitude kept)",
                        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                        key="attenuate_gain",
                    )
                elif op_key == "amplify":
                    gain = st.slider(
                        "Amplification factor",
                        min_value=1.0, max_value=10.0, value=2.0, step=0.5,
                        key="amplify_gain",
                    )
                else:
                    st.caption("No gain parameter needed for this operation.")

            with ctrl_col3:
                apply_clicked = st.button("✂️ Apply", use_container_width=True, type="primary")

            if apply_clicked:
                processed = audio_utils.apply_band_operation(
                    samples, sr, low_sel, high_sel, op_key, gain=gain,
                )
                st.session_state.processed_audio = processed
                st.session_state.processed_bytes = audio_utils.samples_to_wav_bytes(processed, sr)
                st.session_state.bands.append(
                    {"low": low_sel, "high": high_sel, "operation": op_key, "gain": gain}
                )
                st.rerun()

    # --- results: modified spectrum + modified audio player ---
    if st.session_state.processed_audio is not None:
        st.markdown("**Modified Frequency Spectrum**")
        result_spectrum_box = st.container(border=True)
        result_player_box = st.container(border=True)

        with result_spectrum_box:
            proc_freqs, proc_mag_db = audio_utils.compute_spectrum(
                st.session_state.processed_audio, sr
            )
            proc_log_freqs, proc_log_mag_db = resample_log_uniform(proc_freqs, proc_mag_db)
            proc_fig = go.Figure()
            proc_fig.add_trace(go.Scattergl(
                x=proc_log_freqs, y=proc_log_mag_db, mode="lines",
                line=dict(color="#f43f5e", width=1),
            ))
            proc_fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
                xaxis=dict(type="log", dtick=1, tickformat="~s"),
                template="plotly_dark",
            )
            st.plotly_chart(proc_fig, use_container_width=True, key="processed_spectrum_plot")

        with result_player_box:
            st.subheader("4. Listen to the Modified Audio")
            proc_peaks = audio_utils.compute_peaks(st.session_state.processed_audio, num_points=400)
            audio_player.render_audio_player(
                st.session_state.processed_bytes,
                f"processed_{st.session_state.filename}",
                proc_peaks,
                key="processed",
            )

        if st.button("↺ Reset to original (clear all cuts)"):
            _reset_processed()
            st.session_state.bands = []
            st.rerun()

