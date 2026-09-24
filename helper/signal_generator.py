import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import audio_utils, storage, audio_player

st.set_page_config(page_title="Music Mixer", page_icon="🎧", layout="wide")
st.title("🎧 Multi-Track Music Mixer & Synthesizer")

# Session state initialization for track selection
if "selected_tracks" not in st.session_state:
    st.session_state.selected_tracks = []

# Fetch available audio files from workspace uploads directory
available_files = storage.list_uploaded_files()

# --- Sidebar Controls ---
st.sidebar.header("🎵 Load Music")

if not available_files:
    st.sidebar.warning("No files found in workspace. Upload clips via the editing page first.")
else:
    selected_file = st.sidebar.selectbox(
        "Select music track to add",
        options=["— select a track —"] + available_files,
        index=0
    )
    
    if st.sidebar.button("➕ Load Music", use_container_width=True):
        if selected_file != "— select a track —":
            file_path = storage.uploaded_file_path(selected_file)
            st.session_state.selected_tracks.append({
                "filename": selected_file,
                "path": file_path,
                "gain": 1.0
            })
            st.rerun()

# --- Section 1: Track Controls ---
st.subheader("1. Loaded Tracks")

if not st.session_state.selected_tracks:
    st.info("No tracks loaded yet. Select and load audio files from the sidebar menu to begin mixing.")
else:
    cols = st.columns(min(len(st.session_state.selected_tracks), 4))
    tracks_to_remove = []

    for i, trk in enumerate(st.session_state.selected_tracks):
        col_idx = i % 4
        with cols[col_idx]:
            with st.container(border=True):
                st.markdown(f"**Track {i+1}:** `{trk['filename']}`")
                trk["gain"] = st.slider(
                    f"Volume Gain #{i+1}",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(trk["gain"]),
                    step=0.1,
                    key=f"gain_{i}"
                )
                if st.button(f"🗑️ Remove", key=f"del_{i}", use_container_width=True):
                    tracks_to_remove.append(i)

    if tracks_to_remove:
        for idx in sorted(tracks_to_remove, reverse=True):
            st.session_state.selected_tracks.pop(idx)
        st.rerun()

# --- Signal Processing & Synthesis ---
if st.session_state.selected_tracks:
    target_sr = 44100
    composite_signal = audio_utils.mix_audio_files(
        st.session_state.selected_tracks, target_sr=target_sr
    )
    mp3_bytes = audio_utils.samples_to_mp3_bytes(composite_signal, target_sr)

    # --- Section 2: Visualizations ---
    st.subheader("2. Combined Signal Waveform & Spectrum")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Combined Time-Domain Waveform**")
        t, wave_vals = audio_utils.downsample_waveform(composite_signal, target_sr)
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scattergl(x=t, y=wave_vals, mode="lines", line=dict(color="#2dd4bf")))
        fig_wave.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(fig_wave, use_container_width=True)

    with col2:
        st.markdown("**Combined Frequency Spectrum**")
        freqs, mag_db = audio_utils.compute_spectrum(composite_signal, target_sr)
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scattergl(
            x=freqs[:int(len(freqs)/2)], 
            y=mag_db[:int(len(mag_db)/2)], 
            mode="lines", 
            line=dict(color="#f59e0b")
        ))
        fig_spec.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_title="Hz")
        st.plotly_chart(fig_spec, use_container_width=True)

    # --- Section 3: Preview Player ---
    st.subheader("3. Audio Demo")
    player_box = st.container(border=True)
    with player_box:
        st.caption("Listen to the combined multi-track mix:")
        peaks = audio_utils.compute_peaks(composite_signal, num_points=400)
        audio_player.render_audio_player(
            mp3_bytes,
            "demo_mix.mp3",
            peaks,
            key="mixed_demo"
        )

    # --- Section 4: Auto-Incrementing Export ---
    st.subheader("4. Export Combined Track")

    # --- Trimming / Segment Selection Controls ---
    st.markdown("**✂️ Audio Trimming & Segment Selection**")

    total_duration = float(len(composite_signal)) / float(target_sr)

    trim_col1, trim_col2 = st.columns(2)
    with trim_col1:
        start_time = st.number_input(
            "Start Time (seconds)",
            min_value=0.0,
            max_value=total_duration,
            value=0.0,
            step=0.5,
        )
    with trim_col2:
        end_time = st.number_input(
            "End Time (seconds)",
            min_value=0.0,
            max_value=total_duration,
            value=total_duration,
            step=0.5,
        )

    # Validate trim range
    if start_time >= end_time:
        st.error(
            "Start time must be strictly less than End time. Defaulting to full audio length."
        )
        start_time, end_time = 0.0, total_duration

    # Convert seconds to array sample indices
    start_idx = int(start_time * target_sr)
    end_idx = int(end_time * target_sr)

    # Slice the composite signal array
    trimmed_signal = composite_signal[start_idx:end_idx]
    trimmed_duration = float(len(trimmed_signal)) / float(target_sr)

    st.caption(
        f"Selected range: `{start_time:.2f}s` → `{end_time:.2f}s` (Total export duration: `{trimmed_duration:.2f}s`)"
    )

    # Re-encode only the trimmed portion to MP3 bytes for saving
    export_mp3_bytes = audio_utils.samples_to_mp3_bytes(
        trimmed_signal, target_sr
    )
    
    # Helper function to find the next available sequential filename (new_music_i.mp3)
    def get_next_default_filename() -> str:
        i = 1
        while (storage.UPLOADS_DIR / f"new_music_{i}.mp3").exists():
            i += 1
        return f"new_music_{i}.mp3"

    default_name = get_next_default_filename()

    filename_input = st.text_input(
        "Rename the generated music file (leave blank for automatic sequence):",
        placeholder=default_name,
    )

    if st.button("💾 Save & Add to Workspace", type="primary"):
        chosen_name = (
            filename_input.strip() if filename_input.strip() else default_name
        )

        if not chosen_name.lower().endswith(".mp3"):
            chosen_name += ".mp3"

        save_path = storage.UPLOADS_DIR / chosen_name

        # Write the trimmed byte array to disk
        with open(save_path, "wb") as f:
            f.write(export_mp3_bytes)

        st.success(
            f"Saved `{chosen_name}` (`{trimmed_duration:.2f}s`) to workspace uploads!"
        )