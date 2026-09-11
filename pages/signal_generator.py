import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import audio_utils, storage, audio_player

st.set_page_config(page_title="Signal Synthesizer", page_icon="🎼", layout="wide")
st.title("🎼 Frequency Synthesizer (Add Frequencies)")

if "synth_components" not in st.session_state:
    st.session_state.synth_components = [
        {"freq": 440.0, "amplitude": 1.0, "phase": 0.0, "wave_type": "sine"},
        {"freq": 880.0, "amplitude": 0.5, "phase": 0.0, "wave_type": "sine"}
    ]

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Signal Settings")
duration = st.sidebar.slider("Duration (seconds)", 0.5, 5.0, 2.0, 0.5)
sr = st.sidebar.selectbox("Sample Rate (Hz)", [22050, 44100, 48000], index=1)

if st.sidebar.button("➕ Add Frequency Component"):
    st.session_state.synth_components.append(
        {"freq": 220.0, "amplitude": 0.5, "phase": 0.0, "wave_type": "sine"}
    )
    st.rerun()

# --- Inputs for components ---
st.subheader("1. Frequency Components")
cols = st.columns(len(st.session_state.synth_components))

for i, comp in enumerate(st.session_state.synth_components):
    with cols[i]:
        st.markdown(f"**Component {i+1}**")
        comp["freq"] = st.number_input(
            f"Frequency (Hz) #{i+1}", 
            min_value=20.0, 
            max_value=10000.0, 
            value=comp["freq"], 
            step=10.0
        )
        comp["amplitude"] = st.slider(
            f"Amplitude #{i+1}", 
            0.0, 1.0, comp["amplitude"], 0.1
        )
        comp["wave_type"] = st.selectbox(
            f"Type #{i+1}", 
            ["sine", "cosine", "square"], 
            key=f"type_{i}"
        )
        if len(st.session_state.synth_components) > 1:
            if st.button(f"🗑️ Remove #{i+1}"):
                st.session_state.synth_components.pop(i)
                st.rerun()

# --- Generate Combined Signal ---
composite_signal = audio_utils.generate_composite_signal(
    st.session_state.synth_components, duration=duration, sr=sr
)
composite_bytes = audio_utils.samples_to_wav_bytes(composite_signal, sr)

# --- Visualizations ---
st.subheader("2. Combined Signal Waveform & Spectrum")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Combined Time-Domain Waveform**")
    t, wave_vals = audio_utils.downsample_waveform(composite_signal, sr)
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scattergl(x=t, y=wave_vals, mode="lines", line=dict(color="#2dd4bf")))
    fig_wave.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
    st.plotly_chart(fig_wave, use_container_width=True)

with col2:
    st.markdown("**Combined Frequency Spectrum**")
    freqs, mag_db = audio_utils.compute_spectrum(composite_signal, sr)
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scattergl(
        x=freqs[:int(len(freqs)/2)], 
        y=mag_db[:int(len(mag_db)/2)], 
        mode="lines", 
        line=dict(color="#f59e0b")
    ))
    fig_spec.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", xaxis_title="Hz")
    st.plotly_chart(fig_spec, use_container_width=True)

# --- Audio Demo Player ---
st.subheader("3. Audio Demo")
player_box = st.container(border=True)
with player_box:
    st.caption("Listen to a demo preview of your synthesized composite wave before saving:")
    peaks = audio_utils.compute_peaks(composite_signal, num_points=400)
    audio_player.render_audio_player(
        composite_bytes,
        "demo_synth.wav",
        peaks,
        key="synth_demo"
    )

# --- Save Generated File ---
st.subheader("4. Export Composite Wave")
filename_input = st.text_input("Save file as", "summed_frequencies.wav")

if st.button("💾 Save & Add to Uploads Workspace"):
    save_path = storage.UPLOADS_DIR / filename_input
    with open(save_path, "wb") as f:
        f.write(composite_bytes)
    st.success(f"Saved `{filename_input}` to workspace! You can now load it on the `frequency_editing` page.")