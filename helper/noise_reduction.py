import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import audio_utils, storage, audio_player

st.set_page_config(page_title="Noise Detection & Removal", page_icon="🧹", layout="wide")
st.title("🧹 Noise Detection & Removal Workspace")

st.caption(
    "Add synthetic noise to a clip for testing (constant hiss/hum, slowly "
    "drifting noise, or short click/burst noise), then detect and clean it up."
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
defaults = {
    "raw_audio": None,       # shared with the Frequency Cutting page
    "sr": None,
    "filename": None,
    "audio_bytes": None,
    "noisy_audio": None,     # np.ndarray, clip with test noise added
    "noisy_bytes": None,
    "noise_report": None,    # dict from audio_utils.analyze_noise
    "denoised_audio": None,  # np.ndarray, cleaned result
    "denoised_bytes": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _load_clip(path, display_name):
    samples, sr = audio_utils.load_audio(path)
    st.session_state.raw_audio = samples
    st.session_state.sr = sr
    st.session_state.filename = display_name
    st.session_state.audio_bytes = storage.read_bytes(path)
    st.session_state.noisy_audio = None
    st.session_state.noisy_bytes = None
    st.session_state.noise_report = None
    st.session_state.denoised_audio = None
    st.session_state.denoised_bytes = None


def _spectrum_fig(samples, sr, color):
    freqs, mag_db = audio_utils.compute_spectrum(samples, sr)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=freqs, y=mag_db, mode="lines", line=dict(color=color, width=1)))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
        xaxis=dict(type="log", dtick=1, tickformat="~s"),
        template="plotly_dark",
    )
    return fig


# ----------------------------------------------------------------------
# Sidebar — reuse whatever clip is loaded on the Frequency Cutting page,
# or load one directly here so this page also works standalone.
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

if st.session_state.raw_audio is None:
    st.info("Upload or choose a clip from the sidebar to get started.")
    st.stop()

samples = st.session_state.raw_audio
sr = st.session_state.sr

# ----------------------------------------------------------------------
# 1. Add test noise (for testing purposes)
# ----------------------------------------------------------------------
st.subheader("1. Add Test Noise")
st.caption(
    "Optional — create a noisy version of the clip to try the detect/remove steps "
    "below. Skip this if your own clip is already noisy."
)

NOISE_TYPES = {
    "White noise (hiss) — stationary": "white",
    "Pink noise — stationary": "pink",
    "Hum (mains tone) — stationary": "hum",
    "White + Hum — stationary": "white+hum",
    "Drifting noise — level rises/falls over time": "drift",
    "Bursts / clicks / pops — brief transient noise": "burst",
}

with st.container(border=True):
    n_col1, n_col2, n_col3 = st.columns([2, 2, 1], vertical_alignment="bottom")

    with n_col1:
        noise_type_label = st.selectbox("Noise type", options=list(NOISE_TYPES.keys()), key="noise_type_select")
    noise_type = NOISE_TYPES[noise_type_label]

    with n_col2:
        snr_db = st.slider(
            "Target SNR (dB) — lower = noisier", min_value=-10.0, max_value=40.0, value=15.0, step=1.0,
            key="noise_snr",
        )
        hum_freq = 50.0
        n_bursts, burst_ms = 6, 30.0
        if noise_type in ("hum", "white+hum"):
            hum_freq = st.select_slider("Hum frequency (Hz)", options=[50.0, 60.0], value=50.0, key="hum_freq")
        elif noise_type == "burst":
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                n_bursts = st.number_input("Number of bursts", min_value=1, max_value=30, value=6, step=1)
            with b_col2:
                burst_ms = st.number_input("Burst duration (ms)", min_value=5.0, max_value=200.0, value=30.0, step=5.0)

    with n_col3:
        add_clicked = st.button("🔊 Add Noise", use_container_width=True, type="primary")

    if add_clicked:
        noisy = audio_utils.add_noise(
            samples, sr, noise_type=noise_type, snr_db=snr_db, hum_freq=hum_freq,
            n_bursts=n_bursts, burst_duration_ms=burst_ms,
        )
        st.session_state.noisy_audio = noisy
        st.session_state.noisy_bytes = audio_utils.samples_to_wav_bytes(noisy, sr)
        st.session_state.noise_report = None
        st.session_state.denoised_audio = None
        st.session_state.denoised_bytes = None
        st.rerun()

    if st.session_state.noisy_audio is not None:
        st.markdown("**Noisy Spectrum**")
        st.plotly_chart(
            _spectrum_fig(st.session_state.noisy_audio, sr, "#f43f5e"),
            use_container_width=True, key="noisy_spectrum_plot",
        )
        peaks = audio_utils.compute_peaks(st.session_state.noisy_audio, num_points=400)
        audio_player.render_audio_player(
            st.session_state.noisy_bytes, f"noisy_{st.session_state.filename}", peaks, key="noisy",
        )
        if st.button("Clear test noise"):
            st.session_state.noisy_audio = None
            st.session_state.noisy_bytes = None
            st.session_state.noise_report = None
            st.session_state.denoised_audio = None
            st.session_state.denoised_bytes = None
            st.rerun()

# The working signal: the noisy test clip if one was created, otherwise
# whatever was loaded (in case it's already noisy real-world audio).
working_audio = st.session_state.noisy_audio if st.session_state.noisy_audio is not None else samples
working_label = "noisy test clip" if st.session_state.noisy_audio is not None else "loaded clip"

st.divider()

# ----------------------------------------------------------------------
# 2. Detect noise
# ----------------------------------------------------------------------
st.subheader("2. Detect Noise")
st.caption(f"Analyzing the {working_label}.")

with st.container(border=True):
    method_label = st.radio(
        "How should the noise level be estimated?",
        options=[
            "Automatic — assumes a constant noise floor",
            "Adaptive — tracks noise that rises/falls over time",
            "Manual region — I can point at a known quiet stretch",
        ],
        horizontal=False,
    )
    profile_method = {
        "Automatic — assumes a constant noise floor": "auto",
        "Adaptive — tracks noise that rises/falls over time": "adaptive",
        "Manual region — I can point at a known quiet stretch": "region",
    }[method_label]

    noise_start, noise_end = 0.0, None
    if profile_method == "region":
        total_dur = audio_utils.get_duration(working_audio, sr)
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            noise_start = st.number_input("Region start (s)", min_value=0.0, max_value=total_dur, value=0.0, step=0.1)
        with r_col2:
            noise_end = st.number_input("Region end (s)", min_value=0.0, max_value=total_dur, value=min(0.5, total_dur), step=0.1)

    transient_sensitivity = st.slider(
        "Click/burst detection sensitivity (lower = catches more, but more false positives)",
        min_value=1.5, max_value=6.0, value=3.0, step=0.5,
    )

    if st.button("🔍 Analyze Noise", type="primary"):
        report = audio_utils.analyze_noise(
            working_audio, sr, method=profile_method, noise_start=noise_start, noise_end=noise_end,
            transient_sensitivity=transient_sensitivity,
        )
        st.session_state.noise_report = report
        st.rerun()

    report = st.session_state.noise_report
    if report is not None:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Noise floor", f"{report['noise_floor_db']:.1f} dB")
        m_col2.metric("Estimated SNR", f"{report['estimated_snr_db']:.1f} dB")
        m_col3.metric("Hiss/hum", "🔴 Noisy" if report["is_noisy"] else "🟢 Clean")
        m_col4.metric("Clicks/bursts", f"{report['transient_count']} found" if report["has_transients"] else "None found")

        if report["has_transients"]:
            preview = ", ".join(f"{s:.2f}–{e:.2f}s" for s, e in report["transient_regions_sec"][:8])
            more = "" if report["transient_count"] <= 8 else f" (+{report['transient_count'] - 8} more)"
            st.caption(f"Detected burst/click locations: {preview}{more}")

        freqs, mag_db = audio_utils.compute_spectrum(working_audio, sr)
        profile = report["profile"]
        profile_avg = profile.mean(axis=0) if report["profile_is_adaptive"] else profile
        profile_db = 20.0 * np.log10(profile_avg + 1e-10)
        profile_freqs = np.fft.rfftfreq(2048, d=1.0 / sr)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=freqs, y=mag_db, mode="lines", name="Signal", line=dict(color="#38bdf8", width=1)))
        fig.add_trace(go.Scattergl(x=profile_freqs, y=profile_db, mode="lines", name="Detected noise profile", line=dict(color="#f43f5e", width=2, dash="dash")))
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude (dB)",
            xaxis=dict(type="log", dtick=1, tickformat="~s"),
            template="plotly_dark", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True, key="noise_profile_plot")

st.divider()

# ----------------------------------------------------------------------
# 3. Remove noise
# ----------------------------------------------------------------------
st.subheader("3. Remove Noise")

with st.container(border=True):
    rc_col1, rc_col2 = st.columns(2)
    with rc_col1:
        strength = st.slider(
            "Reduction strength", min_value=0.0, max_value=3.0, value=1.3, step=0.1,
            help="0 = no change, 1 = subtract the full estimated noise profile, >1 = more aggressive.",
        )
    with rc_col2:
        floor = st.slider(
            "Noise floor to keep (fraction of original)",
            min_value=0.0, max_value=0.3, value=0.02, step=0.01,
            help="Prevents harsh gating artifacts in quiet passages.",
        )

    suppress_transients = st.checkbox("Also detect & suppress clicks/bursts", value=True)

    remove_clicked = st.button("🧹 Remove Noise", type="primary")

    if remove_clicked:
        denoised = audio_utils.reduce_noise(
            working_audio, sr,
            profile_method=profile_method,
            noise_start=noise_start,
            noise_end=noise_end,
            reduction_strength=strength,
            floor=floor,
            suppress_transients=suppress_transients,
            transient_sensitivity=transient_sensitivity,
        )
        st.session_state.denoised_audio = denoised
        st.session_state.denoised_bytes = audio_utils.samples_to_wav_bytes(denoised, sr)
        st.rerun()

    if st.session_state.denoised_audio is not None:
        st.markdown("**Cleaned Spectrum**")
        st.plotly_chart(
            _spectrum_fig(st.session_state.denoised_audio, sr, "#22c55e"),
            use_container_width=True, key="denoised_spectrum_plot",
        )
        st.markdown("**Listen to the Cleaned Audio**")
        peaks = audio_utils.compute_peaks(st.session_state.denoised_audio, num_points=400)
        audio_player.render_audio_player(
            st.session_state.denoised_bytes, f"denoised_{st.session_state.filename}", peaks, key="denoised",
        )

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                "⬇️ Download cleaned WAV",
                data=st.session_state.denoised_bytes,
                file_name=f"denoised_{st.session_state.filename or 'clip'}.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        with d_col2:
            if st.button("💾 Save to workspace uploads", use_container_width=True):
                target = storage.UPLOADS_DIR / f"denoised_{st.session_state.filename or 'clip'}.wav"
                with open(target, "wb") as f:
                    f.write(st.session_state.denoised_bytes)
                st.success(f"Saved `{target.name}` to workspace uploads!")