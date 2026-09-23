import io
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:  # pydub / ffmpeg might not be installed
    _HAS_PYDUB = False


def load_audio(file_path) -> tuple[np.ndarray, int]:
    """
    Load an audio file (wav/flac/ogg natively, mp3/m4a/... via pydub+ffmpeg)
    and return (samples, sample_rate) where samples is a mono float32
    NumPy array normalized to roughly [-1, 1].
    """
    file_path = Path(file_path)
    try:
        data, sr = sf.read(str(file_path), dtype="float32", always_2d=False)
    except Exception as e:
        if not _HAS_PYDUB:
            raise RuntimeError(
                f"Could not read '{file_path.name}' with soundfile and pydub "
                f"is not available for fallback decoding. Original error: {e}"
            )
        seg = AudioSegment.from_file(file_path)
        sr = seg.frame_rate
        raw = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels > 1:
            raw = raw.reshape((-1, seg.channels))
        max_val = float(2 ** (8 * seg.sample_width - 1))
        data = raw / max_val

    if data.ndim > 1:
        data = data.mean(axis=1)

    return data.astype(np.float32), int(sr)


def get_duration(samples: np.ndarray, sr: int) -> float:
    return float(len(samples)) / float(sr)


def compute_spectrum(samples: np.ndarray, sr: int):
    """Return (freqs_hz, magnitude_db) using a real FFT of the whole clip."""
    n = len(samples)
    fft_vals = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    magnitude = np.abs(fft_vals)
    magnitude_db = 20.0 * np.log10(magnitude + 1e-10)
    return freqs, magnitude_db


def apply_frequency_cuts(samples: np.ndarray, sr: int, bands: list[dict]) -> np.ndarray:
    """
    Apply a list of frequency-band "scissor cuts" in the frequency domain
    and reconstruct the audio with the inverse FFT.

    bands: list of {"low": float_hz, "high": float_hz, "gain": float}
        gain = 0.0  -> completely cut (band-stop)
        gain < 1.0  -> attenuate
        gain > 1.0  -> amplify
        gain = 1.0  -> unchanged
    """
    n = len(samples)
    fft_vals = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    processed_fft = fft_vals.copy()
    for band in bands:
        low = float(band["low"])
        high = float(band["high"])
        gain = float(band["gain"])
        mask = (freqs >= low) & (freqs <= high)
        processed_fft[mask] *= gain

    processed = np.fft.irfft(processed_fft, n=n).astype(np.float32)

    peak = np.max(np.abs(processed)) if len(processed) else 0.0
    if peak > 1.0:
        processed = processed / peak

    return processed


def apply_band_operation(
    samples: np.ndarray,
    sr: int,
    low: float,
    high: float,
    operation: str,
    gain: float = 1.0,
) -> np.ndarray:
    """
    Apply ONE scissor operation to a single selected frequency band
    [low, high] Hz and rebuild the waveform with the inverse FFT.

    operation:
        "remove"     -> zero out everything INSIDE the band (band-stop)
        "isolate"    -> zero out everything OUTSIDE the band (band-pass)
        "attenuate"  -> multiply the band by `gain` (expected 0.0 - 1.0)
        "amplify"    -> multiply the band by `gain` (expected >= 1.0)
    """
    n = len(samples)
    fft_vals = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    low, high = sorted((float(low), float(high)))
    mask = (freqs >= low) & (freqs <= high)

    processed_fft = fft_vals.copy()

    if operation == "remove":
        processed_fft[mask] = 0.0
    elif operation == "isolate":
        processed_fft[~mask] = 0.0
    elif operation == "attenuate":
        g = float(np.clip(gain, 0.0, 1.0))
        processed_fft[mask] *= g
    elif operation == "amplify":
        g = max(float(gain), 1.0)
        processed_fft[mask] *= g
    else:
        raise ValueError(f"Unknown operation: {operation!r}")

    processed = np.fft.irfft(processed_fft, n=n).astype(np.float32)

    peak = np.max(np.abs(processed)) if len(processed) else 0.0
    if peak > 1.0:
        processed = processed / peak

    return processed


def samples_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    """
    Encode a float32 NumPy array as an in-memory 16-bit PCM WAV file.
    """
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_wav(samples: np.ndarray, sr: int, path) -> None:
    sf.write(str(path), samples, sr, subtype="PCM_16")


def downsample_waveform(samples: np.ndarray, sr: int, max_points: int = 3000):
    """
    Min/max envelope downsampling so long clips still render a faithful
    waveform shape quickly. Returns (time_seconds, values) arrays where
    values alternate low/high per bin, ready for a line/area plot.
    """
    n = len(samples)
    if n <= max_points:
        t = np.arange(n) / sr
        return t, samples

    bin_size = int(np.ceil(n / max_points))
    n_bins = int(np.ceil(n / bin_size))

    times = []
    values = []
    for i in range(n_bins):
        start = i * bin_size
        end = min(start + bin_size, n)
        chunk = samples[start:end]
        if len(chunk) == 0:
            continue
        t_center = (start + end) / 2 / sr
        times.append(t_center)
        values.append(float(chunk.min()))
        times.append(t_center)
        values.append(float(chunk.max()))

    return np.array(times), np.array(values)


def compute_peaks(samples: np.ndarray, num_points: int = 400) -> list:
    """
    Compute a normalized (0..1) amplitude envelope for the mini waveform
    drawn on the <canvas> playhead widget.
    """
    n = len(samples)
    if n == 0:
        return [0.0] * num_points

    bin_size = max(1, n // num_points)
    peaks = []
    for i in range(0, n, bin_size):
        chunk = samples[i:i + bin_size]
        if len(chunk) == 0:
            continue
        peaks.append(float(np.max(np.abs(chunk))))

    peaks = peaks[:num_points]
    max_val = max(peaks) if peaks else 1.0
    if max_val <= 0:
        max_val = 1.0
    l = [p / max_val for p in peaks]
    return l




def mix_audio_files(
    tracks: list[dict], target_sr: int = 44100
) -> np.ndarray:
    """
    Mix multiple audio clips into a single normalized composite signal.
    Correctly aligns sample lengths and balances gains across all tracks.
    """
    if not tracks:
        return np.array([], dtype=np.float32)

    loaded_signals = []

    for trk in tracks:
        path = trk["path"]
        gain = float(trk.get("gain", 1.0))
        
        # Load audio using existing load_audio pipeline
        samples, sr = load_audio(path)
        
        # Proper length-preserving resampling if sample rates differ
        if sr != target_sr:
            num_target_samples = int(round(len(samples) * float(target_sr) / float(sr)))
            old_indices = np.linspace(0, len(samples) - 1, num=len(samples))
            new_indices = np.linspace(0, len(samples) - 1, num=num_target_samples)
            samples = np.interp(new_indices, old_indices, samples).astype(np.float32)

        # Apply track-specific volume multiplier
        samples = samples * gain
        loaded_signals.append(samples)

    if not loaded_signals:
        return np.array([], dtype=np.float32)

    # Find maximum duration among loaded tracks
    max_len = max(len(s) for s in loaded_signals)

    # Allocate a zero-filled composite buffer of the maximum length
    composite = np.zeros(max_len, dtype=np.float32)

    # Sum signals into the buffer simultaneously starting from t = 0
    for sig in loaded_signals:
        composite[:len(sig)] += sig

    # Soft peak normalization to prevent distortion while preserving relative volumes
    peak = np.max(np.abs(composite))
    if peak > 1.0:
        composite = composite / peak

    return composite.astype(np.float32)

def samples_to_mp3_bytes(samples: np.ndarray, sr: int) -> bytes:
    """
    Encode a float32 NumPy array into MP3 bytes using pydub/soundfile.
    Falls back to WAV container if MP3 encoder is missing on target system.
    """
    buf = io.BytesIO()
    try:
        if _HAS_PYDUB:
            # Convert float32 array (-1.0 to 1.0) to int16 PCM
            pcm_data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            segment = AudioSegment(
                pcm_data.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            segment.export(buf, format="mp3", bitrate="192k")
            return buf.getvalue()
    except Exception:
        pass

    # Fallback export via soundfile WAV encoding
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ----------------------------------------------------------------------
# Noise: add (for testing), detect, and remove
# ----------------------------------------------------------------------
def add_noise(
    samples: np.ndarray,
    sr: int,
    noise_type: str = "white",
    snr_db: float = 20.0,
    hum_freq: float = 50.0,
    n_bursts: int = 6,
    burst_duration_ms: float = 30.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Inject synthetic noise into a clean signal, purely for testing the
    detect/remove pipeline below. Not part of the "real" editing chain.

    noise_type: "white" | "pink" | "hum" | "white+hum" | "drift" | "burst"
        "white"  -> flat-spectrum Gaussian noise (hiss), constant level -> STATIONARY
        "pink"   -> 1/f-shaped noise (softer hiss), constant level -> STATIONARY
        "hum"    -> a pure tone at `hum_freq` Hz (mains hum, 50/60 Hz) -> STATIONARY
        "white+hum" -> a mix of both -> STATIONARY
        "drift"  -> white noise whose amplitude slowly rises and falls over
                    the clip (like passing traffic) -> SLOWLY-VARYING
        "burst"  -> `n_bursts` short, isolated noise bursts (clicks/pops/
                    coughs) scattered through an otherwise clean clip ->
                    TRANSIENT (confined to just a few frames each)
    snr_db: desired signal-to-noise ratio in dB (power-averaged over the
        whole clip). Lower = noisier.
    seed: optional RNG seed so a "noisy test clip" is reproducible.
    """
    rng = np.random.default_rng(seed)
    n = len(samples)
    if n == 0:
        return samples.copy()

    signal_power = float(np.mean(samples.astype(np.float64) ** 2)) + 1e-12

    def _white(n):
        return rng.standard_normal(n).astype(np.float32)

    def _pink(n):
        # Shape white noise with a 1/sqrt(f) magnitude response -> pink (1/f power)
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs = freqs.copy()
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        spec = spec / np.sqrt(freqs)
        pink = np.fft.irfft(spec, n=n)
        return pink.astype(np.float32)

    def _hum(n, freq):
        t = np.arange(n) / float(sr)
        return np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    def _drift(n):
        # White noise under a slow, randomly-phased amplitude envelope so
        # the noise level rises and falls over several seconds.
        base = rng.standard_normal(n).astype(np.float32)
        t = np.arange(n) / float(sr)
        slow_hz = 0.1 + rng.uniform(0.0, 0.15)  # one swell every ~7-10s
        env = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2.0 * np.pi * slow_hz * t + rng.uniform(0, 2 * np.pi)))
        return (base * env).astype(np.float32)

    def _burst(n, count, duration_ms):
        out = np.zeros(n, dtype=np.float32)
        burst_len = max(4, int(sr * duration_ms / 1000.0))
        fade = np.hanning(burst_len).astype(np.float32)
        for _ in range(count):
            if n <= burst_len:
                start = 0
            else:
                start = int(rng.integers(0, n - burst_len))
            spike = rng.standard_normal(burst_len).astype(np.float32) * fade
            out[start:start + burst_len] += spike
        return out

    if noise_type == "white":
        noise = _white(n)
    elif noise_type == "pink":
        noise = _pink(n)
    elif noise_type == "hum":
        noise = _hum(n, hum_freq)
    elif noise_type == "white+hum":
        noise = 0.6 * _white(n) + 0.4 * _hum(n, hum_freq)
    elif noise_type == "drift":
        noise = _drift(n)
    elif noise_type == "burst":
        noise = _burst(n, n_bursts, burst_duration_ms)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    noise = (noise * scale).astype(np.float32)

    noisy = samples.astype(np.float32) + noise
    peak = np.max(np.abs(noisy)) if len(noisy) else 0.0
    if peak > 1.0:
        noisy = noisy / peak

    return noisy.astype(np.float32)


def _stft(samples: np.ndarray, frame_size: int = 2048, hop_size: int = 512):
    """Simple windowed STFT with zero-padding on both ends, returns complex spectra
    of shape (n_frames, frame_size // 2 + 1)."""
    window = np.hanning(frame_size).astype(np.float32)
    samples = samples.astype(np.float32)
    padded = np.concatenate(
        [np.zeros(frame_size, dtype=np.float32), samples, np.zeros(frame_size, dtype=np.float32)]
    )
    n_frames = max(1, 1 + (len(padded) - frame_size) // hop_size)
    frames = np.empty((n_frames, frame_size), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        chunk = padded[start:start + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        frames[i] = chunk * window
    spec = np.fft.rfft(frames, axis=1)
    return spec


def _istft(spec: np.ndarray, frame_size: int = 2048, hop_size: int = 512, out_len: int | None = None):
    """Inverse of `_stft` via overlap-add, trimming back to `out_len` samples."""
    window = np.hanning(frame_size).astype(np.float32)
    frames = np.fft.irfft(spec, n=frame_size, axis=1).astype(np.float32)
    n_frames = frames.shape[0]
    total_len = (n_frames - 1) * hop_size + frame_size

    out = np.zeros(total_len, dtype=np.float32)
    norm = np.zeros(total_len, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        out[start:start + frame_size] += frames[i] * window
        norm[start:start + frame_size] += window ** 2
    norm[norm < 1e-8] = 1e-8
    out = out / norm

    pad = frame_size
    if out_len is not None:
        out = out[pad: pad + out_len]
    else:
        out = out[pad:-pad]
    return out.astype(np.float32)


def _quiet_frame_profile(mag: np.ndarray, quiet_frame_fraction: float = 0.2) -> np.ndarray:
    """
    Estimate a noise magnitude profile by picking the quietest fraction of
    FRAMES (by total energy) and averaging their magnitude spectra.

    This replaces an earlier, buggier approach that took a low PERCENTILE
    of each bin's magnitude across all frames. That undershoots badly: a
    single frame's noise magnitude at a given bin naturally fluctuates a
    lot from frame to frame (its distribution is Rayleigh-shaped), so a
    low percentile of that spread lands well below the true average noise
    level -- empirically only ~35% of it for white noise, which is why
    noise removal felt weak even at "full strength". Averaging over
    several genuinely quiet frames converges to the correct level instead,
    because averaging cancels the fluctuation rather than picking its
    unlucky low tail.
    """
    frame_energy = np.sum(mag.astype(np.float64) ** 2, axis=1)
    threshold = np.percentile(frame_energy, quiet_frame_fraction * 100.0)
    quiet_frames = mag[frame_energy <= threshold]
    if len(quiet_frames) < 3:
        quiet_frames = mag
    return quiet_frames.mean(axis=0)


def _rolling_quiet_frame_profile(mag: np.ndarray, window_frames: int = 40, quiet_frame_fraction: float = 0.2) -> np.ndarray:
    """
    Adaptive/tracking noise profile: for EACH frame, estimate the noise
    magnitude per frequency bin from the quietest frames within a sliding
    window of neighboring frames (centered on that frame), instead of one
    fixed profile for the whole clip.

    This is what lets the denoiser follow noise that slowly rises and falls
    in level over time (traffic swelling in and out, an AC unit cycling,
    etc.) instead of using a single average that's wrong half the time.

    Returns shape (n_frames, n_bins) -- one profile per frame.
    """
    n_frames = mag.shape[0]
    half = max(1, window_frames // 2)
    profile = np.empty_like(mag)
    for i in range(n_frames):
        lo = max(0, i - half)
        hi = min(n_frames, i + half + 1)
        profile[i] = _quiet_frame_profile(mag[lo:hi], quiet_frame_fraction)
    return profile


def _profile_from_mag(
    mag: np.ndarray,
    sr: int,
    hop_size: int,
    method: str = "auto",
    noise_start: float = 0.0,
    noise_end: float | None = None,
    quiet_frame_fraction: float = 0.2,
    window_frames: int = 40,
) -> np.ndarray:
    """Shared implementation behind estimate_noise_profile / reduce_noise
    so a magnitude spectrogram only has to be computed once. method:
    "region" -> shape (n_bins,); "auto" -> shape (n_bins,);
    "adaptive" -> shape (n_frames, n_bins)."""
    if method == "region":
        start_frame = max(0, int((noise_start * sr) / hop_size))
        end_frame = min(mag.shape[0], int((noise_end * sr) / hop_size)) if noise_end else mag.shape[0]
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        noise_frames = mag[start_frame:end_frame]
        if len(noise_frames) == 0:
            noise_frames = mag
        return noise_frames.mean(axis=0).astype(np.float32)
    elif method == "adaptive":
        return _rolling_quiet_frame_profile(mag, window_frames, quiet_frame_fraction).astype(np.float32)
    else:  # "auto"
        return _quiet_frame_profile(mag, quiet_frame_fraction).astype(np.float32)


def estimate_noise_profile(
    samples: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    method: str = "auto",
    noise_start: float = 0.0,
    noise_end: float | None = None,
    quiet_frame_fraction: float = 0.2,
    window_frames: int = 40,
) -> np.ndarray:
    """
    Estimate a noise magnitude profile.

    method:
      "region"   -> average magnitude over [noise_start, noise_end] (use
                    when you can point at a known quiet stretch)
      "auto"     -> average magnitude over the quietest `quiet_frame_fraction`
                    of frames in the whole clip -- good for a CONSTANT
                    hiss/hum (stationary noise)
      "adaptive" -> a per-frame profile built the same way but from a
                    sliding window of neighboring frames -- good for noise
                    whose LEVEL slowly drifts over time (e.g. traffic, an
                    AC hum that cycles on/off)

    If `noise_end` is given and `method` is left as "auto", this
    automatically switches to "region" (keeps old call sites working).

    Returns a (n_bins,) array for "region"/"auto", or (n_frames, n_bins)
    for "adaptive".
    """
    if method == "auto" and noise_end is not None and noise_end > noise_start:
        method = "region"
    spec = _stft(samples, frame_size, hop_size)
    mag = np.abs(spec)
    return _profile_from_mag(
        mag, sr, hop_size, method=method, noise_start=noise_start, noise_end=noise_end,
        quiet_frame_fraction=quiet_frame_fraction, window_frames=window_frames,
    )


def detect_transients(
    samples: np.ndarray,
    sr: int,
    sensitivity: float = 3.0,
    frame_size: int = 512,
    hop_size: int = 128,
    flatness_threshold: float = 0.25,
) -> np.ndarray:
    """
    Detect short, isolated burst noise (clicks, pops, coughs, a door slam)
    confined to a brief stretch of the clip -- too short-lived and too
    broadband for a frequency *profile* to describe, so it needs its own
    detector rather than spectral subtraction.

    Runs at a FINE time resolution (small frame/hop, independent of the
    frame_size/hop_size used elsewhere for hiss/hum removal) so a brief
    event is localized precisely instead of smeared across a big analysis
    window. Two conditions must BOTH hold for a moment to be flagged:

      1. Spectral flux spike -- the frame-to-frame magnitude jump (summed
         over all frequency bins) is a strong outlier (`sensitivity` MADs
         above the median). Loud is not the same as noisy, but a SUDDEN
         change usually means something new started.
      2. Broadband ("flat") spectrum -- measured via spectral flatness
         (geometric mean / arithmetic mean of the power spectrum, 0..1).
         A true click/pop spreads energy evenly across all frequencies
         (flatness close to 1, like white noise). A real note or voice
         onset, even a sudden one, is still mostly TONAL -- energy
         concentrated in a few harmonics (flatness close to 0). Requiring
         both conditions is what keeps this from flagging every sharp
         attack in music/speech as "noise".

    Returns a boolean array, ONE ENTRY PER SAMPLE (not per frame), so it
    lines up directly with the waveform for suppression.
    """
    n = len(samples)
    spec = _stft(samples, frame_size, hop_size)
    mag = np.abs(spec)
    n_frames = mag.shape[0]
    if n_frames < 5:
        return np.zeros(n, dtype=bool)

    power = mag.astype(np.float64) ** 2 + 1e-12

    flux = np.sum(np.maximum(np.diff(mag, axis=0), 0.0), axis=1)
    flux = np.concatenate([[flux[0]], flux])
    median_flux = np.median(flux)
    mad_flux = np.median(np.abs(flux - median_flux)) + 1e-9
    flux_spike = flux > median_flux + sensitivity * 1.4826 * mad_flux

    geo_mean = np.exp(np.mean(np.log(power), axis=1))
    arith_mean = np.mean(power, axis=1)
    flatness = geo_mean / arith_mean
    is_broadband = flatness > flatness_threshold

    is_transient_frame = flux_spike & is_broadband

    sample_mask = np.zeros(n, dtype=bool)
    pad_extra = hop_size // 2
    for i in np.where(is_transient_frame)[0]:
        center = i * hop_size
        start = max(0, center - hop_size // 2 - pad_extra)
        end = min(n, center + hop_size // 2 + pad_extra)
        sample_mask[start:end] = True
    return sample_mask


def _contiguous_regions_sec(mask: np.ndarray, sr: int) -> list:
    """Turn a per-sample boolean mask into a list of (start_sec, end_sec) spans."""
    n = len(mask)
    if not mask.any():
        return []
    diffs = np.diff(mask.astype(np.int8))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [n]])
    return [(round(s / sr, 3), round(e / sr, 3)) for s, e in zip(starts, ends)]


def _dirty_coarse_frames(
    sample_mask: np.ndarray, n_samples: int, frame_size: int, hop_size: int, overlap_thresh: float = 0.3
) -> np.ndarray:
    """
    Map a fine, sample-level transient mask onto the coarser STFT grid used
    for the main hiss/hum removal: a coarse frame is only marked "dirty" if
    more than `overlap_thresh` of its samples are flagged, not just any
    overlap at all -- otherwise nearly every frame touching a short click
    gets flagged and far too much real audio gets overwritten with it.
    """
    n_frames = max(1, 1 + (n_samples + 2 * frame_size - frame_size) // hop_size)
    is_dirty = np.zeros(n_frames, dtype=bool)
    pad = frame_size
    for i in range(n_frames):
        s = i * hop_size - pad
        e = s + frame_size
        s_c, e_c = max(0, s), min(n_samples, e)
        if s_c < e_c and sample_mask[s_c:e_c].mean() > overlap_thresh:
            is_dirty[i] = True
    return is_dirty


def _suppress_transient_frames(mag: np.ndarray, is_transient: np.ndarray, neighbor_frames: int = 4) -> np.ndarray:
    """
    Replace flagged transient frames' magnitude with the per-bin median of
    nearby NON-transient frames, so a brief burst is tamed regardless of
    what frequencies it happens to contain -- before the stationary/
    adaptive profile subtraction even runs. The overlap-add reconstruction
    afterwards smooths this into the surrounding audio rather than leaving
    a hard edge.
    """
    n_frames = mag.shape[0]
    out = mag.copy()
    clean_idx = np.where(~is_transient)[0]
    if len(clean_idx) == 0:
        return out  # everything flagged (unusual) -- nothing clean to reference

    for i in np.where(is_transient)[0]:
        lo, hi = max(0, i - neighbor_frames), min(n_frames, i + neighbor_frames + 1)
        local_clean = [j for j in range(lo, hi) if not is_transient[j]]
        if local_clean:
            out[i] = np.median(mag[local_clean], axis=0)
        else:
            nearest = clean_idx[np.argmin(np.abs(clean_idx - i))]
            out[i] = mag[nearest]
    return out


def analyze_noise(
    samples: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    method: str = "auto",
    noise_start: float = 0.0,
    noise_end: float | None = None,
    transient_sensitivity: float = 3.0,
) -> dict:
    """
    "Detect noise" step: quantify how noisy a clip is without modifying it,
    covering all three cases -- constant hiss/hum, slowly-drifting noise,
    and isolated transient bursts. Returns a report dict for the UI.
    """
    if method == "auto" and noise_end is not None and noise_end > noise_start:
        method = "region"

    spec = _stft(samples, frame_size, hop_size)
    mag = np.abs(spec)

    profile = _profile_from_mag(
        mag, sr, hop_size, method=method, noise_start=noise_start, noise_end=noise_end
    )
    avg_profile = profile.mean(axis=0) if profile.ndim == 2 else profile
    avg_mag = mag.mean(axis=0)
    signal_est = np.maximum(avg_mag - avg_profile, 0.0)

    noise_power = float(np.mean(avg_profile.astype(np.float64) ** 2)) + 1e-12
    signal_power = float(np.mean(signal_est.astype(np.float64) ** 2)) + 1e-12
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    noise_floor_db = 20.0 * np.log10(float(np.mean(avg_profile)) + 1e-10)

    transient_mask = detect_transients(samples, sr, sensitivity=transient_sensitivity)
    transient_regions_sec = _contiguous_regions_sec(transient_mask, sr)

    return {
        "noise_floor_db": float(noise_floor_db),
        "estimated_snr_db": float(snr_db),
        "is_noisy": bool(snr_db < 15.0),
        "profile": profile,
        "profile_is_adaptive": bool(profile.ndim == 2),
        "transient_count": len(transient_regions_sec),
        "transient_regions_sec": transient_regions_sec,
        "has_transients": bool(len(transient_regions_sec) > 0),
    }


def reduce_noise(
    samples: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    profile_method: str = "auto",
    noise_start: float = 0.0,
    noise_end: float | None = None,
    quiet_frame_fraction: float = 0.2,
    window_frames: int = 40,
    reduction_strength: float = 1.3,
    floor: float = 0.02,
    suppress_transients: bool = True,
    transient_sensitivity: float = 3.0,
    transient_overlap_thresh: float = 0.3,
) -> np.ndarray:
    """
    Remove noise, handling all three noise behaviors in one pass:

    1. TRANSIENT (clicks/pops/coughs confined to a brief stretch): located
       precisely via `detect_transients` (fine time resolution), mapped
       onto the coarser hiss-removal grid via `_dirty_coarse_frames` (only
       frames MOSTLY covered by the flagged stretch count -- this is what
       keeps normal audio around a short click from being overwritten),
       then those frames' magnitude is replaced with nearby clean frames'
       median -- a frequency-domain fix, not a straight time-domain
       "blank and interpolate", because a burst can be tens of
       milliseconds long and a straight-line bridge over that much real
       signal is more damaging than the burst itself.
    2. STATIONARY / SLOWLY-DRIFTING noise (hiss, hum, traffic swells):
       handled by spectral subtraction using a profile from
       `_profile_from_mag`. Pass `profile_method="auto"` for a constant
       noise floor, `"adaptive"` to track a noise level that rises and
       falls over time, or `"region"` (with noise_start/noise_end) if you
       can point at a known quiet stretch.
    3. Original phase is kept throughout; the waveform is rebuilt with the
       inverse STFT (overlap-add).

    reduction_strength: 0.0 = no change, 1.0 = subtract the full estimated
        noise profile, >1.0 = more aggressive (may distort quiet signal).
    floor: minimum magnitude to keep, as a fraction of the ORIGINAL bin
        magnitude (prevents harsh gating artifacts / musical noise).
    """
    if profile_method == "auto" and noise_end is not None and noise_end > noise_start:
        profile_method = "region"

    spec = _stft(samples, frame_size, hop_size)
    mag = np.abs(spec)
    phase = np.angle(spec)

    working_mag = mag
    if suppress_transients:
        sample_mask = detect_transients(samples, sr, sensitivity=transient_sensitivity)
        if sample_mask.any():
            is_dirty = _dirty_coarse_frames(
                sample_mask, len(samples), frame_size, hop_size, transient_overlap_thresh
            )
            if is_dirty.any():
                working_mag = _suppress_transient_frames(mag, is_dirty)

    profile = _profile_from_mag(
        working_mag, sr, hop_size, method=profile_method, noise_start=noise_start,
        noise_end=noise_end, quiet_frame_fraction=quiet_frame_fraction, window_frames=window_frames,
    )
    profile_b = profile[np.newaxis, :] if profile.ndim == 1 else profile

    gated_mag = working_mag - reduction_strength * profile_b
    gated_mag = np.maximum(gated_mag, floor * mag)

    gated_spec = gated_mag * np.exp(1j * phase)
    denoised = _istft(gated_spec, frame_size, hop_size, out_len=len(samples))

    peak = np.max(np.abs(denoised)) if len(denoised) else 0.0
    if peak > 1.0:
        denoised = denoised / peak

    return denoised.astype(np.float32)