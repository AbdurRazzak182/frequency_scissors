
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

    # Prevent clipping after amplification, keep silence as silence
    peak = np.max(np.abs(processed)) if len(processed) else 0.0
    if peak > 1.0:
        processed = processed / peak

    return processed


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
    return [p / max_val for p in peaks]
