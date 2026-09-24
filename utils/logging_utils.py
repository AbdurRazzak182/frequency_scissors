"""
logging_utils.py
-----------------
Tracks a history of every action performed on audio in the app -- loading a
clip, frequency-band edits, adding/removing noise, mixing tracks together,
and exporting a result -- using a Pandas DataFrame persisted as a local CSV
file. No database / backend involved, just data/logs/session_log.csv.

Every action type shares the same row shape; what makes each row specific
is the `details` JSON blob, which holds whatever parameters that action
type needs (frequency band + gain, noise type + SNR, mix gains, etc).
"""
from datetime import datetime
import json
import pandas as pd

from utils.storage import LOGS_DIR

LOG_PATH = LOGS_DIR / "session_log.csv"

COLUMNS = [
    "timestamp",
    "action_type",       # "load" | "frequency_edit" | "noise_add" | "noise_remove" | "mix" | "export"
    "source_filename",
    "sample_rate",
    "duration_sec",
    "details",            # JSON string, action-specific parameters
    "output_filename",
]

ACTION_LABELS = {
    "load": "📂 Loaded clip",
    "frequency_edit": "✂️ Frequency edit",
    "noise_add": "🔊 Added test noise",
    "noise_remove": "🧹 Removed noise",
    "mix": "🎚️ Mixed track",
    "export": "💾 Exported audio",
}


def _load_or_init() -> pd.DataFrame:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        # keep old logs readable even if a column was added later
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def load_log() -> pd.DataFrame:
    """Return the full history log, newest entries last (as written)."""
    return _load_or_init()


def log_action(
    action_type: str,
    source_filename: str,
    sample_rate: int,
    duration_sec: float,
    details: dict | None = None,
    output_filename: str = "",
) -> pd.DataFrame:
    """
    Append one row describing an action to the persistent log and return
    the updated DataFrame. `details` is any small dict of action-specific
    parameters (e.g. {"operation": "remove", "low": 200, "high": 800}) --
    it's stored as JSON so the schema stays the same for every action type.
    """
    df = _load_or_init()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action_type": action_type,
        "source_filename": source_filename,
        "sample_rate": int(sample_rate) if sample_rate else 0,
        "duration_sec": round(float(duration_sec), 3) if duration_sec else 0.0,
        "details": json.dumps(details or {}, default=str),
        "output_filename": output_filename,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return df


def describe_action(action_type: str, details: dict) -> str:
    """One human-readable line summarizing an action, for the UI."""
    d = details or {}
    if action_type == "load":
        return "Loaded the clip into the studio"
    if action_type == "frequency_edit":
        op = d.get("operation", "?")
        low, high = d.get("low"), d.get("high")
        gain = d.get("gain")
        band = f"{low:,.0f}–{high:,.0f} Hz" if low is not None and high is not None else "?"
        extra = f" (gain ×{gain:.2f})" if gain not in (None, 1.0) else ""
        return f"{op.capitalize()}d band {band}{extra}"
    if action_type == "noise_add":
        nt = d.get("noise_type", "?")
        snr = d.get("snr_db")
        return f"Added '{nt}' test noise" + (f" at {snr:.0f} dB SNR" if snr is not None else "")
    if action_type == "noise_remove":
        method = d.get("profile_method", "auto")
        strength = d.get("reduction_strength")
        transients = " + click suppression" if d.get("suppress_transients") else ""
        return f"Removed noise ({method} profile, strength {strength}){transients}"
    if action_type == "mix":
        other = d.get("mixed_with", "?")
        gain = d.get("gain")
        return f"Mixed in '{other}'" + (f" at gain ×{gain:.2f}" if gain is not None else "")
    if action_type == "export":
        fmt = d.get("format", "wav")
        return f"Exported final audio as .{fmt}"
    return action_type


def get_log_for_file(filename: str) -> pd.DataFrame:
    """All log rows whose source_filename matches (useful per-clip history)."""
    df = _load_or_init()
    return df[df["source_filename"] == filename]


def clear_log() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()