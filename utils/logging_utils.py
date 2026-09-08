"""
logging_utils.py
-----------------
Tracks metadata about every audio clip that passes through the app
(sample rate, duration, ...) and a history of the frequency cuts the
user applies, using a Pandas DataFrame persisted as a local CSV file.
No database / backend involved - just data/logs/session_log.csv
"""
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

from utils.storage import LOGS_DIR

LOG_PATH = LOGS_DIR / "session_log.csv"

COLUMNS = [
    "timestamp",
    "source_filename",
    "sample_rate",
    "duration_sec",
    "num_bands",
    "bands_applied",
    "processed_filename",
]


def _load_or_init() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=COLUMNS)


def load_log() -> pd.DataFrame:
    return _load_or_init()


def log_entry(source_filename: str, sample_rate: int, duration_sec: float,
              bands: list[dict], processed_filename: str = "") -> pd.DataFrame:
    """
    Append a new row describing this action (upload preview, or a set of
    frequency cuts that were applied) to the persistent log and return the
    updated DataFrame.
    """
    df = _load_or_init()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_filename": source_filename,
        "sample_rate": sample_rate,
        "duration_sec": round(float(duration_sec), 3),
        "num_bands": len(bands),
        "bands_applied": json.dumps(bands),
        "processed_filename": processed_filename,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return df


def clear_log() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
