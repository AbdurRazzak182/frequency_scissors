
from pathlib import Path
from datetime import datetime
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"

for _d in (UPLOADS_DIR, PROCESSED_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".aiff", ".wma"}


def list_uploaded_files():
    """Return sorted list of previously uploaded audio filenames (newest first)."""
    files = [f for f in UPLOADS_DIR.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.name for f in files]


def uploaded_file_path(filename: str) -> Path:
    return UPLOADS_DIR / filename


def save_uploaded_file(uploaded_file) -> Path:
    """
    Persist a Streamlit UploadedFile object to data/uploads.
    If a file with the same name already exists, a timestamp suffix is added
    so previously uploaded clips are never overwritten.
    """
    target = UPLOADS_DIR / uploaded_file.name
    if target.exists():
        stem, suffix = target.stem, target.suffix
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = UPLOADS_DIR / f"{stem}_{stamp}{suffix}"

    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def save_processed_audio(source_tmp_path: Path, original_name: str) -> Path:
    """Move/copy a freshly rendered processed clip into data/processed with a
    unique, traceable filename."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_name).stem
    target = PROCESSED_DIR / f"{stem}_processed_{stamp}.wav"
    shutil.copy(source_tmp_path, target)
    return target


def read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()
