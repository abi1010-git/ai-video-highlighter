from __future__ import annotations

import uuid
import os
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from highlighter import (
    MissingApiKey,
    TranscriptionError,
    analyze_video,
    highlight_to_dict,
    load_transcript_segments,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
MODEL_PATH = BASE_DIR / "models" / "highlight_model.joblib"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ALLOWED_TRANSCRIPT_EXTENSIONS = {".srt", ".vtt", ".json"}

UPLOAD_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze():
    video_file = request.files.get("video")
    transcript_file = request.files.get("transcript")
    query = request.form.get("query", "").strip()
    max_results = _safe_int(request.form.get("max_results"), default=8, minimum=1, maximum=15)

    if not video_file or not video_file.filename:
        return render_template("index.html", error="Choose a video file first.")

    video_path = _save_upload(video_file, UPLOAD_DIR, ALLOWED_VIDEO_EXTENSIONS)
    transcript_path = None
    provided_segments = None

    try:
        if transcript_file and transcript_file.filename:
            transcript_path = _save_upload(
                transcript_file,
                TRANSCRIPT_DIR,
                ALLOWED_TRANSCRIPT_EXTENSIONS,
            )
            provided_segments = load_transcript_segments(transcript_path)

        highlights = analyze_video(
            video_path,
            query=query,
            max_results=max_results,
            provided_segments=provided_segments,
            transcript_cache_dir=TRANSCRIPT_DIR,
            highlight_model_path=MODEL_PATH if MODEL_PATH.exists() else None,
        )
    except (MissingApiKey, RuntimeError, TranscriptionError, ValueError) as exc:
        return render_template(
            "index.html",
            error=str(exc),
            video_url=url_for("uploaded_file", filename=video_path.name),
            query=query,
        )

    return render_template(
        "index.html",
        video_url=url_for("uploaded_file", filename=video_path.name),
        query=query,
        highlights=[highlight_to_dict(item) for item in highlights],
        transcript_name=transcript_path.name if transcript_path else None,
    )


@app.get("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


def _save_upload(file_storage, folder: Path, allowed_extensions: set[str]) -> Path:
    original_name = secure_filename(file_storage.filename or "")
    suffix = Path(original_name).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Unsupported file type. Use one of: {allowed}")

    filename = f"{uuid.uuid4().hex}_{original_name}"
    destination = folder / filename
    file_storage.save(destination)
    return destination


def _safe_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value or default)
    except ValueError:
        return default
    return max(minimum, min(maximum, number))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
