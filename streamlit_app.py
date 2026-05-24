from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import streamlit as st

from highlighter import (
    TranscriptionError,
    VideoEditingError,
    analyze_video,
    cut_highlight_reel,
    highlight_to_dict,
    load_transcript_segments,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "highlight_model.joblib"

for folder in (UPLOAD_DIR, TRANSCRIPT_DIR, OUTPUT_DIR):
    folder.mkdir(exist_ok=True)


def main() -> None:
    st.set_page_config(
        page_title="AI Video Highlighter",
        page_icon=":movie_camera:",
        layout="wide",
    )
    _apply_theme()

    _render_header()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Analyze</div>', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Video",
            type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
        )
        query = st.text_input(
            "Topic or question",
            placeholder="gradient descent or exam tips",
        )
        transcript_file = st.file_uploader(
            "Transcript file",
            type=["srt", "vtt", "json"],
            help="Optional. Without this, the app transcribes the video with local OpenAI Whisper.",
        )
        whisper_model = st.selectbox(
            "Whisper model",
            options=["tiny.en", "base.en", "small.en"],
            index=0,
            help="tiny.en is fastest for hosted demos; base.en and small.en are usually more accurate.",
        )
        max_results = st.slider("Highlights", min_value=3, max_value=15, value=8)
        use_trained_model = st.checkbox(
            "Use trained model",
            value=MODEL_PATH.exists(),
            disabled=not MODEL_PATH.exists(),
            help="Train one with train_highlight_model.py.",
        )
        make_reel = st.checkbox("Create jumpcut edit", value=True)
        padding_seconds = st.slider(
            "Context around each moment",
            min_value=1,
            max_value=12,
            value=4,
            help="Extra seconds kept before and after each important moment.",
        )
        analyze_clicked = st.button("Analyze video", type="primary", use_container_width=True)

    if not video_file:
        _render_empty_state()
        return

    video_bytes = video_file.getvalue()
    _render_video_summary(
        filename=video_file.name,
        file_size=len(video_bytes),
        whisper_model=whisper_model,
        transcript_uploaded=bool(transcript_file),
        make_reel=make_reel,
    )
    video_digest = hashlib.sha1(video_bytes).hexdigest()
    if st.session_state.get("current_video_digest") != video_digest:
        _clear_previous_analysis()
        st.session_state.current_video_digest = video_digest
        st.session_state.selected_start = 0

    video_path = _save_uploaded_bytes(video_file.name, video_bytes, UPLOAD_DIR, "video")

    if "selected_start" not in st.session_state:
        st.session_state.selected_start = 0

    if not analyze_clicked and "last_highlights" not in st.session_state:
        video_col, result_col = st.columns([1.2, 1], gap="large")
        with video_col:
            _render_videos(video_bytes)
        with result_col:
            st.subheader("Highlights")
            st.info("Pick your settings, then click Analyze video.")
        return

    analysis_failed = False
    if analyze_clicked:
        progress_bar, progress_note = _render_progress_panel()
        try:
            _update_progress(progress_bar, progress_note, 8, "Preparing video")
            transcript_path = None
            provided_segments = None
            if transcript_file:
                _update_progress(progress_bar, progress_note, 24, "Reading transcript file")
                transcript_path = _save_uploaded_bytes(
                    transcript_file.name,
                    transcript_file.getvalue(),
                    TRANSCRIPT_DIR,
                    "transcript",
                )
                provided_segments = load_transcript_segments(transcript_path)
                _update_progress(progress_bar, progress_note, 45, "Transcript loaded")
            else:
                _update_progress(
                    progress_bar,
                    progress_note,
                    24,
                    f"Transcribing with Whisper {whisper_model}",
                )

            highlights = analyze_video(
                video_path,
                query=query,
                max_results=max_results,
                provided_segments=provided_segments,
                transcript_cache_dir=TRANSCRIPT_DIR,
                highlight_model_path=MODEL_PATH if use_trained_model else None,
                whisper_model=whisper_model,
            )
            _update_progress(progress_bar, progress_note, 68, "Ranking key moments")

            st.session_state.last_video_path = str(video_path)
            st.session_state.last_video_name = video_file.name
            st.session_state.last_query = query
            st.session_state.last_highlights = [highlight_to_dict(item) for item in highlights]
            st.session_state.last_jumpcut_path = None
            st.session_state.jumpcut_warning = None

            if make_reel and highlights:
                _update_progress(progress_bar, progress_note, 82, "Rendering jumpcut edit")
                try:
                    reel_path = OUTPUT_DIR / f"{video_path.stem}_jumpcut.mp4"
                    output = cut_highlight_reel(
                        video_path,
                        highlights,
                        output_path=reel_path,
                        padding_seconds=padding_seconds,
                        max_clips=max_results,
                    )
                    st.session_state.last_jumpcut_path = str(output) if output else None
                except VideoEditingError as exc:
                    st.session_state.jumpcut_warning = str(exc)

            _update_progress(progress_bar, progress_note, 100, "Analysis complete")
            progress_note.success("Analysis complete. Results are ready below.")
        except TranscriptionError as exc:
            analysis_failed = True
            _clear_previous_analysis(keep_current_video=True)
            _update_progress(progress_bar, progress_note, 100, "Analysis stopped")
            _render_analysis_error(str(exc), include_key_help=True)
        except ValueError as exc:
            analysis_failed = True
            _clear_previous_analysis(keep_current_video=True)
            _update_progress(progress_bar, progress_note, 100, "Analysis stopped")
            _render_analysis_error(str(exc))
        except RuntimeError as exc:
            analysis_failed = True
            _clear_previous_analysis(keep_current_video=True)
            _update_progress(progress_bar, progress_note, 100, "Analysis stopped")
            _render_analysis_error(str(exc))
        except Exception as exc:
            analysis_failed = True
            _clear_previous_analysis(keep_current_video=True)
            _update_progress(progress_bar, progress_note, 100, "Analysis stopped")
            _render_analysis_error(
                "Something unexpected happened while analyzing this video.",
                details=str(exc),
            )

    video_col, result_col = st.columns([1.2, 1], gap="large")
    with video_col:
        _render_videos(video_bytes)
    with result_col:
        if analysis_failed:
            st.subheader("Highlights")
            st.info("Fix the error, then run the analysis again.")
        else:
            _render_highlights()


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f6f7f4;
            --surface: #ffffff;
            --surface-muted: #eef3ef;
            --ink: #17201b;
            --muted: #617066;
            --line: #d8e0da;
            --accent: #0f766e;
            --accent-dark: #115e59;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(246, 247, 244, 0.96), rgba(246, 247, 244, 1) 260px),
                var(--page-bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.75rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #f0f4f1;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.85rem;
        }

        .app-header {
            border-bottom: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            gap: 1.25rem;
            margin-bottom: 1.35rem;
            padding: 0.25rem 0 1.1rem;
        }

        .app-header h1 {
            color: var(--ink);
            font-size: 3rem;
            font-weight: 780;
            letter-spacing: 0;
            line-height: 1.02;
            margin: 0.1rem 0 0.45rem;
        }

        .app-header p {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.55;
            margin: 0;
            max-width: 680px;
        }

        .eyebrow {
            color: var(--accent-dark);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .header-pills {
            align-content: start;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: flex-end;
            min-width: 220px;
            padding-top: 0.35rem;
        }

        .header-pills span {
            background: var(--surface-muted);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--accent-dark);
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.42rem 0.62rem;
            white-space: nowrap;
        }

        .sidebar-title {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 760;
            margin: 0.2rem 0 0.35rem;
        }

        .summary-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0 0 1.1rem;
        }

        .summary-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            min-height: 86px;
        }

        .summary-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0;
            margin-bottom: 0.28rem;
            text-transform: uppercase;
        }

        .summary-value {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 740;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .summary-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }

        .progress-title {
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 760;
            margin-bottom: 0.55rem;
        }

        .section-kicker {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0;
            margin-bottom: 0.15rem;
            text-transform: uppercase;
        }

        .empty-panel {
            background: var(--surface);
            border: 1px dashed #b8c5bd;
            border-radius: 8px;
            padding: 1.2rem;
        }

        .moment-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 760;
            line-height: 1.35;
            margin-bottom: 0.35rem;
        }

        .moment-summary {
            color: #2e3a33;
            font-size: 0.94rem;
            line-height: 1.48;
            margin-bottom: 0.6rem;
        }

        .moment-meta {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.45;
            margin-top: 0.42rem;
        }

        .score-track {
            background: #e4ebe6;
            border-radius: 8px;
            height: 6px;
            overflow: hidden;
            width: 100%;
        }

        .score-fill {
            background: var(--accent);
            height: 100%;
        }

        div[data-testid="stProgress"] > div > div > div {
            background-color: var(--accent);
        }

        .stButton > button[kind="primary"] {
            background: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 8px;
            color: #ffffff;
            font-weight: 760;
        }

        .stButton > button[kind="primary"]:hover {
            background: var(--accent-dark);
            border-color: var(--accent-dark);
            color: #ffffff;
        }

        .stDownloadButton > button,
        .stButton > button {
            border-radius: 8px;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.72);
            border-color: #b8c5bd;
            border-radius: 8px;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
        }

        @media (max-width: 760px) {
            .app-header {
                display: block;
            }
            .header-pills {
                justify-content: flex-start;
                margin-top: 0.9rem;
            }
            .summary-grid {
                grid-template-columns: 1fr;
            }
            .app-header h1 {
                font-size: 2.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="eyebrow">Local Whisper video workflow</div>
                <h1>AI Video Highlighter</h1>
                <p>Upload a lecture, find the strongest timestamped moments, and render a cleaner jumpcut edit for review.</p>
            </div>
            <div class="header-pills">
                <span>Whisper transcript</span>
                <span>Ranked moments</span>
                <span>Jumpcut MP4</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_video_summary(
    filename: str,
    file_size: int,
    whisper_model: str,
    transcript_uploaded: bool,
    make_reel: bool,
) -> None:
    transcript_label = "Uploaded transcript" if transcript_uploaded else f"Whisper {whisper_model}"
    reel_label = "Enabled" if make_reel else "Off"
    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-label">Source</div>
                <div class="summary-value">{_escape_html(filename)}</div>
                <div class="summary-note">{_format_bytes(file_size)}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Transcript</div>
                <div class="summary-value">{_escape_html(transcript_label)}</div>
                <div class="summary-note">Timestamped segments</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Output</div>
                <div class="summary-value">Jumpcut {reel_label}</div>
                <div class="summary-note">Highlights below full video</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Mode</div>
                <div class="summary-value">Local only</div>
                <div class="summary-note">No API key required</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_progress_panel():
    panel = st.container(border=True)
    with panel:
        st.markdown('<div class="progress-title">Analysis progress</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0, text="Waiting to start")
        progress_note = st.empty()
    return progress_bar, progress_note


def _update_progress(progress_bar, progress_note, value: int, label: str) -> None:
    progress_bar.progress(value, text=label)
    progress_note.caption(label)


def _render_empty_state() -> None:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown('<div class="section-kicker">Video</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="empty-panel">Upload a video to get started.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-kicker">Highlights</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="empty-panel">Your timestamped moments will appear here.</div>',
            unsafe_allow_html=True,
        )


def _render_videos(video_bytes: bytes) -> None:
    st.markdown('<div class="section-kicker">Full lecture</div>', unsafe_allow_html=True)
    st.video(video_bytes, start_time=int(st.session_state.selected_start))

    jumpcut_path = st.session_state.get("last_jumpcut_path")
    if jumpcut_path and Path(jumpcut_path).exists():
        st.markdown('<div class="section-kicker">Jumpcut edit</div>', unsafe_allow_html=True)
        jumpcut_bytes = Path(jumpcut_path).read_bytes()
        st.video(jumpcut_bytes)
        st.download_button(
            "Download jumpcut edit",
            data=jumpcut_bytes,
            file_name=Path(jumpcut_path).name,
            mime="video/mp4",
            use_container_width=True,
        )

    jumpcut_warning = st.session_state.get("jumpcut_warning")
    if jumpcut_warning:
        st.warning(f"Highlights were found, but the jumpcut edit could not be created. {jumpcut_warning}")


def _render_highlights() -> None:
    st.markdown('<div class="section-kicker">Highlights</div>', unsafe_allow_html=True)
    highlights = st.session_state.get("last_highlights", [])
    if not highlights:
        st.warning("No highlights found.")
        return

    for index, highlight in enumerate(highlights, start=1):
        with st.container(border=True):
            score_percent = _score_percent(highlight["score"])
            st.markdown(
                f"""
                <div class="moment-title">{index}. {_escape_html(str(highlight["title"]))}</div>
                <div class="moment-summary">{_escape_html(str(highlight["summary"]))}</div>
                <div class="score-track">
                    <div class="score-fill" style="width: {score_percent}%"></div>
                </div>
                <div class="moment-meta">
                    {_escape_html(str(highlight["time_label"]))} | score {_escape_html(str(highlight["score_display"]))} | {_escape_html(str(highlight["reason"]))}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Jump to {highlight['start_display']}",
                key=f"jump_{index}_{highlight['start']}",
                use_container_width=True,
            ):
                st.session_state.selected_start = int(float(highlight["start"]))
                st.rerun()


def _render_key_help() -> None:
    st.info(
        "Local Whisper does not need an API key, but it does need the openai-whisper package "
        "and FFmpeg. You can also upload an SRT, VTT, or timestamped JSON transcript."
    )


def _render_analysis_error(
    message: str,
    include_key_help: bool = False,
    details: str | None = None,
) -> None:
    st.error(message)
    if include_key_help:
        _render_key_help()
    if details:
        with st.expander("Error details"):
            st.code(details)


def _save_uploaded_bytes(
    filename: str,
    data: bytes,
    folder: Path,
    key_prefix: str,
) -> Path:
    digest = hashlib.sha1(data).hexdigest()
    state_key = f"{key_prefix}_{digest}_path"
    existing_path = st.session_state.get(state_key)
    if existing_path and Path(existing_path).exists():
        return Path(existing_path)

    safe_name = Path(filename).name.replace(" ", "_")
    destination = folder / f"{uuid.uuid4().hex}_{safe_name}"
    destination.write_bytes(data)
    st.session_state[state_key] = str(destination)
    return destination


def _format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size} B"


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _score_percent(score: object) -> int:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return 12
    return int(max(12, min(100, numeric_score / 8 * 100)))


def _clear_previous_analysis(keep_current_video: bool = False) -> None:
    keys = [
        "last_video_path",
        "last_video_name",
        "last_query",
        "last_highlights",
        "last_jumpcut_path",
        "jumpcut_warning",
    ]
    if not keep_current_video:
        keys.append("current_video_digest")

    for key in keys:
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
