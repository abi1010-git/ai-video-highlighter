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

    st.title("AI Video Highlighter")
    st.caption("Upload a lecture or video, search for a topic, and jump to the most relevant moments.")

    with st.sidebar:
        st.header("Analyze")
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
        with st.spinner("Finding the most relevant moments..."):
            try:
                transcript_path = None
                provided_segments = None
                if transcript_file:
                    transcript_path = _save_uploaded_bytes(
                        transcript_file.name,
                        transcript_file.getvalue(),
                        TRANSCRIPT_DIR,
                        "transcript",
                    )
                    provided_segments = load_transcript_segments(transcript_path)

                highlights = analyze_video(
                    video_path,
                    query=query,
                    max_results=max_results,
                    provided_segments=provided_segments,
                    transcript_cache_dir=TRANSCRIPT_DIR,
                    highlight_model_path=MODEL_PATH if use_trained_model else None,
                    whisper_model=whisper_model,
                )

                st.session_state.last_video_path = str(video_path)
                st.session_state.last_video_name = video_file.name
                st.session_state.last_query = query
                st.session_state.last_highlights = [highlight_to_dict(item) for item in highlights]
                st.session_state.last_jumpcut_path = None
                st.session_state.jumpcut_warning = None

                if make_reel and highlights:
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
            except TranscriptionError as exc:
                analysis_failed = True
                _clear_previous_analysis(keep_current_video=True)
                _render_analysis_error(str(exc), include_key_help=True)
            except ValueError as exc:
                analysis_failed = True
                _clear_previous_analysis(keep_current_video=True)
                _render_analysis_error(str(exc))
            except RuntimeError as exc:
                analysis_failed = True
                _clear_previous_analysis(keep_current_video=True)
                _render_analysis_error(str(exc))
            except Exception as exc:
                analysis_failed = True
                _clear_previous_analysis(keep_current_video=True)
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


def _render_empty_state() -> None:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Video")
        st.info("Upload a video to get started.")
    with right:
        st.subheader("Highlights")
        st.write("Your timestamped moments will appear here.")


def _render_videos(video_bytes: bytes) -> None:
    st.subheader("Full lecture")
    st.video(video_bytes, start_time=int(st.session_state.selected_start))

    jumpcut_path = st.session_state.get("last_jumpcut_path")
    if jumpcut_path and Path(jumpcut_path).exists():
        st.subheader("Jumpcut edit")
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
    st.subheader("Highlights")
    highlights = st.session_state.get("last_highlights", [])
    if not highlights:
        st.warning("No highlights found.")
        return

    for index, highlight in enumerate(highlights, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {highlight['title']}**")
            st.write(highlight["summary"])
            st.caption(
                f"{highlight['time_label']} | score {highlight['score_display']} | {highlight['reason']}"
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
