from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st

from highlighter import (
    MissingApiKey,
    TranscriptionError,
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
    _load_streamlit_secret()

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
            help="Optional. Without this, the app uses your OpenAI API key to transcribe.",
        )
        max_results = st.slider("Highlights", min_value=3, max_value=15, value=8)
        use_trained_model = st.checkbox(
            "Use trained model",
            value=MODEL_PATH.exists(),
            disabled=not MODEL_PATH.exists(),
            help="Train one with train_highlight_model.py.",
        )
        make_reel = st.checkbox("Create highlight reel", value=False)
        analyze_clicked = st.button("Analyze video", type="primary", use_container_width=True)

    if not video_file:
        _render_empty_state()
        return

    video_path = _save_uploaded_file(video_file, UPLOAD_DIR)
    video_bytes = video_file.getvalue()

    if "selected_start" not in st.session_state:
        st.session_state.selected_start = 0

    video_col, result_col = st.columns([1.2, 1], gap="large")
    with video_col:
        st.subheader("Video")
        st.video(video_bytes, start_time=int(st.session_state.selected_start))

    if not analyze_clicked and "last_highlights" not in st.session_state:
        with result_col:
            st.subheader("Highlights")
            st.info("Pick your settings, then click Analyze video.")
        return

    if analyze_clicked:
        with st.spinner("Finding the most relevant moments..."):
            try:
                transcript_path = None
                provided_segments = None
                if transcript_file:
                    transcript_path = _save_uploaded_file(transcript_file, TRANSCRIPT_DIR)
                    provided_segments = load_transcript_segments(transcript_path)

                highlights = analyze_video(
                    video_path,
                    query=query,
                    max_results=max_results,
                    provided_segments=provided_segments,
                    transcript_cache_dir=TRANSCRIPT_DIR,
                    highlight_model_path=MODEL_PATH if use_trained_model else None,
                )

                st.session_state.last_video_path = str(video_path)
                st.session_state.last_video_name = video_file.name
                st.session_state.last_query = query
                st.session_state.last_highlights = [highlight_to_dict(item) for item in highlights]
                st.session_state.last_reel_path = None

                if make_reel and highlights:
                    reel_path = OUTPUT_DIR / f"{video_path.stem}_highlights.mp4"
                    output = cut_highlight_reel(video_path, highlights, output_path=reel_path)
                    st.session_state.last_reel_path = str(output) if output else None
            except (MissingApiKey, TranscriptionError, ValueError) as exc:
                st.error(str(exc))
                _render_key_help()
                return

    with result_col:
        _render_highlights()


def _render_empty_state() -> None:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Video")
        st.info("Upload a video to get started.")
    with right:
        st.subheader("Highlights")
        st.write("Your timestamped moments will appear here.")


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

    reel_path = st.session_state.get("last_reel_path")
    if reel_path and Path(reel_path).exists():
        st.divider()
        st.download_button(
            "Download highlight reel",
            data=Path(reel_path).read_bytes(),
            file_name=Path(reel_path).name,
            mime="video/mp4",
            use_container_width=True,
        )


def _render_key_help() -> None:
    st.info(
        "For automatic transcription, add OPENAI_API_KEY as a local .env value, "
        "a Streamlit secret, or a Render environment variable. You can also upload "
        "an SRT, VTT, or timestamped JSON transcript to avoid API usage."
    )


def _save_uploaded_file(uploaded_file, folder: Path) -> Path:
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    destination = folder / f"{uuid.uuid4().hex}_{safe_name}"
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def _load_streamlit_secret() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)


if __name__ == "__main__":
    main()
