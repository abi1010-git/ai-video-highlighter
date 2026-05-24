from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


class TranscriptionError(RuntimeError):
    """Raised when a video cannot be transcribed."""


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Highlight:
    start: float
    end: float
    score: float
    title: str
    summary: str
    reason: str


@dataclass(frozen=True)
class ClipRange:
    start: float
    end: float


class VideoEditingError(RuntimeError):
    """Raised when the app cannot create the jumpcut edit."""


IMPORTANT_PHRASES: tuple[tuple[str, float], ...] = (
    ("important", 2.2),
    ("key idea", 2.4),
    ("main idea", 2.4),
    ("remember", 2.0),
    ("exam", 2.4),
    ("quiz", 2.0),
    ("definition", 2.0),
    ("this means", 1.8),
    ("in summary", 2.4),
    ("to summarize", 2.4),
    ("takeaway", 2.4),
    ("for example", 1.5),
    ("notice that", 1.6),
    ("the reason", 1.6),
    ("because", 1.1),
    ("therefore", 1.4),
    ("so we can", 1.2),
)

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "get",
    "has",
    "have",
    "how",
    "into",
    "its",
    "just",
    "like",
    "more",
    "most",
    "not",
    "now",
    "our",
    "out",
    "over",
    "see",
    "she",
    "should",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "was",
    "way",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def highlight_to_dict(highlight: Highlight) -> dict[str, object]:
    data = asdict(highlight)
    data["time_label"] = (
        f"{seconds_to_timestamp(highlight.start)} - {seconds_to_timestamp(highlight.end)}"
    )
    data["start_display"] = seconds_to_timestamp(highlight.start)
    data["end_display"] = seconds_to_timestamp(highlight.end)
    data["score_display"] = f"{highlight.score:.2f}"
    return data


def analyze_video(
    video_path: str | Path,
    query: str = "",
    max_results: int = 8,
    transcript_path: str | Path | None = None,
    provided_segments: list[TranscriptSegment] | None = None,
    transcript_cache_dir: str | Path = "transcripts",
    force_transcribe: bool = False,
    highlight_model_path: str | Path | None = None,
    whisper_model: str | None = None,
) -> list[Highlight]:
    video_path = Path(video_path)
    if provided_segments is not None:
        segments = provided_segments
    elif transcript_path:
        segments = load_transcript_segments(transcript_path)
    else:
        segments = transcribe_video(
            video_path,
            cache_dir=Path(transcript_cache_dir),
            force=force_transcribe,
            whisper_model=whisper_model,
        )

    chunks = chunk_segments(segments)
    return rank_chunks(
        chunks,
        query=query,
        max_results=max_results,
        highlight_model_path=highlight_model_path,
    )


def transcribe_video(
    video_path: str | Path,
    cache_dir: str | Path = "transcripts",
    force: bool = False,
    whisper_model: str | None = None,
) -> list[TranscriptSegment]:
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = whisper_model or os.getenv("WHISPER_MODEL") or "tiny.en"
    cache_label = _safe_cache_label(f"local-whisper-{whisper_model}")
    cache_file = cache_dir / f"{video_path.stem}.{cache_label}.transcript.json"

    if cache_file.exists() and not force:
        return load_transcript_segments(cache_file)

    segments = transcribe_video_with_local_whisper(video_path, model_name=whisper_model)
    cache_file.write_text(
        json.dumps({"segments": [asdict(segment) for segment in segments]}, indent=2),
        encoding="utf-8",
    )
    return segments


def transcribe_video_with_local_whisper(
    video_path: str | Path,
    model_name: str = "tiny.en",
) -> list[TranscriptSegment]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise TranscriptionError(f"Video file not found: {video_path}")

    try:
        model = _load_local_whisper_model(model_name)
    except ImportError as exc:
        raise TranscriptionError(
            "Install OpenAI Whisper with: python -m pip install -U openai-whisper"
        ) from exc
    except Exception as exc:
        raise TranscriptionError(f"Could not load local Whisper model: {exc}") from exc

    _ensure_ffmpeg_on_path()

    try:
        result = model.transcribe(
            str(video_path),
            fp16=False,
            verbose=False,
            language="en" if model_name.endswith(".en") else None,
        )
    except Exception as exc:
        raise TranscriptionError(f"Local Whisper transcription failed: {exc}") from exc

    return _segments_from_data(result)


@lru_cache(maxsize=4)
def _load_local_whisper_model(model_name: str):
    import whisper

    return whisper.load_model(model_name)


def _ensure_ffmpeg_on_path() -> None:
    try:
        import imageio_ffmpeg

        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return

    if not ffmpeg_path.exists():
        return

    if ffmpeg_path.name.lower() not in {"ffmpeg", "ffmpeg.exe"}:
        shim_dir = Path(tempfile.gettempdir()) / "ai_video_highlighter_ffmpeg"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not shim_path.exists():
            shutil.copy2(ffmpeg_path, shim_path)
        ffmpeg_path = shim_path

    ffmpeg_dir = str(ffmpeg_path.parent)
    current_paths = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_dir not in current_paths:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def load_transcript_segments(transcript_path: str | Path) -> list[TranscriptSegment]:
    transcript_path = Path(transcript_path)
    suffix = transcript_path.suffix.lower()
    text = transcript_path.read_text(encoding="utf-8-sig")

    if suffix == ".json":
        return _segments_from_data(json.loads(text))
    if suffix in {".srt", ".vtt"}:
        return _segments_from_caption_text(text)

    raise ValueError("Transcript must be .srt, .vtt, or .json with timestamped segments.")


def chunk_segments(
    segments: list[TranscriptSegment],
    target_seconds: float = 55.0,
) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    chunk_start: float | None = None

    for segment in segments:
        clean_text = segment.text.strip()
        if not clean_text:
            continue

        if chunk_start is None:
            chunk_start = segment.start

        should_close = current and segment.end - chunk_start >= target_seconds
        if should_close:
            chunks.append(_make_chunk(current))
            current = []
            chunk_start = segment.start

        current.append(segment)

    if current:
        chunks.append(_make_chunk(current))

    return chunks


def rank_chunks(
    chunks: list[TranscriptChunk],
    query: str = "",
    max_results: int = 8,
    highlight_model_path: str | Path | None = None,
) -> list[Highlight]:
    query_terms = set(_tokenize(query))
    all_terms = Counter(
        term for chunk in chunks for term in _tokenize(chunk.text) if term not in query_terms
    )
    common_terms = {term for term, _count in all_terms.most_common(35)}
    model_probabilities = _load_model_probabilities(chunks, highlight_model_path)

    scored: list[Highlight] = []
    for index, chunk in enumerate(chunks):
        chunk_terms = _tokenize(chunk.text)
        if not chunk_terms:
            continue

        phrase_matches = _phrase_matches(chunk.text)
        phrase_score = sum(weight for _phrase, weight in phrase_matches)
        unique_terms = set(chunk_terms)

        if query_terms:
            overlap = query_terms & unique_terms
            query_score = len(overlap) / math.sqrt(max(1, len(query_terms))) * 6.0
            density_score = sum(chunk_terms.count(term) for term in overlap) * 0.45
            score = query_score + density_score + phrase_score * 0.35
        else:
            repeated_signal = sum(1 for term in chunk_terms if term in common_terms)
            repeated_score = min(3.5, repeated_signal / max(1, len(chunk_terms)) * 8.0)
            length_score = min(1.2, len(chunk_terms) / 85.0)
            score = phrase_score + repeated_score + length_score

        if score <= 0:
            score = min(1.0, len(unique_terms) / 60.0)

        reason = _make_reason(query_terms, unique_terms, phrase_matches)
        if model_probabilities is not None:
            model_probability = model_probabilities[index]
            score = score * 0.65 + model_probability * 8.0 * 0.35
            reason = f"{reason}; trained model {model_probability:.0%}"

        scored.append(
            Highlight(
                start=chunk.start,
                end=chunk.end,
                score=round(score, 3),
                title=_make_title(chunk.text, query_terms),
                summary=_summarize(chunk.text),
                reason=reason,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return sorted(scored[:max_results], key=lambda item: item.start)


def cut_highlight_reel(
    video_path: str | Path,
    highlights: list[Highlight],
    output_path: str | Path = "outputs/highlights.mp4",
    padding_seconds: float = 4.0,
    max_clips: int = 10,
    merge_gap_seconds: float = 6.0,
) -> Path | None:
    if not highlights:
        return None

    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)
    if not video_path.exists():
        raise VideoEditingError(f"Video file not found: {video_path}")

    try:
        video = VideoFileClip(str(video_path))
    except Exception as exc:
        raise VideoEditingError(f"Could not open video for editing: {exc}") from exc

    clips = []
    try:
        ranges = _clip_ranges_from_highlights(
            highlights,
            duration=float(video.duration),
            padding_seconds=padding_seconds,
            max_clips=max_clips,
            merge_gap_seconds=merge_gap_seconds,
        )
        for clip_range in ranges:
            clips.append(_subclip(video, clip_range.start, clip_range.end))

        if not clips:
            return None

        final = concatenate_videoclips(clips, method="compose")
        try:
            final.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
        except Exception as exc:
            raise VideoEditingError(f"Could not write jumpcut edit: {exc}") from exc
        finally:
            final.close()
    finally:
        for clip in clips:
            clip.close()
        video.close()

    return output_path


def jumpcut_duration_estimate(
    highlights: list[Highlight],
    source_duration: float,
    padding_seconds: float = 4.0,
    max_clips: int = 10,
    merge_gap_seconds: float = 6.0,
) -> float:
    ranges = _clip_ranges_from_highlights(
        highlights,
        duration=source_duration,
        padding_seconds=padding_seconds,
        max_clips=max_clips,
        merge_gap_seconds=merge_gap_seconds,
    )
    return sum(clip_range.end - clip_range.start for clip_range in ranges)


def _make_chunk(segments: list[TranscriptSegment]) -> TranscriptChunk:
    return TranscriptChunk(
        start=segments[0].start,
        end=segments[-1].end,
        text=" ".join(segment.text.strip() for segment in segments),
    )


def _segments_from_data(data: object) -> list[TranscriptSegment]:
    if isinstance(data, list):
        raw_segments = data
    elif isinstance(data, dict):
        raw_segments = data.get("segments") or data.get("transcript_segments") or []
    else:
        raw_segments = []

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if end > start and text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))

    if not segments:
        raise TranscriptionError("No timestamped transcript segments were found.")
    return segments


def _segments_from_caption_text(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized)

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper().startswith(("WEBVTT", "NOTE")):
            continue

        for index, line in enumerate(lines):
            if "-->" not in line:
                continue
            start_text, end_text = line.split("-->", 1)
            start = _parse_caption_time(start_text.strip())
            end = _parse_caption_time(end_text.strip().split()[0])
            caption_text = " ".join(lines[index + 1 :])
            caption_text = re.sub(r"<[^>]+>", "", caption_text).strip()
            if caption_text and end > start:
                segments.append(TranscriptSegment(start=start, end=end, text=caption_text))
            break

    if not segments:
        raise TranscriptionError("No timestamped captions were found in the transcript.")
    return segments


def _parse_caption_time(value: str) -> float:
    value = value.replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"Invalid caption timestamp: {value}")
    return hours * 3600 + minutes * 60 + seconds


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
    return [word for word in words if len(word) > 2 and word not in STOPWORDS]


def _phrase_matches(text: str) -> list[tuple[str, float]]:
    lower_text = text.lower()
    matches: list[tuple[str, float]] = []
    for phrase, weight in IMPORTANT_PHRASES:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pattern, lower_text):
            matches.append((phrase, weight))
    return matches


def _make_title(text: str, query_terms: set[str]) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return "Relevant moment"

    def sentence_score(sentence: str) -> float:
        terms = set(_tokenize(sentence))
        query_score = len(query_terms & terms) * 3.0
        phrase_score = sum(weight for _phrase, weight in _phrase_matches(sentence))
        return query_score + phrase_score + min(1.0, len(terms) / 18.0)

    best = max(sentences, key=sentence_score)
    return _truncate(best, 92)


def _summarize(text: str) -> str:
    return _truncate(" ".join(text.split()), 220)


def _make_reason(
    query_terms: set[str],
    chunk_terms: set[str],
    phrase_matches: list[tuple[str, float]],
) -> str:
    reasons: list[str] = []
    if query_terms:
        matches = sorted(query_terms & chunk_terms)
        if matches:
            reasons.append("matches " + ", ".join(matches[:5]))
    if phrase_matches:
        phrases = [phrase for phrase, _weight in phrase_matches[:3]]
        reasons.append("contains cue " + ", ".join(phrases))
    if not reasons:
        reasons.append("dense section with repeated lecture terms")
    return "; ".join(reasons)


def _truncate(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rsplit(" ", 1)[0] + "..."


def _safe_cache_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._-") or "transcript"


def _subclip(video, start: float, end: float):
    if hasattr(video, "subclipped"):
        return video.subclipped(start, end)
    return video.subclip(start, end)


def _clip_ranges_from_highlights(
    highlights: list[Highlight],
    duration: float,
    padding_seconds: float,
    max_clips: int,
    merge_gap_seconds: float,
) -> list[ClipRange]:
    if duration <= 0:
        return []

    selected = sorted(
        sorted(highlights, key=lambda item: item.score, reverse=True)[:max_clips],
        key=lambda item: item.start,
    )

    ranges: list[ClipRange] = []
    for highlight in selected:
        start = max(0.0, highlight.start - padding_seconds)
        end = min(duration, highlight.end + padding_seconds)
        if end - start < 1:
            continue

        if ranges and start <= ranges[-1].end + merge_gap_seconds:
            ranges[-1] = ClipRange(start=ranges[-1].start, end=max(ranges[-1].end, end))
        else:
            ranges.append(ClipRange(start=start, end=end))

    return ranges


def _load_model_probabilities(
    chunks: list[TranscriptChunk],
    highlight_model_path: str | Path | None,
) -> list[float] | None:
    if not highlight_model_path:
        return None

    model_path = Path(highlight_model_path)
    if not model_path.exists():
        raise ValueError(f"Highlight model not found: {model_path}")

    try:
        from highlight_model import predict_highlight_probabilities

        return predict_highlight_probabilities(chunks, model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not use highlight model: {exc}") from exc
