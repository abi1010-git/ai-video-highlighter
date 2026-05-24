from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from highlighter import (
    IMPORTANT_PHRASES,
    TranscriptChunk,
    TranscriptSegment,
    chunk_segments,
    load_transcript_segments,
)


TIMESTAMPED_EXTENSIONS = {".srt", ".vtt", ".json"}
TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm"}
MODEL_VERSION = "highlight-logreg-v1"


@dataclass(frozen=True)
class TrainingExample:
    source: str
    start: float
    end: float
    text: str
    label: int
    weak_score: float


def transcript_files(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    allowed = TIMESTAMPED_EXTENSIONS | TEXT_EXTENSIONS
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in allowed)


def load_chunks(path: str | Path) -> list[TranscriptChunk]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIMESTAMPED_EXTENSIONS:
        return chunk_segments(load_transcript_segments(path))

    if suffix in TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        if suffix in {".html", ".htm"}:
            text = _html_to_text(text)
        return _plain_text_chunks(text)

    raise ValueError(f"Unsupported transcript file: {path}")


def build_weak_examples(paths: list[Path]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []

    for path in paths:
        chunks = load_chunks(path)
        if len(chunks) < 4:
            continue

        scored = [
            (index, chunk, _weak_highlight_score(chunk, index, len(chunks)))
            for index, chunk in enumerate(chunks)
            if len(chunk.text.split()) >= 18
        ]
        if len(scored) < 4:
            continue

        scored_by_value = sorted(scored, key=lambda item: item[2])
        negative_count = max(1, len(scored_by_value) // 3)
        positive_count = max(1, len(scored_by_value) // 4)
        negative_indexes = {item[0] for item in scored_by_value[:negative_count]}
        positive_indexes = {item[0] for item in scored_by_value[-positive_count:]}

        for index, chunk, weak_score in scored:
            if index in positive_indexes:
                label = 1
            elif index in negative_indexes:
                label = 0
            else:
                continue

            examples.append(
                TrainingExample(
                    source=str(path),
                    start=chunk.start,
                    end=chunk.end,
                    text=chunk.text,
                    label=label,
                    weak_score=round(weak_score, 3),
                )
            )

    return examples


def train_highlight_model(
    input_dir: str | Path = "data/raw_transcripts",
    output_path: str | Path = "models/highlight_model.joblib",
    examples_path: str | Path | None = "data/training_examples.csv",
) -> dict[str, object]:
    paths = transcript_files(input_dir)
    if not paths:
        raise ValueError(f"No transcript files found in {input_dir}.")

    examples = build_weak_examples(paths)
    positive_count = sum(example.label for example in examples)
    negative_count = len(examples) - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            "Not enough positive and negative examples. Add more transcript files."
        )

    if examples_path:
        write_examples_csv(examples, examples_path)

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=12000,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    metrics: dict[str, object]
    if len(examples) >= 12 and positive_count >= 4 and negative_count >= 4:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(test_texts)
        metrics = {
            "accuracy": round(float(accuracy_score(test_labels, predictions)), 3),
            "report": classification_report(
                test_labels,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }
    else:
        pipeline.fit(texts, labels)
        metrics = {"accuracy": None, "report": None}

    pipeline.fit(texts, labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": MODEL_VERSION,
        "pipeline": pipeline,
        "metadata": {
            "transcript_count": len(paths),
            "example_count": len(examples),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sources": [str(path) for path in paths],
            "metrics": metrics,
        },
    }
    joblib.dump(bundle, output_path)
    return bundle["metadata"]


def predict_highlight_probabilities(
    chunks: list[TranscriptChunk],
    model_path: str | Path = "models/highlight_model.joblib",
) -> list[float]:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"] if isinstance(bundle, dict) else bundle
    probabilities = pipeline.predict_proba([chunk.text for chunk in chunks])
    return [float(row[1]) for row in probabilities]


def write_examples_csv(
    examples: list[TrainingExample],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source", "start", "end", "label", "weak_score", "text"],
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "source": example.source,
                    "start": example.start,
                    "end": example.end,
                    "label": example.label,
                    "weak_score": example.weak_score,
                    "text": example.text,
                }
            )


def _weak_highlight_score(
    chunk: TranscriptChunk,
    index: int,
    total_chunks: int,
) -> float:
    text = chunk.text.lower()
    phrase_score = 0.0
    for phrase, weight in IMPORTANT_PHRASES:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        if re.search(pattern, text):
            phrase_score += weight

    word_count = len(chunk.text.split())
    length_score = min(2.0, word_count / 90.0)
    question_score = min(1.2, text.count("?") * 0.4)
    structure_score = 0.0
    if re.search(r"\b(first|second|third|finally|next|notice|suppose)\b", text):
        structure_score += 0.8
    if re.search(r"\b(there are|we can see|this tells us|the point is)\b", text):
        structure_score += 0.8

    position = index / max(1, total_chunks - 1)
    position_score = 0.5 if 0.15 <= position <= 0.9 else 0.0
    return phrase_score + length_score + question_score + structure_score + position_score


def _plain_text_chunks(text: str, words_per_chunk: int = 140) -> list[TranscriptChunk]:
    clean = " ".join(text.split())
    words = clean.split()
    chunks: list[TranscriptChunk] = []

    for index in range(0, len(words), words_per_chunk):
        chunk_words = words[index : index + words_per_chunk]
        if len(chunk_words) < 18:
            continue
        start = float(index // words_per_chunk * 60)
        end = start + 60.0
        chunks.append(TranscriptChunk(start=start, end=end, text=" ".join(chunk_words)))

    return chunks


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        return main.get_text(" ", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)

