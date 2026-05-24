from __future__ import annotations

import argparse
from pathlib import Path

from highlighter import (
    TranscriptionError,
    VideoEditingError,
    analyze_video,
    cut_highlight_reel,
    seconds_to_timestamp,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find relevant timestamped moments in a video."
    )
    parser.add_argument("video", help="Path to the video file.")
    parser.add_argument(
        "--query",
        default="",
        help='What to look for, for example "exam tips" or "gradient descent".',
    )
    parser.add_argument(
        "--transcript",
        help="Optional timestamped transcript file (.srt, .vtt, or .json).",
    )
    parser.add_argument(
        "--whisper-model",
        default="tiny.en",
        help="Local Whisper model to use, for example tiny.en, base.en, or small.en.",
    )
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument(
        "--make-reel",
        action="store_true",
        help="Also export a jumpcut edit that skips low-importance sections.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=4.0,
        help="Extra seconds before and after each selected moment in the jumpcut.",
    )
    parser.add_argument(
        "--model",
        help="Optional trained highlight model, for example models/highlight_model.joblib.",
    )
    parser.add_argument(
        "--output",
        default="outputs/highlights.mp4",
        help="Output path when --make-reel is used.",
    )
    args = parser.parse_args()

    try:
        highlights = analyze_video(
            args.video,
            query=args.query,
            max_results=args.max_results,
            transcript_path=args.transcript,
            highlight_model_path=args.model,
            whisper_model=args.whisper_model,
        )
    except (TranscriptionError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    if not highlights:
        print("No highlights found.")
        return 0

    print("\nTop moments")
    print("-----------")
    for index, highlight in enumerate(highlights, start=1):
        start = seconds_to_timestamp(highlight.start)
        end = seconds_to_timestamp(highlight.end)
        print(f"{index}. {start} - {end} | score {highlight.score:.2f}")
        print(f"   {highlight.title}")
        print(f"   Why: {highlight.reason}")

    if args.make_reel:
        try:
            output = cut_highlight_reel(
                Path(args.video),
                highlights,
                output_path=args.output,
                padding_seconds=args.padding,
                max_clips=args.max_results,
            )
        except VideoEditingError as exc:
            print(f"\nCould not create jumpcut edit: {exc}")
            return 1

        if output:
            print(f"\nSaved jumpcut edit: {output}")
        else:
            print("\nNo jumpcut edit was created.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
