from __future__ import annotations

import argparse
import json

from highlight_model import train_highlight_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a small personal lecture-highlight model."
    )
    parser.add_argument("--input-dir", default="data/raw_transcripts")
    parser.add_argument("--output", default="models/highlight_model.joblib")
    parser.add_argument("--examples", default="data/training_examples.csv")
    args = parser.parse_args()

    metadata = train_highlight_model(
        input_dir=args.input_dir,
        output_path=args.output,
        examples_path=args.examples,
    )

    print("Trained highlight model")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

