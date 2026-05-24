from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


TEXT_EXTENSIONS = {".srt", ".vtt", ".json", ".txt", ".md"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download transcript pages/files from a URL list."
    )
    parser.add_argument("--urls", default="data/transcript_urls.txt")
    parser.add_argument("--output-dir", default="data/raw_transcripts")
    args = parser.parse_args()

    urls_path = Path(args.urls)
    if not urls_path.exists():
        print(f"Missing URL list: {urls_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_path = output_dir / "sources.jsonl"

    urls = [
        line.strip()
        for line in urls_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        print(f"No URLs found in {urls_path}.")
        return 1

    with sources_path.open("a", encoding="utf-8") as sources_file:
        for url in urls:
            destination = download_transcript(url, output_dir)
            sources_file.write(json.dumps({"url": url, "path": str(destination)}) + "\n")
            print(f"Saved {url} -> {destination}")

    return 0


def download_transcript(url: str, output_dir: Path) -> Path:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "ai-video-highlighter/0.1"},
    )
    response.raise_for_status()

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    base_name = _safe_name(parsed.netloc + parsed.path)

    if suffix in TEXT_EXTENSIONS:
        destination = output_dir / f"{base_name}{suffix}"
        destination.write_bytes(response.content)
        return destination

    text = _extract_main_text(response.text)
    destination = output_dir / f"{base_name}.txt"
    destination.write_text(text, encoding="utf-8")
    return destination


def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [line.strip() for line in main.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value[:120] or "transcript"


if __name__ == "__main__":
    raise SystemExit(main())

