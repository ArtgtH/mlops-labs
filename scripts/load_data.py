from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import ContentTooShortError, URLError
from urllib.request import urlretrieve


DEFAULT_DATA_URL = "https://www.openml.org/data/download/22102452/dataset"


def download_dataset(url: str, output_path: Path, retries: int = 5) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            urlretrieve(url, tmp_path)
            tmp_path.replace(output_path)
            return
        except (ContentTooShortError, URLError, TimeoutError) as error:
            last_error = error
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(attempt * 2)

    raise RuntimeError(f"Failed to download dataset after {retries} attempts") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_DATA_URL)
    parser.add_argument("--output", default="data/raw/creditcard.arff")
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(args.url, Path(args.output), retries=args.retries)


if __name__ == "__main__":
    main()
