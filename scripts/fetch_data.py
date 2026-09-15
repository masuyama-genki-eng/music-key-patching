"""Fetch optional external corpora into data/ and record the fetched revisions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPORA = {
    "bach": ("bach-370-chorales", "https://github.com/craigsapp/bach-370-chorales.git"),
    "analyses": ("When-in-Rome", "https://github.com/MarkGotham/When-in-Rome.git"),
    "pop909": ("POP909-CL", "https://github.com/AndyWeasley2004/POP909-CL-Dataset.git"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", nargs="+", choices=list(CORPORA), default=list(CORPORA)
    )
    parser.add_argument("--root", type=Path, default=REPO / "data")
    args = parser.parse_args()
    git = shutil.which("git")
    if git is None:
        parser.error("Git must be installed to fetch datasets.")
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for corpus in args.corpus:
        directory, url = CORPORA[corpus]
        destination = args.root.resolve() / directory
        if not destination.exists():
            subprocess.run(
                [git, "clone", "--depth", "1", "--", url, str(destination)], check=True
            )
        elif not (destination / ".git").exists():
            parser.error(
                f"Existing {destination} is not a Git checkout; it was left untouched."
            )
        # Existing clones are recorded without updating or checking out another revision.
        revision = subprocess.run(
            [git, "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        origin = subprocess.run(
            [git, "-C", str(destination), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if origin != url:
            parser.error(
                f"{destination}: origin differs from the expected public repository."
            )
        manifest[corpus] = {"source": url, "commit": revision, "directory": directory}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{corpus}: {revision}")


if __name__ == "__main__":
    main()
