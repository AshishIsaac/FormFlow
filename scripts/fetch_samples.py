"""Download the openly licensed exercise videos FormFlow's samples are made from (Wikimedia Commons).

    python scripts/fetch_samples.py OUT_DIR [title ...]

Records each file's licence, author and page next to it (``<name>.json``), so the provenance of every
committed sample is reproducible. Public-domain US Army training videos and CC BY / CC0 clips only.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "FormFlow-sample-fetch/1.0 (https://github.com/AshishIsaac/FormFlow; portfolio project)"
API = "https://commons.wikimedia.org/w/api.php"

TITLES = [
    "Squat - exercise demonstration video.webm",
    "Deadlift - exercise demonstration video.webm",
    "Strength Training Circuit- Forward Lunge.webm",
    "Strength Training Circuit- Overhead Push Press.webm",
    "Army Combat Fitness Test- Hand-Release Push-Up (HRP) (Event 3).webm",
    "Free Weight Training Assistive Exercise- Assistive Sumo Deadlift.webm",
    "Jumping jack slow motion.webm",
    "Pull-ups - exercise demonstration video.webm",
    "Bench press - exercise demonstration video.webm",
    "Shoulder press - exercise demonstration video.webm",
    "Army Combat Fitness Test- 3 Repetition Maximum Deadlift (MDL) (Event 1).webm",
]


def _get(url: str, retries: int = 5) -> bytes:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # rate limit / hiccup: back off
            wait = 5 * (i + 1)
            print(f"  retry in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def info(title: str) -> dict:
    q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": "File:" + title, "prop": "imageinfo",
                                "iiprop": "url|extmetadata|size"})
    d = json.loads(_get(f"{API}?{q}"))
    page = next(iter(d["query"]["pages"].values()))
    ii = page["imageinfo"][0]
    m = ii.get("extmetadata", {})
    clean = lambda k: re.sub(r"<[^>]+>", "", m.get(k, {}).get("value", "")).strip()
    return {"title": title, "url": ii["url"], "page": ii["descriptionurl"], "license": clean("LicenseShortName"),
            "license_url": clean("LicenseUrl"), "author": clean("Artist")[:200], "credit": clean("Credit")[:200],
            "bytes": ii.get("size")}


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else Path("commons")
    out.mkdir(parents=True, exist_ok=True)
    for t in argv[1:] or TITLES:
        name = re.sub(r"[^A-Za-z0-9.-]+", "_", t)
        dest = out / name
        meta = out / (name + ".json")
        if dest.is_file() and meta.is_file():
            print(f"have {name}")
            continue
        i = info(t)
        time.sleep(1.5)
        print(f"{name}  [{i['license']}]  {i['bytes'] / 1e6:.1f} MB", flush=True)
        dest.write_bytes(_get(i["url"]))
        meta.write_text(json.dumps(i, indent=1), encoding="utf-8")
        time.sleep(1.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
