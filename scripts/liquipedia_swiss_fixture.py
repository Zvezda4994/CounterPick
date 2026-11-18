"""Build a JSON fixture of a Major's Swiss stages from Liquipedia's raw wikitext.

Usage:
    uv run python scripts/liquipedia_swiss_fixture.py \
        --event "BLAST.tv Austin Major 2025" --out tests/fixtures/austin_2025.json \
        BLAST/Major/2025/Austin/Stage_1 BLAST/Major/2025/Austin/Stage_2 \
        BLAST/Major/2025/Austin/Stage_3

For each stage page the fixture records the seed order and every match per round with its
winner and the number of maps played (parsing lives in ``pickem.data.liquipedia``). Liquipedia
content is licensed CC BY-SA 3.0; the fixture keeps the source URLs for attribution. Requests are
spaced 2 s apart to respect Liquipedia's rate limits.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from pickem.data.liquipedia import fetch_wikitext, page_url, parse_stage


def build_stage(page: str, aliases: dict[str, str]) -> dict[str, Any]:
    stage = parse_stage(fetch_wikitext(page), aliases)
    return {
        "page": page,
        "url": page_url(page),
        "seeds": list(stage.seeds),
        "rounds": [
            [{"teams": [m.team1, m.team2], "winner": m.winner, "maps": m.maps} for m in matches]
            for matches in stage.rounds
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("pages", nargs="+", help="Liquipedia page paths, one per stage")
    parser.add_argument("--event", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--alias", action="append", default=[], metavar="FROM=TO", help="extra team alias"
    )
    args = parser.parse_args()
    aliases = dict(item.partition("=")[::2] for item in args.alias)

    stages = []
    for i, page in enumerate(args.pages):
        if i:
            time.sleep(2.0)
        stages.append({"stage": i + 1, **build_stage(page, aliases)})
    fixture = {
        "event": args.event,
        "source": "Liquipedia (https://liquipedia.net), CC BY-SA 3.0",
        "retrieved": date.today().isoformat(),
        "stages": stages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(stages)} stages)")


if __name__ == "__main__":
    main()
