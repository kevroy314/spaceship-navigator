#!/usr/bin/env python
"""Render the difficulty curve as an animated page: two routes per rung, side by side.

    python scripts/make_curve_page.py --curve curve_v1 --out curve.html

Reads data/curriculum/<curve>.json (from scripts/make_curve.py) and inlines it
into scripts/curve_template.html, so the page is self-contained and can be
published without a server behind it.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="curve_v1")
    ap.add_argument("--out", default="curve.html")
    args = ap.parse_args()

    data = json.loads((ROOT / "data" / "curriculum" / f"{args.curve}.json").read_text())

    # Only the representative of each rung is animated; the rest contribute their
    # outcome to the rate table and nothing else, so drop their geometry and
    # trajectories rather than shipping megabytes the page never draws.
    kept = 0
    for L in data["levels"]:
        if L.get("rep", True):
            kept += 1
            continue
        L.pop("geometry", None)
        for r in L["routes"].values():
            r.pop("path", None)
            r.pop("actions", None)

    html = (ROOT / "scripts" / "curve_template.html").read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "data" / "curriculum" / args.out
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB; {kept} animated of "
          f"{len(data['levels'])} levels)")


if __name__ == "__main__":
    main()
