#!/usr/bin/env python
"""Render the causal atlas: one card per lesson, with its volume, necessity and alpha.

    python scripts/make_causal_page.py --causal causal_v1 --out causal.html

Geometry is dropped from the payload: this page draws parameter space, not
trajectories, so the level layouts would be dead weight.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", default="causal_v1")
    ap.add_argument("--out", default="causal.html")
    args = ap.parse_args()

    data = json.loads((ROOT / "data" / "curriculum" / f"{args.causal}.json").read_text())
    for L in data["lessons"]:
        L.pop("geometry", None)
        if L.get("plan"):
            L["plan"].pop("path", None)
    html = (ROOT / "scripts" / "causal_template.html").read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "data" / "curriculum" / args.out
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(data['lessons'])} lessons)")


if __name__ == "__main__":
    main()
