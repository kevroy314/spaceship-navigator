"""VTT -> deduped, timestamped plain text (auto-captions repeat rolling lines)."""
import re, sys
def parse(path):
    out, last = [], ""
    for block in open(path).read().split("\n\n"):
        m = re.search(r"(\d\d:\d\d:\d\d)\.\d+ --> ", block)
        if not m: continue
        text = " ".join(l for l in block.split("\n")[1:] if l and "-->" not in l)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text or text == last: continue
        # drop lines that are just the tail of the previous rolling caption
        if last and text.startswith(last[-40:]): text = text[len(last[-40:]):].strip()
        if text: out.append((m.group(1), text)); last = text
    return out
for p in sys.argv[1:]:
    lines = parse(p)
    with open(p.replace(".vtt", ".txt"), "w") as f:
        for t, s in lines: f.write(f"[{t}] {s}\n")
    print(p, "->", len(lines), "lines")
