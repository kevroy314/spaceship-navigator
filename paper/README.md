# Learning to Navigate an N-Body Field

Draft paper on the spacenav gravitational navigation environment.

```
make figures   # regenerate every figure from the repo's own data
/home/kevin/anaconda3/envs/spacenavtex/bin/tectonic --keep-logs spacenav.tex
```

`make` assumes `tectonic` is on PATH; on this machine it lives in the
`spacenavtex` conda environment. Figures are produced by
`scripts/paper_figures.py`, `scripts/fig_task.py`, `scripts/fig_phase.py`,
`scripts/fig_missions.py` and `scripts/noop_clock.py`; each reads
measurements committed to this repository rather than numbers typed into the
source. The `eta` and `phase` figures touch the environment and therefore a
GPU; the rest are pure analysis of stored results.

Every quantitative claim in the paper traces to a numbered decision record in
`docs/decisions/`. The writing policy is `STYLE.md`; `style/` holds the
voice reference, the antipattern catalogue and the citation map; `OUTLINE.md`
is the author's structure; `drafts/` keeps superseded drafts.
