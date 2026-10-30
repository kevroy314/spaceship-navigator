# What Makes a Mission Hard?

Draft paper on the spacenav gravitational navigation environment.

```
make figures   # regenerate every figure from the repo's own data
make           # build spacenav.pdf  (needs tectonic, or set TEX=pdflatex)
```

Figures are produced by `scripts/paper_figures.py`; each reads measurements
committed to this repository rather than numbers typed into the source. The
`eta` figure touches the environment and therefore a GPU; the rest are pure
analysis of stored results.

Every quantitative claim in the paper traces to a numbered decision record in
`docs/decisions/`, including the three that record retractions of our own
earlier conclusions (0025 superseded by 0032; 0028 superseded by 0030; the
fractal-objective claim retracted in 0008).
