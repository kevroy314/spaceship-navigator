# Build the paper.  Figures regenerate from the repository's own data; the
# ghost-fleet and eta figures need data/ghosts/*.json and data/pools/long.npz,
# which are produced by scripts/ghost_fleet.py and scripts/build_pools.py.
PY ?= python
TEX ?= tectonic

all: spacenav.pdf

figures:
	cd .. && $(PY) scripts/paper_figures.py --which ghosts,ablation,clock,curves,bands,eta

spacenav.pdf: spacenav.tex
	$(TEX) spacenav.tex

clean:
	rm -f spacenav.pdf *.aux *.log *.out *.bbl *.blg
.PHONY: all figures clean
