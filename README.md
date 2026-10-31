# CS 4630 — The Systems Behind Modern AI

Course presentation and supporting material about the evolution of GPUs and their use in modern AI workloads.

## Repository layout

```text
assets/
  figures/       LaTeX/TikZ and Inkscape figure sources
  images/        Raster images and generated plots
benchmarks/      Standalone benchmark programs
  matmul.cpp
docs/            Research notes and references
presentation/    Beamer presentation source and PDF
src/cs4630/      Python utilities for generating plots
```

## Build the presentation

The presentation expects to be built from its own directory so its relative asset paths resolve consistently:

```sh
latexmk -cd -pdf presentation/main.tex
```

The resulting file is `presentation/main.pdf`. Temporary LaTeX files are ignored by Git.

## Generate the matrix-multiplication plot

Install the Python environment and run the plotting command with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run plot-matmul
```

This compiles `benchmarks/matmul.cpp`, runs the benchmark, and writes `assets/images/matmul_timing.png`.

To compile the benchmark directly:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic benchmarks/matmul.cpp -o matmul
./matmul
```
