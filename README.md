# CS 4630 — The Systems Behind Modern AI

Course presentation and supporting material about the evolution of GPUs and their use in modern AI workloads.

## Repository layout

```text
assets/
  benchmark_results/  CSV inputs for generated plots
  figures/           LaTeX/TikZ, Inkscape, and separated Excalidraw sources
    excalidraw/       Editable scenes rendered by `scripts/render-excalidraw`
  images/            Rendered figures and static images from external sources
    generated/       Plot images produced exclusively by `src/cs4630/plots.py`
docs/                Research notes and references
presentation/        Beamer presentation source and PDF
src/cs4630/          Python plot generators
```

## Build the presentation

The presentation expects to be built from its own directory so its relative asset paths resolve consistently:

```sh
latexmk -cd -pdf presentation/main.tex
```

The resulting file is `presentation/main.pdf`. Temporary LaTeX files are ignored by Git.

## Render the Excalidraw figures

Install the local `draw` CLI, then edit individual scenes and regenerate their
tracked PNG outputs:

```sh
draw assets/figures/excalidraw/tiled_steps/0_load_smem.excalidraw
scripts/render-excalidraw
```

See `assets/figures/excalidraw/README.md` for the source layout and the images
whose original Excalidraw scenes still need to be recreated.

## Generate the matrix-multiplication plots

Install the Python environment and run the plotting command with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run plot-matmul
```

The command reads the CSV files in `assets/benchmark_results/` and rewrites the
plots in `assets/images/generated/`.

### Generated-image policy

Treat `assets/images/generated/` as build output owned by
`src/cs4630/plots.py`. To change one of these images—including labels,
annotations, colors, or layout—edit the Python plotting code and regenerate the
images. Do not patch a generated image manually or add a LaTeX overlay to alter
it in the presentation.

Other images may be rendered from Excalidraw sources or come from external
sources. See `assets/images/README.md` for the same distinction close to the
assets.
