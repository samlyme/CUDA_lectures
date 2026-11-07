# Image sources

- Excalidraw-backed images are rendered from the separated scenes in [`../figures/excalidraw/`](../figures/excalidraw/). Run `scripts/render-excalidraw` from the repository root.
- Files in [`generated/`](generated/) are produced by `src/cs4630/plots.py` from the CSV data in `assets/benchmark_results/`.
- Remaining files are static images from external or manually maintained sources.

Do not edit, annotate, or replace a generated plot by hand or with a LaTeX overlay. Change the plotting code in `src/`, then run:

```sh
uv run plot-matmul
```
