# Image sources

- Files in this directory are static images from external or manually maintained sources.
- Files in [`generated/`](generated/) are produced by `src/cs4630/plots.py` from the CSV data in `assets/benchmark_results/`.

Do not edit, annotate, or replace a generated plot by hand or with a LaTeX overlay. Change the plotting code in `src/`, then run:

```sh
uv run plot-matmul
```
