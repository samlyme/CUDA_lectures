# Repository guidelines

## Generated images

Plots under `assets/images/generated/` are owned by `src/cs4630/plots.py`. To change their content or presentation, modify the plotting code and run `uv run plot-matmul`. Do not edit these image files manually or use LaTeX overlays to alter them.

Excalidraw-backed images under `assets/images/` are owned by the matching scenes in `assets/figures/excalidraw/` and rendered with `scripts/render-excalidraw`. Other images are static assets from external sources.
