# Excalidraw figure sources

Each non-archive `.excalidraw` file is the source for one rendered image under
`assets/images/`. The directory structure mirrors the output directory where
practical.

Edit a source with `draw`, for example:

```sh
draw assets/figures/excalidraw/tiled_steps/0_load_smem.excalidraw
```

Regenerate every Excalidraw image currently used by the presentation at 2x
resolution from the repository root:

```sh
scripts/render-excalidraw
```

Set `SCALE` to override the default render scale, for example
`SCALE=3 scripts/render-excalidraw`.

The files in `archive/` are the original multi-figure canvases. They are kept
only as recovery material; edit the separated sources instead.

All images that appear to have been made with Excalidraw now have an editable
source in this directory tree.
