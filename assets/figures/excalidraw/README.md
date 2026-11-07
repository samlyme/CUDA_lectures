# Excalidraw figure sources

Each non-archive `.excalidraw` file is the source for one rendered image under
`assets/images/`. The directory structure mirrors the output directory where
practical.

Edit a source with `draw`, for example:

```sh
draw assets/figures/excalidraw/tiled_steps/0_load_smem.excalidraw
```

Regenerate every tracked Excalidraw image from the repository root:

```sh
scripts/render-excalidraw
```

The files in `archive/` are the original multi-figure canvases. They are kept
only as recovery material; edit the separated sources instead.

## Images still missing an editable source

These images appear to have been made with Excalidraw, but no complete editable
scene was recoverable from the saved canvases:

- `assets/images/data_use_thread_coarsening.png`
- `assets/images/tiled_steps/2_repeat_1.png`

Save replacements as matching `.excalidraw` files in this directory tree, then
add them to `scripts/render-excalidraw`.
