# live2d-splitter (work in progress)

Cut an anime character illustration into Live2D-style parts (transparent PNG layers + layered PSD + parts sheet),
following the Cubism guide: https://docs.live2d.com/en/cubism-editor-manual/divide-the-material/

```bash
pip install -r requirements.txt
python -m l2dsplit split character.png -o out/          # built-in CPU engine
python -m l2dsplit split character.png --labels out/labels.png   # re-cut after fixing labels.png by hand
python -m l2dsplit from-psd seethrough_output.psd -o out/        # export a See-through PSD the same way
```

Outputs: `layers/` (full-canvas PNGs), `parts/` (trimmed PNGs) + `manifest.json`, `<name>.psd`, `atlas.png`,
`labels.png` (editable colour-coded part map, see `labels_legend.png`), `recomposed.png`.

Status:
- Works: background removal, face/eye detection, part labels on clean flat-shaded art, PNG/PSD/atlas export.
- Known bug: painting of hidden areas (under front layers) leaks white / front-layer line art.
- Busy patterned outfits split poorly; use See-through (https://github.com/shitagaki-lab/see-through, GPU) for those
  and convert its PSD with `from-psd` (untested yet).
