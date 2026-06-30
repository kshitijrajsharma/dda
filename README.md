# dda

Building-level disaster damage assessment with a frozen DINOv3 siamese encoder and a trainable decoder.

The model takes building footprints and post-disaster satellite imagery (optionally pre-disaster
imagery as well) and assigns each building a damage level on the xBD Joint Damage Scale:
`no-damage`, `minor-damage`, `major-damage`, `destroyed`. Each prediction carries a calibrated
confidence so low-certainty buildings can be flagged for human review.

## Approach

- **Backbone**: DINOv3 ViT-L/16, frozen, loaded through TerraTorch. The default checkpoint is the
  satellite variant (`sat493m`, Maxar 0.6 m), with the web variant (`lvd1689m`) available for
  comparison.
- **Siamese pre/post**: pre and post tiles share the backbone. Features are fused per tap and the
  model is trained with modality dropout, so one checkpoint serves both the pre+post and the
  post-only regime. Late feature fusion tolerates the misregistration that comes from footprints
  drawn on different imagery than the post-disaster image.
- **Decoder and heads**: a TerraTorch UperNet decoder feeds a binary localization head and a
  4-class damage head. The damage loss is class-weighted cross-entropy plus a squared
  Earth-Mover term, which penalises predictions by their ordinal distance from the truth.
- **Per-building output**: the dense damage map is pooled inside each input footprint to produce
  one class and one confidence per building.

## Results

Per-building damage F1 on the xBD (xView2) held-out test events (palu-tsunami, joplin-tornado;
6,415 buildings), pooled inside each footprint:

| class | F1 |
| --- | --- |
| no-damage | 0.92 |
| minor-damage | 0.61 |
| major-damage | 0.49 |
| destroyed | 0.91 |
| macro | 0.73 |
| harmonic (xView2 damage F1) | 0.68 |

Splits are held out by disaster event, so these measure cross-event generalisation.

Under the official xView2 pixel metric (`0.3 * localization F1 + 0.7 * damage F1`) on the public
xBD test split (933 images): localization 0.82, damage 0.73, overall **0.76**.

## Data

Training reads xBD (xView2) from `kshitijrajsharma/xview2-xbd` on the Hugging Face Hub (set by
`dataset_repo`). To rebuild your own copy, `scripts/build_xbd_dataset.py` assembles the pre/post
tiles, damage rasters, and polygons from a non-gated mirror (`--mirror`) or a local extraction
(`--xbd-root`). Splits are held out by disaster event, not by random tile, so validation measures
cross-event generalisation.

```
just data <your-user>/xbd-damage      # download mirror, build, push to your HF
```

## Workflow

```
just setup                       # install deps and hooks
just train                       # train the model
just predict <post.tif> <buildings.geojson> <out.geojson> <pre.tif>   # per-building damage
just export <ckpt> <model.onnx>  # ONNX export for serving
just evaluate <ckpt> test        # object-level damage F1 on the held-out test events
```

Passing a pre-disaster GeoTIFF turns on the change signal; it is co-registered onto the post grid
and both scenes are radiometrically normalized before inference.

## Stack

`uv`, PyTorch Lightning, TerraTorch, OmegaConf, rasterio, geopandas. Lint and type-check with
`ruff` and `ty` via `just lint`.
