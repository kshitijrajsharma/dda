"""End-to-end pipeline for turning (pre_img, post_img, aoi) into HF-schema deliverables.

Stages, in run order:
  prepare      fetch pre and post rasters, coregister onto the post grid
  buildings    fAIr building footprints on the coregistered pre
  damage       block-tiled pre+post damage on the post, with a per-block pre-coverage gate
  review       render crops for a workflow agent review, then apply verdicts
  deliver      write HF-schema files (buildings + damage_assessment/fair) and viz
  publish      upload the deliverables tree to a HF dataset repo
"""

from dda.pipeline.paths import PipelinePaths

__all__ = ["PipelinePaths"]
