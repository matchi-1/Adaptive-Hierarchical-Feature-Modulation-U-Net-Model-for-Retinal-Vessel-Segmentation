
import numpy as np
from retina_biomarkers import (
    to_bool_mask, skeletonize_mask, distance_transform, build_skeleton_graph,
    sample_width_along_skeleton, area_density, length_density, caliber_stats,
    tortuosity_stats, fractal_dimension_boxcount, junction_metrics,
    branching_and_bifurcation_angles, gap_metrics, ring_masks_from_disc, metrics_by_rings
)

# Create a simple synthetic vessel mask (512x512) with a main diagonal and two branches
H=W=512
mask = np.zeros((H,W), dtype=np.uint8)
for i in range(50, 462):
    mask[i, i] = 1
for i in range(200, 280):
    mask[300, i] = 1
for i in range(230, 300):
    mask[i, 300] = 1

# Precompute geometry
skel = skeletonize_mask(mask)
dist = distance_transform(mask)
graph = build_skeleton_graph(skel)
widths = sample_width_along_skeleton(dist, graph)

# Global metrics
print("Area density:", area_density(mask))
print("Length density:", length_density(graph, mask.shape))
print("Caliber stats:", caliber_stats(widths))
print("Tortuosity:", tortuosity_stats(graph))
print("Fractal D (skeleton):", fractal_dimension_boxcount(skel))

# Topology & continuity
print("Junction metrics:", junction_metrics(graph, mask.shape))
print("Angles:", branching_and_bifurcation_angles(graph))
print("Gaps:", gap_metrics(mask, graph, max_gap_px=12))

# Regional metrics (assume disc at center with PD=100 px)
disc_center=(H/2, W/2); PD_px=100.0
rings = metrics_by_rings(mask, graph, widths, disc_center, PD_px)
print("Rings:", rings)
